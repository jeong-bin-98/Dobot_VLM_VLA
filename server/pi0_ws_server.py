#!/usr/bin/env python3
"""
Pi0/Pi0-FAST WebSocket 추론 서버

Persistent connection으로 매 요청마다 TCP 재연결 없이 추론.
JSON으로 이미지/state 수신, delta actions 반환.

    PI0_POLICY_TYPE=pi0_fast \\
    PI0_MODEL_PATH=./outputs/pi0fast_dobot/checkpoints/last/pretrained_model \\
    python pi0_ws_server.py
"""

import os
import sys
import time
import json
import base64
import logging
import asyncio

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from safetensors.torch import load_file

# Configuration
MODEL_PATH = os.environ.get(
    "PI0_MODEL_PATH",
    "./outputs/pi0fast_dobot/checkpoints/last/pretrained_model",
)
POLICY_TYPE = os.environ.get("PI0_POLICY_TYPE", "pi0_fast")  # "pi0" or "pi0_fast"
HOST = "0.0.0.0"
PORT = int(os.environ.get("PI0_PORT", "8765"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pi0_ws")
# Normalizer (preprocessor / postprocessor)
class ModelNormalizer:
    def __init__(self, model_path):
        # Input normalization
        for name in [
            "policy_preprocessor_step_5_normalizer_processor.safetensors",
            "policy_preprocessor_step_0_normalizer_processor.safetensors",
        ]:
            path = os.path.join(model_path, name)
            if os.path.exists(path):
                data = load_file(path)
                self.state_mean = data["observation.state.mean"].numpy()
                self.state_std = np.where(data["observation.state.std"].numpy() < 1e-6, 1.0,
                                          data["observation.state.std"].numpy())
                break
        else:
            self.state_mean, self.state_std = np.zeros(5), np.ones(5)

        # Output un-normalization
        post = os.path.join(model_path, "policy_postprocessor_step_0_unnormalizer_processor.safetensors")
        if os.path.exists(post):
            data = load_file(post)
            self.action_mean = data["action.mean"].numpy()
            self.action_std = np.where(data["action.std"].numpy() < 1e-6, 1.0,
                                       data["action.std"].numpy())
        else:
            self.action_mean, self.action_std = np.zeros(5), np.ones(5)

    def normalize_state(self, raw):
        return (np.array(raw, dtype=np.float32) - self.state_mean) / self.state_std

    def unnormalize_action(self, norm):
        return norm * self.action_std + self.action_mean
# Global state
app = FastAPI(title="Pi0 WebSocket Streaming Server")
policy = None
normalizer = None
# Image utilities
def decode_b64_to_tensor(b64_str: str) -> torch.Tensor:
    """base64 JPEG -> [1, 3, H, W] float32 tensor."""
    import cv2
    buf = np.frombuffer(base64.b64decode(b64_str), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Image decode failed")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return (
        torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0)
        .unsqueeze(0).to(DEVICE)
    )
# Model loading
@app.on_event("startup")
def load_model():
    global policy, normalizer
    logger.info(f"Loading {POLICY_TYPE}: {MODEL_PATH}")
    t0 = time.time()

    if POLICY_TYPE == "pi0_fast":
        try:
            from lerobot.policies.pi0_fast.modeling_pi0fast import PI0FastPolicy
        except ImportError:
            from lerobot.common.policies.pi0_fast.modeling_pi0fast import PI0FastPolicy
        policy = PI0FastPolicy.from_pretrained(MODEL_PATH)
    else:
        try:
            from lerobot.policies.pi0.modeling_pi0 import PI0Policy
        except ImportError:
            from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
        policy = PI0Policy.from_pretrained(MODEL_PATH)

    policy.eval().to(DEVICE)
    normalizer = ModelNormalizer(MODEL_PATH)
    logger.info(f"Loaded in {time.time()-t0:.1f}s | {POLICY_TYPE} on {DEVICE}")
# Inference
def run_inference(data: dict) -> dict:
    """Single inference pass. Returns actions dict."""
    t0 = time.time()

    img_top = decode_b64_to_tensor(data["image_top"])
    img_wrist = decode_b64_to_tensor(data["image_wrist"])

    norm_state = normalizer.normalize_state(data["state"])
    state_t = torch.tensor(norm_state, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    observation = {
        "observation.images.top": img_top,
        "observation.images.wrist": img_wrist,
        "observation.state": state_t,
    }
    if data.get("task"):
        observation["task"] = data["task"]

    with torch.no_grad():
        action = policy.select_action(observation)

    if isinstance(action, torch.Tensor):
        action = action.cpu().numpy()
    if action.ndim == 1:
        action = action.reshape(1, -1)

    # Un-normalize -> real delta (mm)
    chunk = min(data.get("chunk_size", 1), action.shape[0])
    actions = [normalizer.unnormalize_action(action[i]).tolist() for i in range(chunk)]

    dt_ms = (time.time() - t0) * 1000
    return {"actions": actions, "inference_ms": round(dt_ms, 1)}
# WebSocket endpoint -- Half-Duplex streaming
@app.websocket("/ws")
async def ws_inference(ws: WebSocket):
    """
    Half-duplex WebSocket loop:
      1. Client sends observation (JSON)
      2. Server runs inference
      3. Server returns actions (JSON)
      4. Client executes, then sends next observation
      ... repeat over persistent connection
    """
    await ws.accept()
    client = ws.client
    logger.info(f"WS connected: {client}")
    cycle = 0

    try:
        while True:
            # Receive observation from client
            raw = await ws.receive_text()
            data = json.loads(raw)

            # Run inference
            result = run_inference(data)

            # Send actions back
            await ws.send_text(json.dumps(result))

            if cycle % 10 == 0:
                a = result["actions"][0]
                logger.info(
                    f"[{cycle}] {result['inference_ms']:.0f}ms | "
                    f"delta=[{a[0]:+.1f},{a[1]:+.1f},{a[2]:+.1f}]"
                )
            cycle += 1

    except WebSocketDisconnect:
        logger.info(f"WS disconnected: {client}")
    except Exception as e:
        logger.error(f"WS error: {e}")
# Health endpoint (HTTP, for monitoring)
@app.get("/health")
def health():
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    return {
        "status": "ok",
        "policy_type": POLICY_TYPE,
        "model_loaded": policy is not None,
        "device": DEVICE,
        "gpu": gpu_name,
        "ws_endpoint": f"ws://{HOST}:{PORT}/ws",
    }
if __name__ == "__main__":
    print(f"""
 ============================================================
  Pi0 WebSocket Streaming Server
  Policy: {POLICY_TYPE}
  WS:     ws://{HOST}:{PORT}/ws
  Health: http://{HOST}:{PORT}/health
 ============================================================
""")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
