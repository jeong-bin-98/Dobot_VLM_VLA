#!/usr/bin/env python3
"""
Pi0/Pi0-FAST HTTP 추론 서버

이미지 + state + 언어 명령 수신, delta actions 반환.

    python pi0_server.py
    curl http://localhost:8000/health
""""""

import io
import os
import time
import json
import base64
import logging
import numpy as np
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from safetensors.torch import load_file

# Configuration -- Update MODEL_PATH and POLICY_TYPE for your setup
MODEL_PATH = os.environ.get(
    "PI0_MODEL_PATH",
    "./outputs/pi0_dobot/checkpoints/last/pretrained_model",
)
# "pi0" = flow-matching (original), "pi0_fast" = autoregressive FAST tokenizer (5x faster training)
POLICY_TYPE = os.environ.get("PI0_POLICY_TYPE", "pi0_fast")
HOST = "0.0.0.0"
PORT = 8000
DEVICE = "cuda"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pi0_server")

# FastAPI
app = FastAPI(title="Pi0 Inference Server", version="1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

policy = None
normalizer = None
# Normalization
class ModelNormalizer:
    def __init__(self, model_path):
        logger.info(f"정규화 통계 로드: {model_path}")

        # Input normalization - Pi0 uses step_5, Pi0-FAST uses step_0
        for name in [
            "policy_preprocessor_step_5_normalizer_processor.safetensors",
            "policy_preprocessor_step_0_normalizer_processor.safetensors",
        ]:
            path = os.path.join(model_path, name)
            if os.path.exists(path):
                data = load_file(path)
                self.state_mean = data["observation.state.mean"].numpy()
                self.state_std  = data["observation.state.std"].numpy()
                self.state_std  = np.where(self.state_std < 1e-6, 1.0, self.state_std)
                logger.info(f"   state mean: {self.state_mean}")
                logger.info(f"   state std:  {self.state_std}")
                break
        else:
            logger.warning("   preprocessor 없음")
            self.state_mean = np.zeros(5)
            self.state_std  = np.ones(5)

        # 출력 역정규화
        path = os.path.join(model_path, "policy_postprocessor_step_0_unnormalizer_processor.safetensors")
        if os.path.exists(path):
            data = load_file(path)
            self.action_mean = data["action.mean"].numpy()
            self.action_std  = data["action.std"].numpy()
            self.action_std  = np.where(self.action_std < 1e-6, 1.0, self.action_std)
            logger.info(f"   action mean: {self.action_mean}")
            logger.info(f"   action std:  {self.action_std}")
        else:
            logger.warning("   postprocessor 없음")
            self.action_mean = np.zeros(5)
            self.action_std  = np.ones(5)

    def normalize_state(self, raw):
        return (np.array(raw, dtype=np.float32) - self.state_mean) / self.state_std

    def unnormalize_action(self, norm):
        return norm * self.action_std + self.action_mean
# 요청 / 응답
class InferenceRequest(BaseModel):
    image_top: str           # base64 JPEG
    image_wrist: str         # base64 JPEG
    state: list[float]       # [x, y, z, r, gripper] raw
    language_instruction: str = ""
    chunk_size: int = 2      # 사용할 액션 스텝 수 (1~50)

class InferenceResponse(BaseModel):
    actions: list[list[float]]  # chunk_size × 5 (역정규화된 real delta)
    raw_actions: list[float]    # 첫 스텝 raw 출력 (디버깅)
    inference_time_ms: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    gpu_name: str
    gpu_memory_used_gb: float
    gpu_memory_total_gb: float
# 유틸
def decode_b64_image(b64: str) -> np.ndarray:
    import cv2
    buf = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("이미지 디코딩 실패")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # HWC, RGB, uint8

def img_to_tensor(img: np.ndarray) -> torch.Tensor:
    """[H,W,3] uint8 -> [1,3,H,W] float32 0~1"""
    return (
        torch.from_numpy(img).permute(2, 0, 1).float().div(255.0)
        .unsqueeze(0).to(DEVICE)
    )
# 모델 로드
@app.on_event("startup")
def startup_load_model():
    global policy, normalizer

    logger.info(f"Loading {POLICY_TYPE} model: {MODEL_PATH}")
    t0 = time.time()

    if POLICY_TYPE == "pi0_fast":
        # Pi0-FAST: autoregressive with FAST action tokenizer (5x faster training)
        try:
            from lerobot.policies.pi0_fast.modeling_pi0fast import PI0FastPolicy
        except ImportError:
            from lerobot.common.policies.pi0_fast.modeling_pi0fast import PI0FastPolicy
        policy = PI0FastPolicy.from_pretrained(MODEL_PATH)
        logger.info("   Policy type: Pi0-FAST (autoregressive)")
    else:
        # Pi0: flow-matching (original)
        try:
            from lerobot.policies.pi0.modeling_pi0 import PI0Policy
        except ImportError:
            from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
        policy = PI0Policy.from_pretrained(MODEL_PATH)
        logger.info("   Policy type: Pi0 (flow-matching)")
    policy.eval()
    policy.to(DEVICE)

    normalizer = ModelNormalizer(MODEL_PATH)

    dt = time.time() - t0
    logger.info(f"로드 완료 ({dt:.1f}s)")
    if torch.cuda.is_available():
        g = torch.cuda.get_device_properties(0)
        logger.info(f"{g.name}  {g.total_mem/1e9:.0f}GB")
# 엔드포인트
@app.get("/health", response_model=HealthResponse)
def health():
    gn, mu, mt = "", 0.0, 0.0
    if torch.cuda.is_available():
        gn = torch.cuda.get_device_name(0)
        mu = torch.cuda.memory_allocated(0) / 1e9
        mt = torch.cuda.get_device_properties(0).total_mem / 1e9
    return HealthResponse(
        status="ok", model_loaded=policy is not None,
        device=DEVICE, gpu_name=gn,
        gpu_memory_used_gb=round(mu, 2), gpu_memory_total_gb=round(mt, 2),
    )
@app.post("/predict", response_model=InferenceResponse)
def predict(req: InferenceRequest):
    """
    Pi0 추론

    흐름:
    raw_state -> normalize -> Pi0 model -> unnormalize -> real_delta
    """
    if policy is None:
        raise HTTPException(503, "모델 미로드")

    t0 = time.time()
    try:
        # 1. 이미지
        img_top   = decode_b64_image(req.image_top)
        img_wrist = decode_b64_image(req.image_wrist)

        # 2. state 정규화
        norm_state = normalizer.normalize_state(req.state)
        state_t = torch.tensor(norm_state, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        # 3. observation 딕셔너리
        observation = {
            "observation.images.top":   img_to_tensor(img_top),
            "observation.images.wrist": img_to_tensor(img_wrist),
            "observation.state":        state_t,
        }

        # 언어 명령 (Pi0 내부 tokenizer 사용)
        if req.language_instruction:
            observation["task"] = req.language_instruction

        # 4. 추론
        with torch.no_grad():
            action = policy.select_action(observation)

        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        if action.ndim == 1:
            action = action.reshape(1, -1)

        # 5. 역정규화
        raw_out = action[0].tolist()
        chunk = min(req.chunk_size, action.shape[0])
        actions = [normalizer.unnormalize_action(action[i]).tolist() for i in range(chunk)]

        dt_ms = (time.time() - t0) * 1000
        logger.info(
            f"{dt_ms:.0f}ms | "
            f"Δ[0]=[{actions[0][0]:+.1f},{actions[0][1]:+.1f},{actions[0][2]:+.1f}] | "
            f"\"{req.language_instruction[:25]}\""
        )
        return InferenceResponse(actions=actions, raw_actions=raw_out, inference_time_ms=round(dt_ms, 1))

    except Exception as e:
        logger.error(f"{e}")
        import traceback; traceback.print_exc()
        raise HTTPException(500, str(e))
@app.get("/model_info")
def model_info():
    p = Path(MODEL_PATH) / "config.json"
    return json.load(open(p)) if p.exists() else {"error": "not found"}
if __name__ == "__main__":
    print(f"""
+-----------------------------------------------------------+
|  Pi0 / Pi0-FAST Inference Server                         |
|  Policy: {POLICY_TYPE:<48}|
|  GET  /health     Server status                          |
|  POST /predict    Inference                              |
|  GET  /model_info Model config                           |
+-----------------------------------------------------------+
""")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
