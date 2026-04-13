#!/usr/bin/env python3
"""
Pi0/Pi0-FAST HTTP 추론 서버

이미지 + state + 언어 명령 수신, delta actions 반환.

    python pi0_server.py
    curl http://localhost:8000/health
"""

import os
import time
import json
import base64
import logging
import tempfile
import numpy as np
from pathlib import Path

import torch
import draccus
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from safetensors.torch import load_file

# Configuration -- Update MODEL_PATH and POLICY_TYPE for your setup
MODEL_PATH = os.environ.get(
    "PI0_MODEL_PATH",
    "./outputs/pi0fast_dobot_testv2/checkpoints/000100/pretrained_model",
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
paligemma_tokenizer = None
tokenizer_max_length = None


# Normalization
class ModelNormalizer:
    def __init__(self, model_path):
        logger.info(f"정규화 통계 로드: {model_path}")

        import glob

        # Input normalization — glob으로 자동 검색
        pre_files = sorted(glob.glob(os.path.join(model_path, "policy_preprocessor_step_*_normalizer_processor.safetensors")))
        if not pre_files:
            raise FileNotFoundError(
                f"Preprocessor not found in {model_path}. "
                f"Expected: policy_preprocessor_step_*_normalizer_processor.safetensors"
            )
        data = load_file(pre_files[0])
        self.state_mean = data["observation.state.mean"].numpy()
        self.state_std = data["observation.state.std"].numpy()
        self.state_std = np.where(self.state_std < 1e-6, 1.0, self.state_std)
        logger.info(f"   preprocessor: {os.path.basename(pre_files[0])}")
        logger.info(f"   state mean: {self.state_mean}")
        logger.info(f"   state std:  {self.state_std}")

        # 출력 역정규화 — glob으로 자동 검색
        post_files = sorted(glob.glob(os.path.join(model_path, "policy_postprocessor_step_*_unnormalizer_processor.safetensors")))
        if not post_files:
            raise FileNotFoundError(
                f"Postprocessor not found in {model_path}. "
                f"Expected: policy_postprocessor_step_*_unnormalizer_processor.safetensors"
            )
        data = load_file(post_files[0])
        self.action_mean = data["action.mean"].numpy()
        self.action_std = data["action.std"].numpy()
        self.action_std = np.where(self.action_std < 1e-6, 1.0, self.action_std)
        logger.info(f"   postprocessor: {os.path.basename(post_files[0])}")
        logger.info(f"   action mean: {self.action_mean}")
        logger.info(f"   action std:  {self.action_std}")

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


def load_ft_config(model_path: str, policy_type: str = "pi0_fast"):
    """fine-tuned config.json 로드 (draccus 호환)"""
    if policy_type == "pi0_fast":
        from lerobot.policies.pi0_fast.configuration_pi0_fast import PI0FastConfig as ConfigClass
    else:
        from lerobot.policies.pi0.configuration_pi0 import PI0Config as ConfigClass

    with open(Path(model_path) / "config.json") as f:
        cfg_data = json.load(f)
    cfg_data.pop("type", None)

    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".json") as f:
        json.dump(cfg_data, f)
        tmpf = f.name

    with draccus.config_type("json"):
        return draccus.parse(ConfigClass, tmpf, args=[])


# 모델 로드
@app.on_event("startup")
def startup_load_model():
    global policy, normalizer, paligemma_tokenizer, tokenizer_max_length

    logger.info(f"Loading {POLICY_TYPE} model: {MODEL_PATH}")
    t0 = time.time()

    if POLICY_TYPE == "pi0_fast":
        from lerobot.policies.pi0_fast.modeling_pi0_fast import PI0FastPolicy
        from peft import PeftModel

        # 1. base 모델 로드
        base_policy = PI0FastPolicy.from_pretrained("lerobot/pi0fast-base")

        # 2. 토크나이저 저장 (merge 후 사라지므로)
        paligemma_tokenizer = base_policy._paligemma_tokenizer
        tokenizer_max_length = base_policy.config.tokenizer_max_length

        # 3. LoRA 어댑터 로드 + merge
        policy = PeftModel.from_pretrained(base_policy, MODEL_PATH)
        policy = policy.merge_and_unload()
        policy._tokenizer_max_length = tokenizer_max_length
        policy._paligemma_tokenizer = paligemma_tokenizer

        # 4. fine-tuned config 로드
        ft_config = load_ft_config(MODEL_PATH)
        policy.config = ft_config
        logger.info(f"   Fine-tuned config: cameras={list(ft_config.input_features.keys())}")
        logger.info("   Policy type: Pi0-FAST (autoregressive + LoRA merged)")
    else:
        from lerobot.policies.pi0.modeling_pi0 import PI0Policy
        from transformers import AutoTokenizer

        adapter_path = Path(MODEL_PATH) / "adapter_config.json"
        if adapter_path.exists():
            from peft import PeftModel

            base_policy = PI0Policy.from_pretrained("lerobot/pi0_base")

            policy = PeftModel.from_pretrained(base_policy, MODEL_PATH)
            policy = policy.merge_and_unload()

            ft_config = load_ft_config(MODEL_PATH, policy_type="pi0")
            policy.config = ft_config
            logger.info(f"   Fine-tuned config: cameras={list(ft_config.input_features.keys())}")
            logger.info("   Policy type: Pi0 (flow-matching + LoRA merged)")
        else:
            policy = PI0Policy.from_pretrained(MODEL_PATH)
            logger.info("   Policy type: Pi0 (flow-matching)")

        # pi0는 모델에 tokenizer가 없으므로 직접 로드
        # (processor_pi0.py의 TokenizerProcessorStep과 동일한 설정)
        tokenizer_max_length = policy.config.tokenizer_max_length
        paligemma_tokenizer = AutoTokenizer.from_pretrained(
            "google/paligemma-3b-pt-224",
            padding_side="right",
        )

        # flow-matching denoising 스텝 수 조절 (기본 10, torch.compile과 함께 사용)
        inference_steps = int(os.environ.get("PI0_INFERENCE_STEPS", 10))
        policy.config.num_inference_steps = inference_steps
        logger.info(f"   num_inference_steps: {inference_steps}")

    policy.eval().to(DEVICE)

    # Ampere GPU TF32 활성화 (matmul 1.5~2배 가속, 품질 저하 거의 없음)
    torch.set_float32_matmul_precision("high")

    # torch.compile로 select_action 가속 (첫 호출 시 컴파일 오버헤드 있음)
    policy.select_action = torch.compile(
        policy.select_action,
        mode="reduce-overhead",
        fullgraph=False,
    )
    logger.info("   torch.compile 적용: mode=reduce-overhead")

    normalizer = ModelNormalizer(MODEL_PATH)

    # Warmup: torch.compile 첫 호출 지연 제거 (dummy 추론 1회)
    logger.info("Warmup 추론 중... (torch.compile 컴파일)")
    try:
        warmup_t0 = time.time()
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy_state = np.zeros(5, dtype=np.float32)
        norm_state = normalizer.normalize_state(dummy_state)
        state_t = torch.tensor(norm_state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        dummy_obs = {
            "observation.images.top": img_to_tensor(dummy_img),
            "observation.images.wrist": img_to_tensor(dummy_img),
            "observation.state": state_t,
        }
        dummy_lang = "pick up the object"
        if POLICY_TYPE == "pi0" and not dummy_lang.endswith("\n"):
            dummy_lang = dummy_lang + "\n"
        tokenized = paligemma_tokenizer(
            dummy_lang,
            return_tensors="pt",
            padding="max_length",
            max_length=tokenizer_max_length,
            truncation=True,
        )
        dummy_obs["observation.language.tokens"] = tokenized.input_ids.to(DEVICE)
        dummy_obs["observation.language.attention_mask"] = tokenized.attention_mask.to(DEVICE).bool()
        with torch.no_grad():
            _ = policy.select_action(dummy_obs)
        logger.info(f"Warmup 완료 ({time.time() - warmup_t0:.1f}s)")
    except Exception as e:
        logger.warning(f"Warmup 실패 (무시하고 계속): {e}")

    dt = time.time() - t0
    logger.info(f"로드 완료 ({dt:.1f}s)")
    if torch.cuda.is_available():
        g = torch.cuda.get_device_properties(0)
        logger.info(f"{g.name}  {g.total_memory / 1e9:.0f}GB")


# 엔드포인트
@app.get("/health", response_model=HealthResponse)
def health():
    gn, mu, mt = "", 0.0, 0.0
    if torch.cuda.is_available():
        gn = torch.cuda.get_device_name(0)
        mu = torch.cuda.memory_allocated(0) / 1e9
        mt = torch.cuda.get_device_properties(0).total_memory / 1e9
    return HealthResponse(
        status="ok", model_loaded=policy is not None,
        device=DEVICE, gpu_name=gn,
        gpu_memory_used_gb=round(mu, 2), gpu_memory_total_gb=round(mt, 2),
    )


@app.post("/predict", response_model=InferenceResponse)
def predict(req: InferenceRequest):
    if policy is None:
        raise HTTPException(503, "모델 미로드")

    t0 = time.time()
    try:
        # 1. 이미지
        img_top = decode_b64_image(req.image_top)
        img_wrist = decode_b64_image(req.image_wrist)

        # 2. state 정규화
        norm_state = normalizer.normalize_state(req.state)
        state_t = torch.tensor(norm_state, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        # 3. observation 딕셔너리 (학습 시 사용한 키 이름과 일치해야 함)
        observation = {
            "observation.images.top": img_to_tensor(img_top),
            "observation.images.wrist": img_to_tensor(img_wrist),
            "observation.state": state_t,
        }

        # 4. 언어 명령 토크나이즈
        lang = req.language_instruction or "pick up the object"
        # pi0는 Pi0NewLineProcessor로 task 끝에 \n을 붙여 학습했으므로 동일하게 처리
        if POLICY_TYPE == "pi0" and not lang.endswith("\n"):
            lang = lang + "\n"
        tokenized = paligemma_tokenizer(
            lang,
            return_tensors="pt",
            padding="max_length",
            max_length=tokenizer_max_length,
            truncation=True,
        )
        observation["observation.language.tokens"] = tokenized.input_ids.to(DEVICE)
        observation["observation.language.attention_mask"] = tokenized.attention_mask.to(DEVICE).bool()

        # 5. 추론
        with torch.no_grad():
            action = policy.select_action(observation)

        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        if action.ndim == 1:
            action = action.reshape(1, -1)

        # 6. 역정규화
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
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {"error": "not found"}


@app.on_event("shutdown")
def on_shutdown():
    """서버 종료 시 GPU 메모리 및 리소스 해제."""
    global policy, normalizer, paligemma_tokenizer
    logger.info("서버 종료 — GPU 메모리 해제 중...")
    try:
        if policy is not None:
            del policy
            policy = None
        if normalizer is not None:
            del normalizer
            normalizer = None
        if paligemma_tokenizer is not None:
            del paligemma_tokenizer
            paligemma_tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        import gc
        gc.collect()
        logger.info("GPU 메모리 해제 완료")
    except Exception as e:
        logger.error(f"종료 정리 실패: {e}")


if __name__ == "__main__":
    import signal

    def graceful_exit(signum, frame):
        """Ctrl+C 등 시그널 시 정리 후 종료."""
        logger.info(f"시그널 {signum} 수신 — 종료합니다")
        on_shutdown()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, graceful_exit)
    signal.signal(signal.SIGTERM, graceful_exit)

    print(f"""
+-----------------------------------------------------------+
|  Pi0 / Pi0-FAST Inference Server                         |
|  Policy: {POLICY_TYPE:<48}|
|  GET  /health     Server status                          |
|  POST /predict    Inference                              |
|  GET  /model_info Model config                           |
+-----------------------------------------------------------+
""")
    try:
        uvicorn.run(app, host=HOST, port=PORT, log_level="info")
    finally:
        on_shutdown()
