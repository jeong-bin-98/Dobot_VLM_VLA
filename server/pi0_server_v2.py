#!/usr/bin/env python3
"""
Pi0 / Pi0-FAST HTTP 추론 서버 (v2 — 공식 lerobot 패턴)

pi0_server.py와 달리 수동 tokenizer/normalizer 대신 lerobot의
make_pre_post_processors를 사용하여 공식 방식으로 추론을 수행함.

핵심 차이점 (pi0_server.py 대비):
- ModelNormalizer 제거 → preprocessor가 자동 정규화
- paligemma_tokenizer 수동 로드 제거 → preprocessor가 토큰화
- Pi0NewLineProcessor 재구현 제거 → preprocessor에 포함
- torch.compile을 sample_actions에 적용 (mode=max-autotune, CUDA graphs 미사용)
- pi0/pi0_fast 분기 최소화

실행:
    PI0_POLICY_TYPE=pi0 PI0_MODEL_PATH=./outputs/xxx/checkpoints/last/pretrained_model \\
        python server/pi0_server_v2.py
    PI0_COMPILE=1 로 torch.compile 활성화 (기본: OFF)
"""

import os
import time
import base64
import logging
import numpy as np
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configuration
MODEL_PATH = os.environ.get(
    "PI0_MODEL_PATH",
    "./outputs/snack_v3_pi0_lora/checkpoints/last/pretrained_model",
)
POLICY_TYPE = os.environ.get("PI0_POLICY_TYPE", "pi0")  # "pi0" or "pi0_fast"
COMPILE_MODEL = os.environ.get("PI0_COMPILE", "0") == "1"
HOST = "0.0.0.0"
PORT = int(os.environ.get("PI0_PORT", 8000))
DEVICE = "cuda"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pi0_server_v2")

app = FastAPI(title="Pi0 Inference Server v2", version="2.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

policy = None
preprocessor = None
postprocessor = None


# 요청 / 응답
class InferenceRequest(BaseModel):
    image_top: str
    image_wrist: str
    state: list[float]
    language_instruction: str = ""
    chunk_size: int = 2


class InferenceResponse(BaseModel):
    actions: list[list[float]]
    raw_actions: list[float]
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
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def img_to_tensor(img: np.ndarray) -> torch.Tensor:
    """[H,W,3] uint8 -> [3,H,W] float32 0~1 (batch 차원은 preprocessor가 추가)"""
    return torch.from_numpy(img).permute(2, 0, 1).float().div(255.0)


# 모델 로드
@app.on_event("startup")
def startup_load_model():
    global policy, preprocessor, postprocessor

    logger.info(f"Loading {POLICY_TYPE} model: {MODEL_PATH}")
    t0 = time.time()

    # 1. Policy 로드 (LoRA 자동 감지)
    adapter_path = Path(MODEL_PATH) / "adapter_config.json"
    is_lora = adapter_path.exists()

    if POLICY_TYPE == "pi0_fast":
        from lerobot.policies.pi0_fast.modeling_pi0_fast import PI0FastPolicy as PolicyClass
        base_repo = "lerobot/pi0fast-base"
    else:
        from lerobot.policies.pi0.modeling_pi0 import PI0Policy as PolicyClass
        base_repo = "lerobot/pi0_base"

    if is_lora:
        from peft import PeftModel
        base_policy = PolicyClass.from_pretrained(base_repo)
        policy = PeftModel.from_pretrained(base_policy, MODEL_PATH)
        policy = policy.merge_and_unload()
        # LoRA merge 후 원본 config를 유지
        policy.config = base_policy.config
        logger.info(f"   {POLICY_TYPE} + LoRA merged")
    else:
        policy = PolicyClass.from_pretrained(MODEL_PATH)
        logger.info(f"   {POLICY_TYPE} (full fine-tuned)")

    policy.eval().to(DEVICE)

    # 2. Preprocessor/Postprocessor 로드 (공식 방식)
    from lerobot.policies.factory import make_pre_post_processors

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=MODEL_PATH,
        preprocessor_overrides={
            "device_processor": {"device": DEVICE},
            "rename_observations_processor": {"rename_map": {}},
        },
        postprocessor_overrides={
            "device_processor": {"device": "cpu"},
        },
    )
    logger.info("   Preprocessor/Postprocessor 로드 완료")

    # 3. torch.compile (선택) — 공식 방식: sample_actions에 max-autotune
    if COMPILE_MODEL:
        torch.set_float32_matmul_precision("high")
        if POLICY_TYPE == "pi0_fast":
            policy.model.sample_actions_fast = torch.compile(
                policy.model.sample_actions_fast, mode="max-autotune"
            )
        else:
            policy.model.sample_actions = torch.compile(
                policy.model.sample_actions, mode="max-autotune"
            )
        logger.info("   torch.compile 적용: mode=max-autotune")

    dt = time.time() - t0
    logger.info(f"로드 완료 ({dt:.1f}s)")
    if torch.cuda.is_available():
        g = torch.cuda.get_device_properties(0)
        logger.info(f"{g.name}  {g.total_memory / 1e9:.0f}GB")

    # 4. Warmup (compile 트리거 + 첫 호출 지연 제거)
    if COMPILE_MODEL:
        logger.info("Warmup 추론 중... (torch.compile 컴파일, 2~5분 소요)")
        warmup_t0 = time.time()
        try:
            dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
            dummy_obs = {
                "observation.images.top": img_to_tensor(dummy_img),
                "observation.images.wrist": img_to_tensor(dummy_img),
                "observation.state": torch.zeros(5, dtype=torch.float32),
                "task": "pick up the object",
            }
            processed = preprocessor(dummy_obs)
            with torch.inference_mode():
                _ = policy.select_action(processed)
            logger.info(f"Warmup 완료 ({time.time() - warmup_t0:.1f}s)")
        except Exception as e:
            logger.warning(f"Warmup 실패 (무시하고 계속): {e}")


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
        # 1. 이미지 디코딩
        img_top = decode_b64_image(req.image_top)
        img_wrist = decode_b64_image(req.image_wrist)

        # 2. Raw observation 구성 (preprocessor가 나머지 처리)
        raw_obs = {
            "observation.images.top": img_to_tensor(img_top),
            "observation.images.wrist": img_to_tensor(img_wrist),
            "observation.state": torch.tensor(req.state, dtype=torch.float32),
            "task": req.language_instruction or "pick up the object",
        }

        # 3. Preprocessor 적용 (정규화 + 토큰화 + 배치 + 디바이스)
        processed = preprocessor(raw_obs)

        # 4. 추론
        with torch.inference_mode():
            action = policy.select_action(processed)

        # 5. Postprocessor 적용 (역정규화 + CPU)
        action = postprocessor(action)

        # 6. 응답 포맷 변환
        if isinstance(action, torch.Tensor):
            action_np = action.cpu().numpy()
        else:
            action_np = np.array(action)
        if action_np.ndim == 1:
            action_np = action_np.reshape(1, -1)

        chunk = min(req.chunk_size, action_np.shape[0])
        actions = action_np[:chunk].tolist()
        raw_out = action_np[0].tolist()

        dt_ms = (time.time() - t0) * 1000
        logger.info(
            f"{dt_ms:.0f}ms | "
            f"Δ[0]=[{actions[0][0]:+.1f},{actions[0][1]:+.1f},{actions[0][2]:+.1f}] | "
            f"\"{req.language_instruction[:25]}\""
        )
        return InferenceResponse(
            actions=actions, raw_actions=raw_out, inference_time_ms=round(dt_ms, 1)
        )

    except Exception as e:
        logger.error(f"{e}")
        import traceback; traceback.print_exc()
        raise HTTPException(500, str(e))


@app.on_event("shutdown")
def on_shutdown():
    global policy, preprocessor, postprocessor
    logger.info("서버 종료 — GPU 메모리 해제 중...")
    try:
        policy = None
        preprocessor = None
        postprocessor = None
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
        logger.info(f"시그널 {signum} 수신 — 종료합니다")
        on_shutdown()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, graceful_exit)
    signal.signal(signal.SIGTERM, graceful_exit)

    print(f"""
+-----------------------------------------------------------+
|  Pi0 / Pi0-FAST Inference Server v2 (공식 lerobot 패턴)  |
|  Policy: {POLICY_TYPE:<48}|
|  Compile: {'ON (max-autotune)' if COMPILE_MODEL else 'OFF':<47}|
|  GET  /health     Server status                          |
|  POST /predict    Inference                              |
+-----------------------------------------------------------+
""")
    try:
        uvicorn.run(app, host=HOST, port=PORT, log_level="info")
    finally:
        on_shutdown()
