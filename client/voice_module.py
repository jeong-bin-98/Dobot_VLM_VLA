#!/usr/bin/env python3
"""
1단계 STT 전처리 파이프라인

마이크 음성 → 전처리 → STT → 한국어 텍스트 출력
순수 STT만 수행. 커맨드 변환은 chatbot_module에서 처리.

백엔드:
  - whisper: faster-whisper (CPU에서 빠름, 테스트/노트북용)
  - qwen:    Qwen2.5-Omni (GPU 권장, 고정밀)

    python voice_module.py                        # whisper (기본)
    python voice_module.py --backend qwen         # qwen (GPU)
"""

import sys
import struct
import numpy as np

try:
    import pyaudio
except ImportError:
    print("pip install pyaudio")
    sys.exit(1)

# 오디오 설정
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK = 1024
FORMAT = pyaudio.paInt16
SILENCE_THRESHOLD = 500
SILENCE_DURATION = 1.5
MIN_RECORD_SEC = 0.5
MAX_RECORD_SEC = 10.0
MIN_TEXT_LENGTH = 2


class VoiceSTT:
    """
    마이크 음성 → 한국어 텍스트 변환

    backend:
        "whisper" — faster-whisper, CPU에서 빠름 (기본)
        "qwen"    — Qwen2.5-Omni, GPU 권장
    """

    def __init__(self, backend="whisper", model_name=None, device=None):
        self.backend = backend
        self.pa = pyaudio.PyAudio()

        if backend == "whisper":
            self._init_whisper(model_name or "base", device or "cpu")
        elif backend == "qwen":
            self._init_qwen(model_name or "Qwen/Qwen2.5-Omni-3B", device)
        else:
            raise ValueError(f"지원하지 않는 backend: {backend}")

    # ── Whisper 초기화 ──
    def _init_whisper(self, model_size, device):
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            print("pip install faster-whisper")
            sys.exit(1)

        compute_type = "int8" if device == "cpu" else "float16"
        print(f"  [STT] faster-whisper 로딩: {model_size} ({device}/{compute_type})")
        self._whisper = WhisperModel(model_size, device=device, compute_type=compute_type)
        print(f"  [STT] 로드 완료")

    # ── Qwen 초기화 ──
    def _init_qwen(self, model_name, device):
        try:
            from transformers import (
                Qwen2_5OmniForConditionalGeneration,
                WhisperFeatureExtractor,
                AutoTokenizer,
            )
            import torch
        except ImportError:
            print("pip install transformers torch accelerate")
            sys.exit(1)

        self._torch = torch

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self._qwen_device = device

        print(f"  [STT] Qwen 로딩: {model_name} ({device})")

        self._feature_extractor = WhisperFeatureExtractor.from_pretrained(
            model_name, local_files_only=True,
        )
        self._qwen_tokenizer = AutoTokenizer.from_pretrained(
            model_name, local_files_only=True,
        )
        self._qwen_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            device_map=device,
            local_files_only=True,
        )
        self._qwen_model.eval()

        # 프롬프트 캐싱
        prompt = (
            "<|im_start|>system\n"
            "You are a speech transcription assistant. "
            "Transcribe the audio exactly as spoken in Korean. "
            "Output only the transcription, nothing else."
            "<|im_end|>\n<|im_start|>user\n"
            "<|audio_bos|><|AUDIO|><|audio_eos|>"
            "<|im_end|>\n<|im_start|>assistant\n"
        )
        self._prompt_ids = self._qwen_tokenizer(prompt, return_tensors="pt").input_ids

        print(f"  [STT] 로드 완료")

    # ── 공통: 녹음 → 인식 ──
    def listen(self) -> str:
        """마이크 녹음 → STT → 한국어 텍스트 반환. 실패 시 빈 문자열."""
        audio_data = self._record()
        if audio_data is None:
            return ""

        audio_np = self._preprocess(audio_data)

        if self.backend == "whisper":
            text = self._transcribe_whisper(audio_np)
        else:
            text = self._transcribe_qwen(audio_np)

        print(f"  [STT] 인식: \"{text}\"")

        if len(text.strip()) < MIN_TEXT_LENGTH:
            print(f"  [STT] 인식 결과 너무 짧음, 무시")
            return ""

        return text.strip()

    # ── Whisper 추론 ──
    def _transcribe_whisper(self, audio_np: np.ndarray) -> str:
        segments, _ = self._whisper.transcribe(
            audio_np, language="ko", beam_size=5, vad_filter=True,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()

    # ── Qwen 추론 ──
    def _transcribe_qwen(self, audio_np: np.ndarray) -> str:
        torch = self._torch

        audio_features = self._feature_extractor(
            audio_np, sampling_rate=SAMPLE_RATE,
            return_tensors="pt", return_attention_mask=True,
        )
        input_features = audio_features.input_features.to(self._qwen_device)
        feature_attention_mask = audio_features.attention_mask.to(self._qwen_device)
        input_ids = self._prompt_ids.to(self._qwen_device)

        with torch.no_grad():
            output_ids = self._qwen_model.generate(
                input_ids=input_ids,
                input_features=input_features,
                feature_attention_mask=feature_attention_mask,
                max_new_tokens=256,
                do_sample=False,
            )

        generated = output_ids[0][input_ids.shape[1]:]
        return self._qwen_tokenizer.decode(generated, skip_special_tokens=True).strip()

    # ── 녹음 ──
    def _record(self) -> bytes | None:
        stream = self.pa.open(
            format=FORMAT, channels=CHANNELS, rate=SAMPLE_RATE,
            input=True, frames_per_buffer=CHUNK,
        )

        print("  [STT] 말씀하세요...")
        frames = []
        silent_chunks = 0
        has_voice = False
        max_chunks = int(MAX_RECORD_SEC * SAMPLE_RATE / CHUNK)
        silence_chunks_limit = int(SILENCE_DURATION * SAMPLE_RATE / CHUNK)

        for _ in range(max_chunks):
            data = stream.read(CHUNK, exception_on_overflow=False)
            rms = self._calc_rms(data)

            if rms > SILENCE_THRESHOLD:
                has_voice = True
                silent_chunks = 0
                frames.append(data)
            elif has_voice:
                silent_chunks += 1
                frames.append(data)
                if silent_chunks >= silence_chunks_limit:
                    break

        stream.stop_stream()
        stream.close()

        if not has_voice or len(frames) < int(MIN_RECORD_SEC * SAMPLE_RATE / CHUNK):
            print("  [STT] 음성 감지 안 됨")
            return None

        duration = len(frames) * CHUNK / SAMPLE_RATE
        print(f"  [STT] 녹음 완료: {duration:.1f}초")
        return b"".join(frames)

    # ── 전처리 ──
    def _preprocess(self, audio_bytes: bytes) -> np.ndarray:
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        rms = np.sqrt(np.mean(audio_np ** 2))
        if rms > 0:
            audio_np = audio_np * (3000.0 / rms)
        audio_np = np.clip(audio_np, -32768, 32767)
        return audio_np / 32768.0

    @staticmethod
    def _calc_rms(data: bytes) -> float:
        count = len(data) // 2
        shorts = struct.unpack(f"{count}h", data)
        sum_sq = sum(s * s for s in shorts)
        return (sum_sq / count) ** 0.5 if count > 0 else 0

    def close(self):
        self.pa.terminate()


# 단독 실행: 마이크 테스트
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="VoiceSTT 마이크 테스트")
    parser.add_argument("--backend", default="whisper", choices=["whisper", "qwen"])
    parser.add_argument("--model", default=None, help="모델명 (whisper: base/small/medium, qwen: Qwen/Qwen2.5-Omni-3B)")
    parser.add_argument("--device", default=None, help="cpu/cuda/mps")
    args = parser.parse_args()

    print("=" * 50)
    print(f"  VoiceSTT 마이크 테스트 ({args.backend})")
    print("  Ctrl+C로 종료")
    print("=" * 50)

    stt = VoiceSTT(backend=args.backend, model_name=args.model, device=args.device)

    try:
        while True:
            text = stt.listen()
            if text:
                print(f"\n  >>> 인식 결과: {text}\n")
            else:
                print("  (인식 실패)")
    except KeyboardInterrupt:
        print("\n종료")
    finally:
        stt.close()
