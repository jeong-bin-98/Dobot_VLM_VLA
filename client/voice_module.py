#!/usr/bin/env python3
"""
1단계 STT 전처리 파이프라인

마이크 음성 → 전처리 → Qwen 2.5 ASR → 한국어 텍스트 출력
순수 STT만 수행. 커맨드 변환은 chatbot_module에서 처리.

    python voice_module.py
"""

import sys
import time
import struct
import numpy as np

try:
    import pyaudio
except ImportError:
    print("pip install pyaudio")
    sys.exit(1)

try:
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    import torch
except ImportError:
    print("pip install transformers torch")
    sys.exit(1)

# 오디오 설정
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK = 1024
FORMAT = pyaudio.paInt16
SILENCE_THRESHOLD = 500       # RMS 기준 무음 판별
SILENCE_DURATION = 1.5        # 이 시간(초) 동안 무음이면 녹음 종료
MIN_RECORD_SEC = 0.5          # 최소 녹음 길이
MAX_RECORD_SEC = 10.0         # 최대 녹음 길이
CONFIDENCE_THRESHOLD = -0.3   # 로그 확률 신뢰도 임계값
MIN_TEXT_LENGTH = 2           # 최소 텍스트 길이


class VoiceSTT:
    """마이크 음성 → 한국어 텍스트 변환 (Qwen 2.5 ASR)"""

    def __init__(self, model_name="Qwen/Qwen2.5-Omni-7B", device=None):
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        print(f"  [STT] 모델 로딩: {model_name} ({self.device})")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            device_map=self.device,
        )
        self.model.eval()
        print(f"  [STT] 모델 로드 완료")

        # PyAudio
        self.pa = pyaudio.PyAudio()

    def listen(self) -> str:
        """
        마이크 녹음 → 전처리 → 1차 인식 → 신뢰도 체크 → (2차 인식) → 텍스트 반환

        Returns:
            한국어 텍스트 문자열. 인식 실패 시 빈 문자열.
        """
        # 1. 녹음
        audio_data = self._record()
        if audio_data is None:
            return ""

        # 2. 전처리 (모노, 16kHz, RMS 볼륨 정규화)
        audio_np = self._preprocess(audio_data)

        # 3. 1차 인식
        text, confidence = self._transcribe(audio_np)
        print(f"  [STT] 1차: \"{text}\" (conf: {confidence:.3f})")

        # 4. 신뢰도 체크 → 2차 인식
        if confidence < CONFIDENCE_THRESHOLD or len(text) < MIN_TEXT_LENGTH:
            print(f"  [STT] 신뢰도 낮음, 2차 정밀 인식 수행")
            text2, confidence2 = self._transcribe(audio_np, precise=True)
            print(f"  [STT] 2차: \"{text2}\" (conf: {confidence2:.3f})")
            if confidence2 > confidence:
                text = text2
                confidence = confidence2

        if len(text.strip()) < MIN_TEXT_LENGTH:
            print(f"  [STT] 인식 결과 너무 짧음, 무시")
            return ""

        return text.strip()

    def _record(self) -> bytes | None:
        """마이크 녹음. 음성 감지 후 무음이 지속되면 종료."""
        stream = self.pa.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK,
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
            # 아직 음성이 시작되지 않았으면 계속 대기

        stream.stop_stream()
        stream.close()

        if not has_voice or len(frames) < int(MIN_RECORD_SEC * SAMPLE_RATE / CHUNK):
            print("  [STT] 음성 감지 안 됨")
            return None

        duration = len(frames) * CHUNK / SAMPLE_RATE
        print(f"  [STT] 녹음 완료: {duration:.1f}초")
        return b"".join(frames)

    def _preprocess(self, audio_bytes: bytes) -> np.ndarray:
        """오디오 전처리: 모노, 16kHz, RMS 볼륨 정규화"""
        # bytes → int16 numpy 배열
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)

        # RMS 기반 볼륨 정규화
        rms = np.sqrt(np.mean(audio_np ** 2))
        if rms > 0:
            target_rms = 3000.0
            audio_np = audio_np * (target_rms / rms)

        # 클리핑 방지
        audio_np = np.clip(audio_np, -32768, 32767)

        # float32 [-1, 1] 범위로 정규화
        audio_np = audio_np / 32768.0

        return audio_np

    def _transcribe(self, audio_np: np.ndarray, precise: bool = False) -> tuple[str, float]:
        """
        Qwen 2.5 ASR로 음성 인식

        Args:
            audio_np: 전처리된 오디오 (float32, [-1, 1])
            precise: True이면 정밀 모드 (beam search, 느림)

        Returns:
            (텍스트, 로그 확률 기반 신뢰도)
        """
        inputs = self.processor(
            audio_np,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
        )

        input_features = inputs.input_features.to(self.device)

        generate_kwargs = {
            "language": "ko",
            "task": "transcribe",
            "return_dict_in_generate": True,
            "output_scores": True,
        }

        if precise:
            generate_kwargs["num_beams"] = 5
        else:
            generate_kwargs["num_beams"] = 1

        with torch.no_grad():
            outputs = self.model.generate(
                input_features,
                **generate_kwargs,
            )

        # 텍스트 디코딩
        token_ids = outputs.sequences[0]
        text = self.processor.batch_decode(
            [token_ids], skip_special_tokens=True
        )[0]

        # 신뢰도 계산 (로그 확률 평균)
        if hasattr(outputs, "scores") and outputs.scores:
            log_probs = []
            for i, score in enumerate(outputs.scores):
                probs = torch.nn.functional.log_softmax(score, dim=-1)
                token_id = token_ids[i + 1] if i + 1 < len(token_ids) else 0
                log_probs.append(probs[0, token_id].item())
            confidence = np.mean(log_probs) if log_probs else -1.0
        else:
            confidence = -1.0

        return text, confidence

    @staticmethod
    def _calc_rms(data: bytes) -> float:
        """오디오 청크의 RMS 볼륨 계산"""
        count = len(data) // 2
        shorts = struct.unpack(f"{count}h", data)
        sum_sq = sum(s * s for s in shorts)
        return (sum_sq / count) ** 0.5 if count > 0 else 0

    def close(self):
        """리소스 해제"""
        self.pa.terminate()


# 단독 실행: 마이크 테스트
if __name__ == "__main__":
    print("=" * 50)
    print("  VoiceSTT 마이크 테스트")
    print("  Ctrl+C로 종료")
    print("=" * 50)

    stt = VoiceSTT()

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
