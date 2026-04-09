#!/usr/bin/env python3
"""
Pi0 HTTP 추론 클라이언트 + LLM 체이닝

LLM(선택) -> Pi0 서버 -> DOBOT 실행.

    python pi0_dobot_client.py \\
        --server http://192.168.1.100:8000 \\
        --task "pick up the red cup"

    python pi0_dobot_client.py \\
        --server http://192.168.1.100:8000 \\
        --llm-mode --goal "책상 정리"
"""

import sys
import os
import time
import argparse
import json
import math
import base64
import numpy as np

try:
    import cv2
    import requests
    import pydobot
    from serial.tools import list_ports
except ImportError as e:
    print(f"필요 패키지: {e}")
    print("pip install opencv-python requests pydobot pyserial")
    sys.exit(1)
# Safety bounds
BOUNDS = {
    "x": (150, 310),
    "y": (-150, 150),
    "z": (-30, 150),
    "r": (-90, 90),
}
# Dobot Magician 기구학 상수
DOBOT_L1 = 135.0   # mm — rear arm 길이
DOBOT_L2 = 147.0   # mm — forearm 길이
SINGULARITY_THRESHOLD = 0.92  # |cos(θ2)| > 이 값이면 singularity 위험
ALARM_POS_THRESHOLD = 5.0  # mm — move_to 후 오차가 이보다 크면 ALARM 판정


def _cos_j2(x, y):
    """목표점 (x,y)에서의 cos(θ2) 계산. |값| → 1이면 singularity."""
    r_sq = x ** 2 + y ** 2
    denom = 2 * DOBOT_L1 * DOBOT_L2
    return float(np.clip((r_sq - DOBOT_L1 ** 2 - DOBOT_L2 ** 2) / denom, -1.0, 1.0))


def _path_crosses_singularity(cx, cy, tx, ty, n_samples=5):
    """직선 경로 상 n개 지점을 샘플링하여 singularity 관통 여부 판단."""
    for t in np.linspace(0, 1, n_samples):
        px = cx + t * (tx - cx)
        py = cy + t * (ty - cy)
        if abs(_cos_j2(px, py)) > SINGULARITY_THRESHOLD:
            return True
    return False


def _compute_via_point(cx, cy, tx, ty):
    """Singularity 영역을 우회하는 경유점 계산.

    전략: 경로의 중점을 safe_radius (θ2=90° 지점)로 밀어서
    팔이 자연스럽게 구부러진 상태를 유지하며 이동.
    """
    safe_r = math.sqrt(DOBOT_L1 ** 2 + DOBOT_L2 ** 2)  # ~200mm (θ2=90°)

    mx, my = (cx + tx) / 2, (cy + ty) / 2
    mid_r = math.sqrt(mx ** 2 + my ** 2)

    if mid_r > 1e-3:
        scale = safe_r / mid_r
        vx, vy = mx * scale, my * scale
    else:
        dx, dy = tx - cx, ty - cy
        path_len = math.sqrt(dx ** 2 + dy ** 2)
        if path_len > 1e-3:
            vx, vy = -dy / path_len * safe_r, dx / path_len * safe_r
        else:
            vx, vy = safe_r, 0

    vx = float(np.clip(vx, *BOUNDS["x"]))
    vy = float(np.clip(vy, *BOUNDS["y"]))

    if abs(_cos_j2(vx, vy)) > SINGULARITY_THRESHOLD:
        return None
    return (vx, vy)

IMG_W, IMG_H = 640, 480
# LLM 플래너 (로컬 CPU)
class LLMPlanner:
    """
    고수준 작업을 로봇이 실행 가능한 하위 명령으로 분해합니다.
    
    지원 백엔드:
    - "simple": 규칙 기반 (LLM 없이 테스트용)
    - "local":  HuggingFace transformers (CPU)
    - "openai": OpenAI API (GPT-4)
    - "anthropic": Claude API
    """

    SYSTEM_PROMPT = """You are a robot task planner for a DOBOT Magician robot arm.
The robot can perform pick-and-place operations with a gripper.
Given a high-level goal and visible objects, decompose it into simple, sequential commands.

Rules:
- Each command should be a single pick-and-place action
- Use simple English: "pick up [object] and place it [location]"
- Maximum 5 sub-tasks per goal
- Output ONLY a JSON array of command strings, nothing else

Example:
Goal: "clean up the desk"
Objects: red cup, blue pen, book
Output: ["pick up the red cup and place it on the left side", "pick up the blue pen and place it in the holder", "pick up the book and place it on the shelf"]"""

    def __init__(self, backend="simple", model_name=None):
        self.backend = backend
        self.model_name = model_name
        self.model = None
        self.tokenizer = None

        if backend == "local":
            self._load_local_model(model_name or "Qwen/Qwen2.5-1.5B-Instruct")
        
        print(f"LLM 플래너: {backend}" + (f" ({model_name})" if model_name else ""))

    def _load_local_model(self, name):
        """HuggingFace 로컬 모델 로드 (CPU)"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            print(f"   로컬 LLM 로드 중: {name} (CPU)...")
            self.tokenizer = AutoTokenizer.from_pretrained(name)
            self.model = AutoModelForCausalLM.from_pretrained(
                name, torch_dtype="auto", device_map="cpu"
            )
            self.model.eval()
            print(f"   LLM 로드 완료")
        except Exception as e:
            print(f"   LLM 로드 실패: {e}")
            print(f"   -> simple 모드로 전환")
            self.backend = "simple"

    def plan(self, goal: str, visible_objects: str = "") -> list[str]:
        """
        고수준 목표 -> 하위 명령 리스트
        
        Args:
            goal: 사용자의 고수준 명령 (한글/영어)
            visible_objects: 현재 보이는 물체 설명
        
        Returns:
            ["pick up X and place Y", ...] 형태의 명령 리스트
        """
        if self.backend == "simple":
            return self._plan_simple(goal)
        elif self.backend == "local":
            return self._plan_local(goal, visible_objects)
        elif self.backend == "openai":
            return self._plan_openai(goal, visible_objects)
        elif self.backend == "anthropic":
            return self._plan_anthropic(goal, visible_objects)
        else:
            return [goal]

    def _plan_simple(self, goal):
        """규칙 기반 간단 분해 (테스트용)"""
        # 이미 영어 단일 명령이면 그대로 반환
        if any(w in goal.lower() for w in ["pick", "place", "move", "grab", "put"]):
            return [goal]
        # 한글이면 기본 명령 반환
        return [f"pick up the object"]

    def _plan_local(self, goal, objects):
        """로컬 LLM으로 계획 생성"""
        prompt = f"Goal: \"{goal}\"\nObjects: {objects}\nOutput:"
        
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt")
        
        import torch
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=256, temperature=0.3, do_sample=True
            )
        
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return self._parse_commands(response, goal)

    def _plan_openai(self, goal, objects):
        """OpenAI API 사용"""
        import openai
        response = openai.chat.completions.create(
            model=self.model_name or "gpt-4",
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": f"Goal: \"{goal}\"\nObjects: {objects}\nOutput:"},
            ],
            temperature=0.3,
            max_tokens=256,
        )
        return self._parse_commands(response.choices[0].message.content, goal)

    def _plan_anthropic(self, goal, objects):
        """Anthropic Claude API 사용"""
        import anthropic
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=self.model_name or "claude-sonnet-4-20250514",
            max_tokens=256,
            system=self.SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": f"Goal: \"{goal}\"\nObjects: {objects}\nOutput:"},
            ],
        )
        return self._parse_commands(response.content[0].text, goal)

    def _parse_commands(self, text, fallback):
        """LLM 출력에서 명령 리스트 파싱"""
        try:
            # JSON 배열 추출
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                commands = json.loads(text[start:end])
                if isinstance(commands, list) and len(commands) > 0:
                    return [str(c) for c in commands]
        except (json.JSONDecodeError, ValueError):
            pass
        
        # 파싱 실패 시 줄 단위로 시도
        lines = [l.strip().strip("-•").strip() for l in text.strip().split("\n") if l.strip()]
        if lines:
            return lines[:5]
        
        return [fallback]
# Pi0 원격 추론 클라이언트
class Pi0Client:
    """Pi0 추론 서버와 통신하는 클라이언트"""

    def __init__(self, server_url: str, chunk_size: int = 2):
        self.server_url = server_url.rstrip("/")
        self.chunk_size = chunk_size
        self.session = requests.Session()
        
        # 서버 연결 확인
        try:
            r = self.session.get(f"{self.server_url}/health", timeout=5)
            info = r.json()
            print(f"Pi0 서버 연결: {info['gpu_name']} ({info['gpu_memory_used_gb']:.1f}/{info['gpu_memory_total_gb']:.1f} GB)")
        except Exception as e:
            print(f"Pi0 서버 연결 실패: {e}")
            print(f"   서버 주소: {self.server_url}")
            sys.exit(1)

    def predict(self, img_top, img_wrist, state, language_instruction=""):
        """
        Pi0 추론 요청
        
        Args:
            img_top:   top 카메라 이미지 (numpy BGR)
            img_wrist: wrist 카메라 이미지 (numpy BGR)
            state:     [x, y, z, r, gripper] raw 값
            language_instruction: 언어 명령
        
        Returns:
            actions: list of [dx, dy, dz, dr, grip] (역정규화된 real delta)
            raw_actions: raw 모델 출력 (디버깅)
            inference_time_ms: 추론 시간
        """
        # 이미지를 JPEG -> base64 인코딩 (대역폭 절약)
        _, buf_top = cv2.imencode(".jpg", img_top, [cv2.IMWRITE_JPEG_QUALITY, 85])
        _, buf_wrist = cv2.imencode(".jpg", img_wrist, [cv2.IMWRITE_JPEG_QUALITY, 85])

        payload = {
            "image_top":   base64.b64encode(buf_top).decode("utf-8"),
            "image_wrist": base64.b64encode(buf_wrist).decode("utf-8"),
            "state":       state,
            "language_instruction": language_instruction,
            "chunk_size":  self.chunk_size,
        }

        try:
            r = self.session.post(
                f"{self.server_url}/predict",
                json=payload,
                timeout=10,
            )
            r.raise_for_status()
            data = r.json()
            return data["actions"], data["raw_actions"], data["inference_time_ms"]
        except requests.exceptions.Timeout:
            print("   서버 타임아웃 (10초)")
            return None, None, 0
        except Exception as e:
            print(f"   추론 요청 실패: {e}")
            return None, None, 0
# DOBOT Controller
class DobotController:
    def __init__(self, port=None):
        self.dobot = None
        self.grip_on = False
        port = port or self._find_port()
        
        self.dobot = pydobot.Dobot(port=port, verbose=False)
        self.dobot.speed(150, 150)
        print(f"DOBOT: {port}")

    def _find_port(self):
        for p in list_ports.comports():
            if "CH340" in p.description or "CP210" in p.description:
                return p.device
        # Mac: /dev/tty.usbserial-* 패턴 검색
        for p in list_ports.comports():
            if "usbserial" in p.device or "usbmodem" in p.device:
                return p.device
        ports = list_ports.comports()
        return ports[0].device if ports else "/dev/tty.usbserial-0001"

    def get_pose(self):
        r = self.dobot.pose()
        return [round(r[i], 2) for i in range(4)]

    def get_state(self):
        """[x, y, z, r, gripper] 5차원 state 반환"""
        pose = self.get_pose()
        return pose + [1.0 if self.grip_on else 0.0]

    def execute(self, delta):
        """
        Execute delta action with singularity avoidance.
        - cos(θ2) 기반 singularity 감지
        - 경유점(via-point)으로 우회
        - ALARM 감지 및 자동 복구
        - j2 관절각도 사후 검증
        """
        cur = self.get_pose()

        tx = float(np.clip(cur[0] + delta[0], *BOUNDS["x"]))
        ty = float(np.clip(cur[1] + delta[1], *BOUNDS["y"]))
        tz = float(np.clip(cur[2] + delta[2], *BOUNDS["z"]))
        tr = float(np.clip(cur[3] + delta[3], *BOUNDS["r"]))

        # 1단계: cos(θ2) 기반 singularity 감지
        target_cos = _cos_j2(tx, ty)
        target_dangerous = abs(target_cos) > SINGULARITY_THRESHOLD
        path_dangerous = _path_crosses_singularity(cur[0], cur[1], tx, ty)

        if target_dangerous or path_dangerous:
            via = _compute_via_point(cur[0], cur[1], tx, ty)
            if via:
                print(f"    >> Singularity 회피: 경유점 ({via[0]:.0f}, {via[1]:.0f})mm "
                      f"[cos(θ2): 목표={target_cos:+.2f}]")
                self.dobot.move_to(via[0], via[1], tz, tr, wait=True)

            # 목표 자체가 위험하면 safe_radius로 스케일다운 (도달 불가)
            if target_dangerous:
                safe_r = math.sqrt(DOBOT_L1 ** 2 + DOBOT_L2 ** 2)
                dist = math.sqrt(tx ** 2 + ty ** 2)
                if dist > 1e-3:
                    scale = safe_r / dist
                    tx, ty = float(tx * scale), float(ty * scale)
                    tx = float(np.clip(tx, *BOUNDS["x"]))
                    ty = float(np.clip(ty, *BOUNDS["y"]))
                print(f"    >> 목표 스케일다운: ({tx:.0f}, {ty:.0f})mm")

        # 2단계: 최종 목표로 이동
        self.dobot.move_to(tx, ty, tz, tr, wait=True)

        # 3단계: ALARM 감지 + j2 관절각도 사후 검증
        try:
            actual = self.dobot.pose()
            error = math.sqrt((actual[0] - tx) ** 2 + (actual[1] - ty) ** 2 + (actual[2] - tz) ** 2)
            if error > ALARM_POS_THRESHOLD:
                print(f"    >> ALARM 감지 (오차 {error:.1f}mm), 복구 중...")
                self.clear_alarm()
                return cur, [200, 0, 50, 0], True  # alarmed=True
            # j2 관절각도 경고
            j2 = actual[5]
            if j2 < 5 or j2 > 80:
                print(f"    >> j2={j2:.1f}deg — singularity 근접 경고")
        except Exception:
            pass

        # 4단계: 그리퍼 (이동 완료 후)
        new_grip = delta[4] > 0.5
        if new_grip != self.grip_on:
            self.grip_on = new_grip
            time.sleep(0.1)
            try:
                self.dobot.grip(self.grip_on)
                time.sleep(0.5)
            except:
                try:
                    self.dobot.suck(self.grip_on)
                    time.sleep(0.5)
                except:
                    pass
            print(f"    그리퍼 {'ON' if self.grip_on else 'OFF'}")

        return cur, [tx, ty, tz, tr], False

    def clear_alarm(self):
        """ALARM 발생 시 시리얼 재연결로 복구 후 안전 위치로 이동."""
        import gc
        port = None
        try:
            ser = getattr(self.dobot, 'ser', None)
            if ser:
                port = ser.port
                if ser.is_open:
                    ser.close()
        except Exception:
            pass

        # 기존 객체 완전 제거 (01_collect_data.py 패턴)
        try:
            del self.dobot
        except Exception:
            pass
        self.dobot = None
        gc.collect()

        # 펌웨어 리셋 대기 (3초 + 여유 1초)
        print(f"    >> 알람 해제 중... (4초 대기)")
        time.sleep(4)

        try:
            self.dobot = pydobot.Dobot(port=port or self._find_port(), verbose=False)
            time.sleep(1)  # 연결 안정화 대기
            self.dobot.speed(100, 100)  # 낮은 속도로 시작
            self.grip_on = False
            self.dobot.move_to(200, 0, 50, 0, wait=True)
            self.dobot.speed(150, 150)  # 원래 속도 복원
            pose = self.get_pose()
            print(f"    >> ALARM 복구 완료: ({pose[0]:.0f},{pose[1]:.0f},{pose[2]:.0f})")
        except Exception as e:
            print(f"    >> ALARM 복구 실패: {e}")

    def go_home(self):
        """Home 위치(200,0,50,0)로 이동. 호밍 완료 후 사용."""
        try:
            print("  Home 위치로 이동 중...")
            self.dobot.move_to(200, 0, 50, 0, wait=True)
            self.grip_on = False
            try:
                self.dobot.grip(False)
            except:
                pass
            print("  Home 위치 도착: (200, 0, 50, 0)")
        except Exception as e:
            print(f"  Go Home 실패: {e}")

    def homing(self):
        """호밍 (리밋스위치 기반 원점 복귀)."""
        from pydobot.dobot import Message, CommunicationProtocolIDs as IDs, ControlValues as CV
        try:
            print("  호밍 중... (리밋스위치 원점 복귀)")
            # SET_HOME_CMD (id=31) — queued, 리밋스위치 찍고 원점 설정
            msg = Message()
            msg.id = IDs.SET_HOME_CMD
            msg.ctrl = CV.THREE
            msg.params = bytearray(4)  # reserved 4 bytes (0)
            self.dobot._send_command(msg, wait=True)
            self.grip_on = False
            pose = self.get_pose()
            print(f"  호밍 완료: x={pose[0]:.1f} y={pose[1]:.1f} z={pose[2]:.1f} r={pose[3]:.1f}")
        except Exception as e:
            print(f"  호밍 실패: {e}")

    def close(self):
        if self.dobot:
            try:
                self.dobot.grip(False)
                self.dobot.suck(False)
                self.dobot.close()
            except:
                pass
# 카메라 관리
class CameraManager:
    """
    카메라 매핑 (데이터 수집 01_collect_data.py와 동일):
      cam1 (index 0) = wrist 카메라
      cam2 (index 1) = top 카메라
    """
    def __init__(self, cam1_id=0, cam2_id=1):
        import platform
        backend = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_V4L2

        self.cap_wrist = cv2.VideoCapture(cam1_id)
        self.cap_top = cv2.VideoCapture(cam2_id)

        if not self.cap_wrist.isOpened():
            self.cap_wrist = cv2.VideoCapture(cam1_id, backend)
        if not self.cap_top.isOpened():
            self.cap_top = cv2.VideoCapture(cam2_id, backend)

        for cap in [self.cap_wrist, self.cap_top]:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMG_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_H)

        print(f"카메라: wrist={cam1_id}, top={cam2_id}")

    def capture(self):
        """Returns (top_frame, wrist_frame) — 서버 API 순서에 맞춤"""
        _, f_wrist = self.cap_wrist.read()
        _, f_top = self.cap_top.read()
        return f_top, f_wrist

    def close(self):
        self.cap_wrist.release()
        self.cap_top.release()
        cv2.destroyAllWindows()
# 메인 클래스
class Pi0DobotPipeline:
    def __init__(self, args):
        # Pi0 원격 클라이언트
        self.pi0 = Pi0Client(args.server, chunk_size=args.chunk_size)

        # DOBOT
        self.dobot = DobotController(args.port)

        # 카메라
        self.cameras = CameraManager(args.cam1, args.cam2)

        # LLM 플래너 (선택)
        self.planner = None
        if args.llm_mode:
            self.planner = LLMPlanner(
                backend=args.llm_backend,
                model_name=args.llm_model,
            )

        self.current_task = args.task or args.goal or "pick up the object"

    # -----------------------------------------
    # LLM 체이닝 모드
    # -----------------------------------------
    def run_llm_chain(self, goal, max_cycles_per_task=20):
        """
        LLM -> Pi0 -> DOBOT 전체 체이닝

        1. LLM이 고수준 목표를 하위 작업으로 분해
        2. 각 하위 작업에 대해 Pi0 추론 + DOBOT 실행
        """
        print(f"\n목표: {goal}")

        # LLM 플래닝
        if self.planner:
            subtasks = self.planner.plan(goal)
        else:
            subtasks = [goal]

        print(f" 계획 ({len(subtasks)}개 하위 작업):")
        for i, task in enumerate(subtasks):
            print(f"   {i+1}. {task}")
        print()

        # 각 하위 작업 실행
        for task_idx, task in enumerate(subtasks):
            print(f"\n{'='*60}")
            print(f"   [{task_idx+1}/{len(subtasks)}] {task}")
            print(f"{'='*60}")

            self.current_task = task
            success = self._execute_task(task, max_cycles_per_task)

            if not success:
                print(f"   작업 실패, 다음으로 진행")

            # 작업 간 잠시 대기
            time.sleep(0.5)

        print(f"\n전체 작업 완료!")

    def _execute_task(self, task, max_cycles):
        """단일 하위 작업 실행 (Pi0 추론 루프)"""
        for cycle in range(max_cycles):
            f1, f2 = self.cameras.capture()
            if f1 is None or f2 is None:
                continue

            state = self.dobot.get_state()

            # Pi0 서버 추론
            actions, raw_out, dt_ms = self.pi0.predict(f1, f2, state, task)

            if actions is None:
                print(f"   추론 실패, 재시도...")
                time.sleep(1)
                continue

            # 디버그 (첫 사이클)
            if cycle == 0:
                print(f"   [DEBUG] state: {state}")
                print(f"   [DEBUG] raw_out: [{', '.join(f'{v:+.3f}' for v in raw_out)}]")
                print(f"   [DEBUG] delta[0]: [{', '.join(f'{v:+.1f}' for v in actions[0])}]")

            # 액션 실행
            alarmed = False
            for i, delta in enumerate(actions):
                cur, tgt, alarmed = self.dobot.execute(delta)
                print(
                    f"   Cycle {cycle+1} [{i+1}/{len(actions)}] "
                    f"Δ[{delta[0]:+.1f},{delta[1]:+.1f},{delta[2]:+.1f},{delta[3]:+.1f},{delta[4]:.2f}]mm "
                    f"({cur[0]:.0f},{cur[1]:.0f},{cur[2]:.0f})->({tgt[0]:.0f},{tgt[1]:.0f},{tgt[2]:.0f}) "
                    f"G:{'ON' if self.dobot.grip_on else 'OFF'} "
                    f"서버:{dt_ms:.0f}ms"
                )
                if alarmed:
                    print(f"   >> ALARM 복구됨 — 나머지 action 스킵, 새로 관측합니다")
                    break

        return True

    # -----------------------------------------
    # Manual mode
    # -----------------------------------------
    def run_manual(self, max_cycles=50):
        """
        키보드 제어 수동 모드
        [SPACE] 1회 추론  [A] 자동  [H] 홈  [G] 그리퍼  [ESC] 종료
        """
        print(f"""
+-----------------------------------------------------------+
|  Pi0 -> DOBOT (원격 추론 모드)                          |
+-----------------------------------------------------------+
|  Server: {self.pi0.server_url:<48}|
|  Task:   {self.current_task:<48}|
|  [SPACE] 1회 추론   [A] 자동   [L] LLM 체이닝             |
|  [Q] Home 위치 이동  [R] 호밍 (리밋스위치 원점)           |
|  [G] 그리퍼   [T] 명령 변경   [ESC] 종료                  |
+-----------------------------------------------------------+
""")
        auto_mode = False
        cycle = 0

        try:
            while cycle < max_cycles:
                f1, f2 = self.cameras.capture()
                if f1 is not None and f2 is not None:
                    self._show_preview(f1, f2, cycle, auto_mode)

                key = cv2.waitKey(30 if not auto_mode else 1) & 0xFF

                if key == 27:  # ESC
                    break

                elif (key == ord(' ') or auto_mode) and f1 is not None and f2 is not None:
                    state = self.dobot.get_state()

                    # Pi0 서버 추론
                    actions, raw_out, dt_ms = self.pi0.predict(
                        f1, f2, state, self.current_task
                    )

                    if actions is None:
                        continue

                    if cycle == 0:
                        print(f"\n  [DEBUG] state: {state}")
                        print(f"  [DEBUG] raw_out: {raw_out}")
                        print(f"  [DEBUG] delta[0]: {actions[0]}\n")

                    for i, delta in enumerate(actions):
                        cur, tgt = self.dobot.execute(delta)
                        print(
                            f"  Cycle {cycle+1} [{i+1}/{len(actions)}] "
                            f"Δ[{delta[0]:+.1f},{delta[1]:+.1f},{delta[2]:+.1f},{delta[3]:+.1f},{delta[4]:.2f}] "
                            f"({cur[0]:.0f},{cur[1]:.0f},{cur[2]:.0f})->({tgt[0]:.0f},{tgt[1]:.0f},{tgt[2]:.0f}) "
                            f"G:{'ON' if self.dobot.grip_on else 'OFF'} "
                            f"서버:{dt_ms:.0f}ms"
                        )
                    cycle += 1

                elif key == ord('a'):
                    auto_mode = not auto_mode
                    print(f"\n  {'자동' if auto_mode else '수동'} 모드")

                elif key == ord('q'):
                    self.dobot.go_home()

                elif key == ord('r'):
                    self.dobot.homing()

                elif key == ord('g'):
                    self.dobot.grip_on = not self.dobot.grip_on
                    try:
                        self.dobot.dobot.grip(self.dobot.grip_on)
                        time.sleep(0.5)
                    except:
                        pass
                    print(f"  그리퍼: {'ON' if self.dobot.grip_on else 'OFF'}")

                elif key == ord('t'):
                    auto_mode = False
                    print("\n  새 명령 입력 (콘솔):")
                    new_task = input("  > ").strip()
                    if new_task:
                        self.current_task = new_task
                        print(f"  Task: {self.current_task}")

                elif key == ord('l'):
                    auto_mode = False
                    if self.planner:
                        print("\n  LLM 체이닝 목표 입력:")
                        goal = input("  > ").strip()
                        if goal:
                            self.run_llm_chain(goal)
                    else:
                        print("  LLM 모드 비활성 (--llm-mode 옵션 필요)")

        except KeyboardInterrupt:
            print("\n중단")

        self.close()

    def _show_preview(self, f1, f2, cycle, auto_mode):
        pose = self.dobot.get_pose()
        mode_str = "AUTO" if auto_mode else "MANUAL"
        color = (0, 0, 255) if auto_mode else (0, 255, 0)

        cv2.putText(f1, f"TOP | Pi0 Remote | {mode_str} | Cycle {cycle}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(f1, f"X:{pose[0]:.0f} Y:{pose[1]:.0f} Z:{pose[2]:.0f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(f1, f"Task: {self.current_task[:50]}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        cv2.putText(f2, f"WRIST | Grip: {'ON' if self.dobot.grip_on else 'OFF'}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        combined = np.hstack([f1, f2])
        cv2.imshow("Pi0 Remote Inference", combined)

    def close(self):
        self.dobot.close()
        self.cameras.close()
        print("종료")
# 실행
def main():
    parser = argparse.ArgumentParser(description="LLM -> Pi0 -> DOBOT 체이닝")

    # 서버
    parser.add_argument("--server", type=str, required=True,
                        help="Pi0 서버 URL (예: http://192.168.1.100:8000)")

    # DOBOT
    parser.add_argument("--port", type=str, default=None, help="DOBOT 시리얼 포트")
    parser.add_argument("--cam1", type=int, default=0, help="Wrist 카메라 ID (데이터 수집과 동일)")
    parser.add_argument("--cam2", type=int, default=1, help="Top 카메라 ID (데이터 수집과 동일)")

    # 작업
    parser.add_argument("--task", type=str, default="pick up the object",
                        help="단일 언어 명령 (수동 모드)")
    parser.add_argument("--chunk-size", type=int, default=2,
                        help="Pi0 액션 청크 사용 스텝 수 (1~50)")
    parser.add_argument("--cycles", type=int, default=50, help="최대 사이클")

    # LLM 체이닝
    parser.add_argument("--llm-mode", action="store_true",
                        help="LLM 체이닝 모드 활성화")
    parser.add_argument("--llm-backend", type=str, default="simple",
                        choices=["simple", "local", "openai", "anthropic"],
                        help="LLM 백엔드 선택")
    parser.add_argument("--llm-model", type=str, default=None,
                        help="LLM 모델명 (예: Qwen/Qwen2.5-1.5B-Instruct)")
    parser.add_argument("--goal", type=str, default=None,
                        help="LLM 체이닝 고수준 목표")
    parser.add_argument("--homing", action="store_true",
                        help="시작 시 리밋스위치 호밍 실행")

    args = parser.parse_args()

    pipeline = Pi0DobotPipeline(args)

    try:
        if args.homing:
            pipeline.dobot.homing()
        pipeline.dobot.go_home()

        if args.goal and args.llm_mode:
            # LLM 체이닝 모드
            pipeline.run_llm_chain(args.goal)
        else:
            # 수동 모드
            pipeline.run_manual(args.cycles)

    except Exception as e:
        print(f"\n에러: {e}")
        import traceback
        traceback.print_exc()
        pipeline.close()
if __name__ == "__main__":
    main()
