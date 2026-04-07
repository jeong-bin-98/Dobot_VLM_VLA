#!/usr/bin/env python3
"""Dobot Magician 간단 테스트 스크립트"""

import time
import pydobot
from serial.tools import list_ports


def find_port():
    for p in list_ports.comports():
        if any(chip in p.description for chip in ("CH340", "CP210")):
            return p.device
    # macOS: usbserial 포트 찾기
    for p in list_ports.comports():
        if "usbserial" in p.device:
            return p.device
    return None


def main():
    port = find_port()
    if not port:
        print("Dobot을 찾을 수 없습니다")
        return

    print(f"연결: {port}")
    print("연결 중... (3초 대기)")
    time.sleep(3)
    bot = pydobot.Dobot(port=port, verbose=True)
    time.sleep(1)
    bot.speed(100, 100)

    pose = bot.pose()
    print(f"현재 위치: x={pose[0]:.1f} y={pose[1]:.1f} z={pose[2]:.1f} r={pose[3]:.1f}")

    print("\n--- 테스트 메뉴 ---")
    print("[1] 홈 위치 이동 (200, 0, 50)")
    print("[2] 앞으로 이동 (250, 0, 50)")
    print("[3] 흡착컵 ON")
    print("[4] 흡착컵 OFF")
    print("[5] 그리퍼 ON")
    print("[6] 그리퍼 OFF")
    print("[7] 현재 위치 확인")
    print("[q] 종료")

    while True:
        cmd = input("\n> ").strip()
        if cmd == "1":
            bot.move_to(200, 0, 50, 0, wait=True)
            print("홈 이동 완료")
        elif cmd == "2":
            bot.move_to(250, 0, 50, 0, wait=True)
            print("앞으로 이동 완료")
        elif cmd == "3":
            bot.suck(True)
            print("흡착컵 ON")
        elif cmd == "4":
            bot.suck(False)
            print("흡착컵 OFF")
        elif cmd == "5":
            bot.grip(True)
            print("그리퍼 ON")
        elif cmd == "6":
            bot.grip(False)
            print("그리퍼 OFF")
        elif cmd == "7":
            p = bot.pose()
            print(f"위치: x={p[0]:.1f} y={p[1]:.1f} z={p[2]:.1f} r={p[3]:.1f}")
        elif cmd == "q":
            break
        else:
            print("1~7 또는 q를 입력하세요")

    bot.suck(False)
    bot.close()
    print("종료")


if __name__ == "__main__":
    main()
