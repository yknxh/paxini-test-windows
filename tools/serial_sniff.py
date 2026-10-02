"""Force gauge 시리얼 프로토콜 확인용 스니퍼.

기본은 수신만 한다 (명령을 보내지 않음). 보레이트를 순환하며 들어온 바이트를 그대로 출력하므로
게이지를 누르거나 게이지의 데이터 출력(PC/RS232) 모드를 켠 상태로 실행한다.

  python tools/serial_sniff.py COM7                  # 모든 보레이트 순환, 60초
  python tools/serial_sniff.py COM7 --baud 9600 --seconds 20
  python tools/serial_sniff.py COM7 --baud 19200 --send "D\r"   # 질의 명령을 0.2초마다 전송
"""
from __future__ import annotations

import argparse
import time

import serial

BAUDS = [9600, 19200, 2400, 4800, 38400, 57600, 115200]


def show(b: bytes) -> str:
    return "".join(chr(c) if 32 <= c < 127 else {13: "\r", 10: "\n"}.get(c, f"<{c:02X}>") for c in b)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("port")
    ap.add_argument("--baud", type=int, action="append", help="반복 지정 가능. 없으면 전체 순환")
    ap.add_argument("--seconds", type=float, default=60.0, help="전체 청취 시간")
    ap.add_argument("--dwell", type=float, default=3.0, help="보레이트 하나당 청취 시간")
    ap.add_argument("--send", default=None, help="주기적으로 보낼 질의 (\r, \n 이스케이프 허용)")
    args = ap.parse_args()
    bauds = args.baud or BAUDS
    cmd = args.send.encode().decode("unicode_escape").encode("latin1") if args.send else None
    end = time.time() + args.seconds
    total = {b: 0 for b in bauds}
    while time.time() < end:
        for b in bauds:
            if time.time() >= end:
                break
            with serial.Serial(args.port, b, timeout=0.05) as ser:
                ser.reset_input_buffer()
                stop = min(end, time.time() + args.dwell)
                next_send = 0.0
                while time.time() < stop:
                    if cmd and time.time() >= next_send:
                        ser.write(cmd)
                        next_send = time.time() + 0.2
                    data = ser.read(256)
                    if data:
                        total[b] += len(data)
                        print(f"{time.strftime('%H:%M:%S')} {b:>6} baud | {len(data):3d} B | {show(data)}", flush=True)
    print("수신 바이트:", total)


if __name__ == "__main__":
    main()
