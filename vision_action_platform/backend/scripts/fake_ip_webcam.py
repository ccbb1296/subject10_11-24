"""휴대폰 없이 IP Webcam을 흉내 내는 시험용 서버 (MJPEG over HTTP, DroidCam 의 /video 와 같은 방식).

영상 파일을 실제 속도로 반복 재생해 http://127.0.0.1:4750/video 로 내보냅니다.
.env 의 ALLOWED_CAMERA_URLS 에 이 주소를 넣으면 카메라 분석 흐름을 시험할 수 있습니다.

    python scripts/fake_ip_webcam.py ~/ai_vision/subject10_pose_action/data/day05/videos/S01/S01_A006_T01.mp4
"""

from __future__ import annotations

import argparse
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

BOUNDARY = "frame"


def make_handler(video_path: str):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/video":
                self.send_error(404)
                return
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            self.send_response(200)
            self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}")
            self.end_headers()
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:  # 끝나면 처음부터 다시
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    jpg = cv2.imencode(".jpg", frame)[1].tobytes()
                    self.wfile.write(
                        f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpg)}\r\n\r\n".encode()
                        + jpg + b"\r\n"
                    )
                    time.sleep(1 / fps)
            except (BrokenPipeError, ConnectionResetError):
                pass  # 수집 쪽이 연결을 끊음 (정상)
            finally:
                cap.release()

        def log_message(self, *args):
            pass

    return Handler


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--port", type=int, default=4750)
    args = parser.parse_args()
    print(f"가짜 IP Webcam: http://127.0.0.1:{args.port}/video  ←  {args.video}")
    ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(args.video)).serve_forever()
