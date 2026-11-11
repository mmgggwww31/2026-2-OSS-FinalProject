import cv2
import time

class RealtimeStreamer:
    def __init__(self, cam_id=0, width=1280, height=720):
        self.cam_id = cam_id
        self.width = width
        self.height = height
        self.cap = None
        self.is_running = False

    def start(self):
        """카메라 스트림 시작"""
        self.cap = cv2.VideoCapture(self.cam_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        if not self.cap.isOpened():
            print(f"[Error] 카메라({self.cam_id})를 열 수 없습니다.")
            return False
        
        self.is_running = True
        return True

    def get_frame(self):
        """실시간 프레임 읽기"""
        if not self.is_running or self.cap is None:
            return False, None
        
        ret, frame = self.cap.read()
        return ret, frame

    def stop(self):
        """카메라 자원 해제"""
        self.is_running = False
        if self.cap and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()

# 단독 테스트 실행용
if __name__ == "__main__":
    streamer = RealtimeStreamer(cam_id=0)
    if streamer.start():
        while True:
            ret, frame = streamer.get_frame()
            if not ret:
                break
            
            cv2.imshow("Realtime Test", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        streamer.stop()