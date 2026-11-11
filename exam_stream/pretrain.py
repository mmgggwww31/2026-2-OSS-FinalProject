import cv2
from stream import RealtimeStreamer  # stream.py 파일에서 가져오거나, 같은 파일에 클래스가 있어야 합니다.

# 1. streamer 객체 생성 및 카메라 시작
streamer = RealtimeStreamer(cam_id=0)

from ultralytics import YOLO

# 별도 학습 없이 공식 제공되는 가장 가벼운 기본 모델 자동 다운로드 (최초 1회)
model = YOLO('yolov8n.pt') 

ret, frame = streamer.get_frame()
if ret:
    # 일반 객체 추론 실행
    results = model(frame, stream=True, conf=0.5)
    
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            
            # 사람(person)이 감지되면 박스 표시 테스트
            if class_name == 'person':
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"Person Detected", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.imshow("Pre-trained Test", frame)