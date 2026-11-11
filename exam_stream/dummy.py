import cv2

# streamer.get_frame()으로 읽어온 frame 위에 가상 박스 그리기
ret, frame = streamer.get_frame()
if ret:
    # 1. 착용자 가상 박스 (초록색)
    cv2.rectangle(frame, (100, 100), (300, 400), (0, 255, 0), 2)
    cv2.putText(frame, "HELMET (0.95)", (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 2. 미착용자 가상 박스 (빨간색) - 미착용 감지 상황 강제 연출
    cv2.rectangle(frame, (400, 100), (600, 400), (0, 0, 255), 2)
    cv2.putText(frame, "WARNING: NO-HELMET (0.88)", (400, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # 3. 미착용 감지시 상단 배너 표시 테스트
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (0, 0, 255), -1)
    cv2.putText(frame, "PPE CHECK WARNING! UNAPPROVED ACCESS DETECTED", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Dummy Test", frame)