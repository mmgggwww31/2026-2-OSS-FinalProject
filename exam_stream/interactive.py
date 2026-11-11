import cv2

# 키보드 'w' 누르면 경고 온/오프 토글
is_warning_mode = False

while True:
    ret, frame = streamer.get_frame()
    if not ret: break

    key = cv2.waitKey(1) & 0xFF
    if key == ord('w'): # 'w' 키로 미착용 경고 발생 시뮬레이션
        is_warning_mode = not is_warning_mode
    elif key == ord('q'):
        break

    if is_warning_mode:
        # 경고 UI 렌더링
        cv2.putText(frame, "[TEST] WARNING ON", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Interactive Test", frame)