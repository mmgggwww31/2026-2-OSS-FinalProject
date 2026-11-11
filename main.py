
# 1. 시스템 관련 라이브러리
import sys

# 2. GUI 실행에 필요한 라이브러리
from PySide6.QtWidgets import QApplication

# 3. 우리가 만든 메인 화면 클래스 가져오기
from ui.main_window import MainWindow


# 4. QApplication 객체 생성
app = QApplication(sys.argv)

# 5. 메인 윈도우 생성
window = MainWindow()

# 6. 화면에 창 표시
window.show()

# 7. GUI 이벤트 루프 실행
sys.exit(app.exec())
