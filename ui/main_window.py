
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout,QListWidgetItem,
    QVBoxLayout, QLabel, QPushButton, QListWidget,QFrame 
)
from PySide6.QtCore import Qt
from datetime import datetime
from PySide6.QtGui import QColor


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Construction Safety Monitor")
        self.resize(1400, 850)

        # 메인 위젯
        central = QWidget()
        self.setCentralWidget(central)

        # 전체 레이아웃
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # -------------------------
        # 1. 왼쪽 사이드바
        # -------------------------

        sidebar = QFrame()
        sidebar.setFixedWidth(230)
        sidebar.setStyleSheet("""
            QFrame {
                background-color: #172238;
            }
        """)

        side_layout = QVBoxLayout(sidebar)

        logo = QLabel("🪖 Construction\nSafety Monitor")
        logo.setStyleSheet("""
            color: white;
            font-size: 20px;
            font-weight: bold;
            padding: 20px;
        """)

        side_layout.addWidget(logo)

        menus = [
            "📹 실시간 모니터링",
            "⚙ 설정",
            "📊 통계",
        ]

        for menu in menus:
            btn = QPushButton(menu)
            btn.setFixedHeight(55)

            btn.setStyleSheet("""
                QPushButton {
                    color: white;
                    background-color: #243653;
                    border: none;
                    text-align: left;
                    padding-left: 15px;
                    font-size: 15px;
                    border-radius: 8px;
                }

                QPushButton:hover {
                    background-color: #3478d4;
                }
            """)

            side_layout.addWidget(btn)

        side_layout.addStretch()

        # -------------------------
        # 2. 중앙 콘텐츠 영역
        # -------------------------

        content = QFrame()
        content.setStyleSheet("""
            QFrame {
                background-color: #0e1728;
            }
        """)

        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(15, 15, 15, 15)

        # 제목
        title = QLabel("실시간 안전장비 모니터링")
        title.setStyleSheet("""
            color: white;
            font-size: 25px;
            font-weight: bold;
            padding: 10px;
        """)

        content_layout.addWidget(title)

        # 영상과 오른쪽 패널을 배치할 가로 레이아웃
        middle_layout = QHBoxLayout()
        middle_layout.setSpacing(15)

        # -------------------------
        # 3. 영상 출력 영역
        # -------------------------

        video_area = QLabel("실시간 카메라 영상 영역")

        video_area.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        video_area.setStyleSheet("""
            background-color: #26364d;
            color: #cbd5e1;
            font-size: 24px;
            border: 2px solid #40516b;
            border-radius: 12px;
        """)

        middle_layout.addWidget(video_area, 3)

        # -------------------------
        # 4. 오른쪽 현황 패널
        # -------------------------

        right_panel = QVBoxLayout()
        right_panel.setSpacing(15)

        # 미착용 경고창
        warning_box = QFrame()
        warning_box.setStyleSheet("""
            QFrame {
                background-color: #dc3545;
                border-radius: 12px;
            }
        """)

        warning_layout = QVBoxLayout(warning_box)

        warning_title = QLabel("⚠ 안전장비 미착용!")
        warning_title.setStyleSheet("""
            color: white;
            font-size: 20px;
            font-weight: bold;
        """)

        warning_message = QLabel("미착용 작업자가 발견되었습니다.")
        warning_message.setStyleSheet("""
            color: white;
            font-size: 13px;
        """)

        warning_layout.addWidget(warning_title)
        warning_layout.addWidget(warning_message)

        right_panel.addWidget(warning_box)

        # 착용 현황 패널
        status_box = QFrame()
        status_box.setStyleSheet("""
            QFrame {
                background-color: #1b2942;
                border: 1px solid #344761;
                border-radius: 12px;
            }
        """)

        status_layout = QVBoxLayout(status_box)

        status_title = QLabel("🪖 안전모 착용 현황")
        status_title.setStyleSheet("""
            color: white;
            font-size: 18px;
            font-weight: bold;
        """)

        wearing = QLabel("착용 인원 : 2명")
        wearing.setStyleSheet("""
            color: #2ee88a;
            font-size: 16px;
        """)

        not_wearing = QLabel("미착용 인원 : 1명")
        not_wearing.setStyleSheet("""
            color: #ff4d5a;
            font-size: 16px;
        """)

        rate = QLabel("안전모 착용률 : 66.7%")
        rate.setStyleSheet("""
            color: white;
            font-size: 16px;
            font-weight: bold;
        """)

        status_layout.addWidget(status_title)
        status_layout.addWidget(wearing)
        status_layout.addWidget(not_wearing)
        status_layout.addWidget(rate)

        right_panel.addWidget(status_box)

        # 감지 인원 패널
        count_box = QFrame()
        count_box.setStyleSheet("""
            QFrame {
                background-color: #1b2942;
                border: 1px solid #344761;
                border-radius: 12px;
            }
        """)

        count_layout = QVBoxLayout(count_box)

        count_title = QLabel("👥 감지 객체 수")
        count_title.setStyleSheet("""
            color: white;
            font-size: 18px;
            font-weight: bold;
        """)

        count_value = QLabel("전체 감지 인원 : 3명")
        count_value.setStyleSheet("""
            color: white;
            font-size: 16px;
        """)

        count_layout.addWidget(count_title)
        count_layout.addWidget(count_value)

        right_panel.addWidget(count_box)

        # 오른쪽 패널 전체 배치
        content_layout.addLayout(middle_layout, 1)
       

        # -------------------------
        # 5. 전체 화면 결합
        # -------------------------

        main_layout.addWidget(sidebar)
        main_layout.addWidget(content, 1)


