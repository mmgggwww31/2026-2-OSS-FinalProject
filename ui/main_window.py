import os
import sys
import time
import json
import csv
import urllib.request
import cv2
from ultralytics import YOLO
from stream.stream import RealtimeStreamer

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QGridLayout, QLabel, QPushButton, QFrame, QProgressBar,
    QListWidget, QListWidgetItem, QStackedWidget, QButtonGroup,
    QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog, QMessageBox
)
from PySide6.QtCore import Qt, QTimer, QDateTime, QThread, Signal, QUrl
from PySide6.QtGui import QPixmap, QImage, QColor, QDesktopServices


# =========================================================
# 실시간 감지 위치 조회 스레드 (GUI 멈춤 방지)
# =========================================================
class LocationThread(QThread):
    location_ready = Signal(str, str)  # (지역명, 위도/경도 좌표)

    def run(self):
        try:
            url = "http://ip-api.com/json/?lang=ko"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=4) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            if data.get("status") == "success":
                region = data.get("regionName", "").strip()
                city = data.get("city", "").strip()
                lat = data.get("lat", 0.0)
                lon = data.get("lon", 0.0)

                if region and city and region != city and not region.startswith(city[:2]):
                    place = f"{region} {city}"
                else:
                    place = region or city or data.get("country", "위치 확인됨")

                now_str = QDateTime.currentDateTime().toString("HH:mm")
                coords = f"{lat:.4f}°N, {lon:.4f}°E ({now_str})"
                self.location_ready.emit(f"📍 {place}", coords)
            else:
                self.location_ready.emit("📍 위치 정보 조회 실패", "좌표 수신 불가")
        except Exception:
            self.location_ready.emit("📍 오프라인 / 위치 불가", "네트워크 연결 확인 필요")


# =========================================================
# 백그라운드 카메라 + YOLO 추론 스레드
# =========================================================
class DetectionThread(QThread):
    frame_ready = Signal(QImage, float)                 # (화면 프레임, FPS)
    detection_ready = Signal(int, int, object, list)    # (착용 인원, 미착용 인원, 원본 프레임, 미착용자 좌표 목록)
    error_occurred = Signal(str)

    def __init__(self, cam_id: int = 0, model_path: str = "yolov8n.pt"):
        super().__init__()
        self.cam_id = cam_id
        self.model_path = model_path
        self.running = False
        self.streamer = None

    def run(self):
        try:
            model = YOLO(self.model_path)
        except Exception as e:
            self.error_occurred.emit(f"YOLO 모델 로드 실패: {e}")
            return

        class_names = [name.lower() for name in model.names.values()]
        is_helmet_model = any("helmet" in name or "hardhat" in name for name in class_names)

        self.streamer = RealtimeStreamer(cam_id=self.cam_id)
        if not self.streamer.start():
            self.error_occurred.emit(f"카메라({self.cam_id}번)를 열 수 없습니다. 연결 상태를 확인하세요.")
            return

        self.running = True
        prev_time = time.time()

        while self.running:
            ret, frame = self.streamer.get_frame()
            if not ret or frame is None:
                self.error_occurred.emit("카메라로부터 영상을 받아오지 못했습니다.")
                break

            results = model(frame, conf=0.5, verbose=False)

            wearing = 0
            not_wearing = 0
            violator_boxes = []

            if is_helmet_model:
                annotated_frame = results[0].plot()
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    name = model.names[cls_id].lower()
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    if "no" in name or "without" in name or name == "head":
                        not_wearing += 1
                        violator_boxes.append((x1, y1, x2, y2))
                    elif "helmet" in name or "hardhat" in name:
                        wearing += 1
            else:
                annotated_frame = frame.copy()
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    if model.names[cls_id] == "person":
                        not_wearing += 1
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        violator_boxes.append((x1, y1, x2, y2))
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(
                            annotated_frame, "Person (No Helmet Test)", (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
                        )

            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0.0
            prev_time = curr_time

            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            q_img = QImage(rgb_frame.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()

            # 미착용자가 있을 때만 확대 촬영용 원본 프레임 복사 전달
            raw_copy = frame.copy() if not_wearing > 0 else None

            self.frame_ready.emit(q_img, fps)
            self.detection_ready.emit(wearing, not_wearing, raw_copy, violator_boxes)

        if self.streamer:
            self.streamer.stop()

    def stop(self):
        self.running = False
        self.wait()


# =========================================================
# 메인 관제 윈도우
# =========================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Construction Safety Monitor - AI 안전모 관제 시스템")
        self.resize(1440, 880)
        self.setMinimumSize(1200, 760)

        # 상태 및 설정 변수
        self.is_monitoring = False
        self.sound_enabled = True
        self.is_alert_active = None
        self.detection_thread = None
        self.location_thread = None
        self.model_path = os.path.join("models", "helmet_yolo26_pt.pt")
        self.current_location_name = "위치 조회 중..."

        # 확대 캡처 저장 폴더 생성
        self.capture_dir = os.path.abspath("captures")
        os.makedirs(self.capture_dir, exist_ok=True)

        # 경고 로그 및 자동 확대 촬영 최소 간격 (초)
        self.log_cooldown_sec = 5.0
        self.last_warning_log_time = 0.0

        # 통계 및 위반 이력 데이터 저장소
        self.violation_records = []
        self.total_violation_events = 0

        # 전체 스타일시트
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0b1120;
            }
            QWidget {
                font-family: 'Pretendard', 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif;
                color: #f8fafc;
            }
            QListWidget {
                background-color: #0f172a;
                border: 1px solid #1e293b;
                border-radius: 8px;
                padding: 6px;
                font-size: 13px;
            }
            QListWidget::item {
                padding: 8px 10px;
                border-bottom: 1px solid #1e293b;
                border-radius: 4px;
            }
            QTableWidget {
                background-color: #0f172a;
                alternate-background-color: #131c31;
                border: 1px solid #1e293b;
                border-radius: 10px;
                gridline-color: #1e293b;
                font-size: 13px;
                color: #e2e8f0;
            }
            QTableWidget::item {
                padding: 10px 8px;
            }
            QTableWidget::item:selected {
                background-color: #1e3a8a;
                color: white;
            }
            QHeaderView::section {
                background-color: #1e293b;
                color: #94a3b8;
                font-weight: 700;
                font-size: 13px;
                padding: 12px 8px;
                border: none;
                border-bottom: 1px solid #334155;
            }
        """)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 1. 페이지 전환 위젯
        self.pages = QStackedWidget()

        # 2. 왼쪽 사이드바
        sidebar = self._create_sidebar()
        main_layout.addWidget(sidebar)

        # 3. 오른쪽 메인 콘텐츠 페이지 등록
        self.monitoring_page = self._create_monitoring_page()
        self.history_page = self._create_history_page()

        self.pages.addWidget(self.monitoring_page)   # index 0: 실시간 모니터링
        self.pages.addWidget(self.history_page)      # index 1: 통계 및 이력
        self.pages.addWidget(self._create_placeholder_page("⚙ YOLO 모델 및 카메라 설정 (준비 중)"))  # index 2

        main_layout.addWidget(self.pages, 1)

        # 실시간 시계 타이머
        self.clock_timer = QTimer(self)
        self.clock_timer.timeout.connect(self._update_clock)
        self.clock_timer.start(1000)
        self._update_clock()

        # 실시간 감지 위치 초기화 및 60초마다 자동 갱신
        self.location_timer = QTimer(self)
        self.location_timer.timeout.connect(self._refresh_location)
        self.location_timer.start(60000)
        self._refresh_location()

        # 초기 상태 (정상 모드)
        self.update_detection_data(wearing=0, not_wearing=0)

    # ---------------------------------------------------------
    # 1. 왼쪽 사이드바
    # ---------------------------------------------------------
    def _create_sidebar(self) -> QFrame:
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(240)
        sidebar.setStyleSheet("""
            QFrame#sidebar {
                background-color: #111827;
                border-right: 1px solid #1f2937;
            }
        """)

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(16, 24, 16, 20)
        layout.setSpacing(10)

        logo_box = QFrame()
        logo_box.setStyleSheet("border: none; background: transparent;")
        logo_layout = QVBoxLayout(logo_box)
        logo_layout.setContentsMargins(6, 0, 6, 16)

        logo_title = QLabel("🪖 SAFETY AI")
        logo_title.setStyleSheet("color: #38bdf8; font-size: 22px; font-weight: 800; letter-spacing: 1px; background: transparent;")
        logo_sub = QLabel("공사현장 안전모 관제 시스템")
        logo_sub.setStyleSheet("color: #94a3b8; font-size: 13px; margin-top: 2px; background: transparent;")

        logo_layout.addWidget(logo_title)
        logo_layout.addWidget(logo_sub)
        layout.addWidget(logo_box)

        self.menu_group = QButtonGroup(self)
        self.menu_group.setExclusive(True)

        menus = [
            ("📹  실시간 모니터링", 0),
            ("📊  통계 및 이력", 1),
            ("⚙  시스템 설정", 2),
        ]

        for text, index in menus:
            btn = QPushButton(text)
            btn.setCheckable(True)
            btn.setFixedHeight(48)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setStyleSheet("""
                QPushButton {
                    color: #cbd5e1;
                    background-color: transparent;
                    border: none;
                    text-align: left;
                    padding-left: 16px;
                    font-size: 15px;
                    font-weight: 600;
                    border-radius: 10px;
                }
                QPushButton:hover {
                    background-color: #1e293b;
                    color: white;
                }
                QPushButton:checked {
                    background-color: #2563eb;
                    color: white;
                }
            """)
            if index == 0:
                btn.setChecked(True)

            self.menu_group.addButton(btn, index)
            layout.addWidget(btn)

        self.menu_group.idClicked.connect(self.pages.setCurrentIndex)
        layout.addStretch()

        # 하단 실시간 감지 위치 카드
        loc_card = QFrame()
        loc_card.setObjectName("locCard")
        loc_card.setStyleSheet("""
            QFrame#locCard {
                background-color: #1e293b;
                border: 1px solid #334155;
                border-radius: 10px;
            }
        """)
        loc_layout = QVBoxLayout(loc_card)
        loc_layout.setContentsMargins(14, 12, 14, 12)
        loc_layout.setSpacing(6)

        loc_header = QHBoxLayout()
        loc_header.setContentsMargins(0, 0, 0, 0)

        loc_status_lbl = QLabel("● 실시간 감지 위치")
        loc_status_lbl.setStyleSheet("color: #34d399; font-size: 11px; font-weight: 700; border: none; background: transparent;")

        btn_refresh_loc = QPushButton("↻")
        btn_refresh_loc.setToolTip("현재 위치 새로고침")
        btn_refresh_loc.setFixedSize(20, 20)
        btn_refresh_loc.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_refresh_loc.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #94a3b8;
                font-size: 13px;
                font-weight: bold;
                border: none;
            }
            QPushButton:hover { color: #38bdf8; }
        """)
        btn_refresh_loc.clicked.connect(self._refresh_location)

        loc_header.addWidget(loc_status_lbl)
        loc_header.addStretch()
        loc_header.addWidget(btn_refresh_loc)

        self.location_name_lbl = QLabel("📍 위치 조회 중...")
        self.location_name_lbl.setWordWrap(True)
        self.location_name_lbl.setStyleSheet("color: #f8fafc; font-size: 13px; font-weight: 700; border: none; background: transparent;")

        self.location_coord_lbl = QLabel("좌표 수신 대기 중")
        self.location_coord_lbl.setStyleSheet("color: #94a3b8; font-size: 11px; border: none; background: transparent;")

        loc_layout.addLayout(loc_header)
        loc_layout.addWidget(self.location_name_lbl)
        loc_layout.addWidget(self.location_coord_lbl)
        layout.addWidget(loc_card)

        return sidebar

    # ---------------------------------------------------------
    # 실시간 감지 위치 조회 및 화면 반영 메서드
    # ---------------------------------------------------------
    def _refresh_location(self):
        if self.location_thread is not None and self.location_thread.isRunning():
            return
        self.location_thread = LocationThread()
        self.location_thread.location_ready.connect(self._on_location_updated)
        self.location_thread.start()

    def _on_location_updated(self, place_text: str, coords_text: str):
        self.location_name_lbl.setText(place_text)
        self.location_coord_lbl.setText(coords_text)
        self.current_location_name = place_text.replace("📍 ", "").strip()

    # ---------------------------------------------------------
    # 2. 실시간 모니터링 메인 페이지 (index 0)
    # ---------------------------------------------------------
    def _create_monitoring_page(self) -> QFrame:
        page = QFrame()
        page.setObjectName("monitoringPage")
        page.setStyleSheet("""
            QFrame#monitoringPage {
                background-color: #0b1120;
                border: none;
            }
        """)

        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(18)

        header = QHBoxLayout()

        title_box = QVBoxLayout()
        title_box.setSpacing(4)
        self.title_lbl = QLabel("실시간 안전장비 착용 모니터링")
        self.title_lbl.setStyleSheet("font-size: 24px; font-weight: 800; color: #f8fafc; background: transparent;")
        self.subtitle_lbl = QLabel("AI 객체 인식을 통해 작업자의 안전모 착용 여부를 실시간으로 감시합니다.")
        self.subtitle_lbl.setStyleSheet("font-size: 13px; color: #94a3b8; background: transparent;")
        title_box.addWidget(self.title_lbl)
        title_box.addWidget(self.subtitle_lbl)

        header.addLayout(title_box)
        header.addStretch()

        self.clock_label = QLabel()
        self.clock_label.setStyleSheet("""
            background-color: #1e293b;
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 8px 14px;
            font-size: 14px;
            font-weight: 600;
            color: #e2e8f0;
        """)
        header.addWidget(self.clock_label)
        layout.addLayout(header)

        body_layout = QHBoxLayout()
        body_layout.setSpacing(18)

        # --- 왼쪽: 비디오 컨테이너 ---
        video_card = QFrame()
        video_card.setObjectName("videoCard")
        video_card.setStyleSheet("""
            QFrame#videoCard {
                background-color: #111827;
                border: 1px solid #1f2937;
                border-radius: 14px;
            }
        """)

        video_card_layout = QVBoxLayout(video_card)
        video_card_layout.setContentsMargins(14, 14, 14, 14)
        video_card_layout.setSpacing(12)

        self.video_label = QLabel("카메라 대기 중...\n하단 [감지 시작] 버튼을 눌러 모니터링을 시작하세요.")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("""
            background-color: #090d16;
            color: #64748b;
            font-size: 17px;
            font-weight: 600;
            border: 1px dashed #334155;
            border-radius: 10px;
        """)
        video_card_layout.addWidget(self.video_label, 1)

        control_bar = QHBoxLayout()
        control_bar.setSpacing(10)

        self.btn_toggle_stream = QPushButton("▶  감지 시작")
        self.btn_toggle_stream.setFixedHeight(40)
        self.btn_toggle_stream.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_toggle_stream.setStyleSheet(self._action_btn_style("#2563eb", "#1d4ed8"))
        self.btn_toggle_stream.clicked.connect(self._toggle_stream)

        self.btn_sound = QPushButton("🔔 경고음 ON")
        self.btn_sound.setFixedHeight(40)
        self.btn_sound.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_sound.setStyleSheet(self._action_btn_style("#334155", "#475569"))
        self.btn_sound.clicked.connect(self._toggle_sound)

        self.fps_label = QLabel("FPS: 0.0  |  Confidence: 0.50")
        self.fps_label.setStyleSheet("color: #94a3b8; font-size: 13px; border: none; background: transparent;")

        control_bar.addWidget(self.btn_toggle_stream)
        control_bar.addWidget(self.btn_sound)
        control_bar.addStretch()
        control_bar.addWidget(self.fps_label)

        video_card_layout.addLayout(control_bar)
        body_layout.addWidget(video_card, 3)

        # --- 오른쪽: 현황 및 경고 패널 ---
        right_panel = QVBoxLayout()
        right_panel.setSpacing(14)

        self.warning_box = QFrame()
        self.warning_box.setObjectName("warningBox")
        self.warning_layout = QVBoxLayout(self.warning_box)
        self.warning_layout.setContentsMargins(18, 16, 18, 16)
        self.warning_layout.setSpacing(4)

        self.warning_title = QLabel("✅ 전원 안전모 착용 양호")
        self.warning_title.setStyleSheet("color: white; font-size: 19px; font-weight: 800; border: none; background: transparent;")
        self.warning_message = QLabel("감지된 모든 작업자가 안전모를 정상 착용 중입니다.")
        self.warning_message.setWordWrap(True)
        self.warning_message.setStyleSheet("color: #d1fae5; font-size: 13px; border: none; background: transparent;")

        self.warning_layout.addWidget(self.warning_title)
        self.warning_layout.addWidget(self.warning_message)
        right_panel.addWidget(self.warning_box)

        status_box = QFrame()
        status_box.setObjectName("statusBox")
        status_box.setStyleSheet("""
            QFrame#statusBox {
                background-color: #111827;
                border: 1px solid #1f2937;
                border-radius: 14px;
            }
        """)
        status_layout = QVBoxLayout(status_box)
        status_layout.setContentsMargins(18, 16, 18, 18)
        status_layout.setSpacing(14)

        status_header = QLabel("🪖 실시간 착용 현황")
        status_header.setStyleSheet("font-size: 16px; font-weight: 700; color: #f1f5f9; border: none; background: transparent;")
        status_layout.addWidget(status_header)

        kpi_grid = QGridLayout()
        kpi_grid.setSpacing(10)

        self.total_val_lbl = self._create_kpi_box(kpi_grid, 0, "전체 감지", "0명", "#38bdf8")
        self.wearing_val_lbl = self._create_kpi_box(kpi_grid, 1, "착용 완료", "0명", "#34d399")
        self.not_wearing_val_lbl = self._create_kpi_box(kpi_grid, 2, "미착용", "0명", "#f87171")

        status_layout.addLayout(kpi_grid)

        rate_header = QHBoxLayout()
        rate_text = QLabel("안전모 착용률")
        rate_text.setStyleSheet("color: #94a3b8; font-size: 13px; border: none; background: transparent;")
        self.rate_percent_lbl = QLabel("100.0%")
        self.rate_percent_lbl.setStyleSheet("color: #f8fafc; font-size: 15px; font-weight: 700; border: none; background: transparent;")
        rate_header.addWidget(rate_text)
        rate_header.addStretch()
        rate_header.addWidget(self.rate_percent_lbl)

        self.rate_bar = QProgressBar()
        self.rate_bar.setRange(0, 100)
        self.rate_bar.setValue(100)
        self.rate_bar.setTextVisible(False)
        self.rate_bar.setFixedHeight(10)
        self.rate_bar.setStyleSheet("""
            QProgressBar {
                background-color: #1e293b;
                border-radius: 5px;
                border: none;
            }
            QProgressBar::chunk {
                background-color: #10b981;
                border-radius: 5px;
            }
        """)

        status_layout.addLayout(rate_header)
        status_layout.addWidget(self.rate_bar)
        right_panel.addWidget(status_box)

        log_box = QFrame()
        log_box.setObjectName("logBox")
        log_box.setStyleSheet("""
            QFrame#logBox {
                background-color: #111827;
                border: 1px solid #1f2937;
                border-radius: 14px;
            }
        """)
        log_layout = QVBoxLayout(log_box)
        log_layout.setContentsMargins(18, 16, 18, 16)
        log_layout.setSpacing(10)

        log_header = QHBoxLayout()
        log_title = QLabel("📋 감지 알림 로그")
        log_title.setStyleSheet("font-size: 16px; font-weight: 700; color: #f1f5f9; border: none; background: transparent;")
        btn_clear_log = QPushButton("지우기")
        btn_clear_log.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_clear_log.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #94a3b8;
                font-size: 12px;
                border: none;
            }
            QPushButton:hover { color: white; }
        """)
        btn_clear_log.clicked.connect(lambda: self.log_list.clear())

        log_header.addWidget(log_title)
        log_header.addStretch()
        log_header.addWidget(btn_clear_log)

        self.log_list = QListWidget()
        log_layout.addLayout(log_header)
        log_layout.addWidget(self.log_list, 1)

        right_panel.addWidget(log_box, 1)

        body_layout.addLayout(right_panel, 1)
        layout.addLayout(body_layout, 1)

        return page

    # ---------------------------------------------------------
    # 3. 통계 및 위반 이력 페이지 (index 1)
    # ---------------------------------------------------------
    def _create_history_page(self) -> QFrame:
        page = QFrame()
        page.setObjectName("historyPage")
        page.setStyleSheet("""
            QFrame#historyPage {
                background-color: #0b1120;
                border: none;
            }
        """)

        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(18)

        # [상단 헤더]
        header = QHBoxLayout()
        title_box = QVBoxLayout()
        title_box.setSpacing(4)
        title = QLabel("📊 안전모 미착용 통계 및 위반 이력")
        title.setStyleSheet("font-size: 24px; font-weight: 800; color: #f8fafc; background: transparent;")
        subtitle = QLabel("적발된 위반 횟수, 발생 시각 로그 및 미착용자 자동 확대 촬영 사진을 확인합니다.")
        subtitle.setStyleSheet("font-size: 13px; color: #94a3b8; background: transparent;")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)

        header.addLayout(title_box)
        header.addStretch()

        btn_open_dir = QPushButton("📂 캡처 폴더 열기")
        btn_open_dir.setFixedHeight(38)
        btn_open_dir.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_open_dir.setStyleSheet(self._action_btn_style("#0f766e", "#115e59"))
        btn_open_dir.clicked.connect(self._open_capture_folder)

        btn_export = QPushButton("📥 CSV 내보내기")
        btn_export.setFixedHeight(38)
        btn_export.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_export.setStyleSheet(self._action_btn_style("#2563eb", "#1d4ed8"))
        btn_export.clicked.connect(self._export_history_csv)

        btn_clear = QPushButton("🗑 이력 초기화")
        btn_clear.setFixedHeight(38)
        btn_clear.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_clear.setStyleSheet(self._action_btn_style("#334155", "#475569"))
        btn_clear.clicked.connect(self._clear_history)

        header.addWidget(btn_open_dir)
        header.addWidget(btn_export)
        header.addWidget(btn_clear)
        layout.addLayout(header)

        # [상단 통계 KPI 카드 2개 (누적 미착용 인원 제외됨)]
        kpi_layout = QHBoxLayout()
        kpi_layout.setSpacing(16)

        self.stat_events_lbl = self._create_stat_banner_card(
            kpi_layout, "🚨 누적 위반 발생 횟수", "0건", "경고 로그 기록 기준 총 적발 횟수", "#f87171"
        )
        self.stat_last_time_lbl = self._create_stat_banner_card(
            kpi_layout, "🕒 최근 위반 발생 시각", "-", "가장 마지막으로 감지된 위반 시간", "#38bdf8"
        )

        layout.addLayout(kpi_layout)

        # [하단 영역] 왼쪽: 위반 상세 로그 테이블(3) / 오른쪽: 위반자 확대 캡처 미리보기(2)
        bottom_layout = QHBoxLayout()
        bottom_layout.setSpacing(16)

        # 왼쪽 테이블 카드
        table_card = QFrame()
        table_card.setObjectName("tableCard")
        table_card.setStyleSheet("""
            QFrame#tableCard {
                background-color: #111827;
                border: 1px solid #1f2937;
                border-radius: 14px;
            }
        """)
        table_layout = QVBoxLayout(table_card)
        table_layout.setContentsMargins(18, 16, 18, 18)
        table_layout.setSpacing(12)

        table_header = QHBoxLayout()
        table_title = QLabel("📋 위반 발생 상세 로그 (행 클릭 시 확대 사진 확인)")
        table_title.setStyleSheet("font-size: 16px; font-weight: 700; color: #f1f5f9; border: none; background: transparent;")
        self.table_count_lbl = QLabel("총 0건의 위반 기록")
        self.table_count_lbl.setStyleSheet("font-size: 13px; color: #94a3b8; border: none; background: transparent;")

        table_header.addWidget(table_title)
        table_header.addStretch()
        table_header.addWidget(self.table_count_lbl)
        table_layout.addLayout(table_header)

        self.history_table = QTableWidget(0, 5)
        self.history_table.setHorizontalHeaderLabels([
            "순번", "위반 발생 일시", "감지 위치", "경고 상태", "확대 캡처 파일"
        ])
        self.history_table.verticalHeader().setVisible(False)
        self.history_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.history_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.history_table.setAlternatingRowColors(True)
        self.history_table.cellClicked.connect(self._on_history_row_clicked)

        header_view = self.history_table.horizontalHeader()
        header_view.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header_view.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header_view.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header_view.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header_view.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)

        table_layout.addWidget(self.history_table, 1)
        bottom_layout.addWidget(table_card, 3)

        # 오른쪽 위반자 확대 캡처 미리보기 카드
        preview_card = QFrame()
        preview_card.setObjectName("previewCard")
        preview_card.setStyleSheet("""
            QFrame#previewCard {
                background-color: #111827;
                border: 1px solid #1f2937;
                border-radius: 14px;
            }
        """)
        preview_layout = QVBoxLayout(preview_card)
        preview_layout.setContentsMargins(18, 16, 18, 18)
        preview_layout.setSpacing(12)

        preview_title = QLabel("📸 위반자 확대(Zoom-in) 캡처")
        preview_title.setStyleSheet("font-size: 16px; font-weight: 700; color: #f1f5f9; border: none; background: transparent;")
        preview_layout.addWidget(preview_title)

        self.capture_preview_lbl = QLabel("미착용자 감지 시\n인물 확대 사진이 여기에 표시됩니다.")
        self.capture_preview_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.capture_preview_lbl.setMinimumSize(280, 280)
        self.capture_preview_lbl.setStyleSheet("""
            background-color: #090d16;
            color: #64748b;
            font-size: 14px;
            border: 1px dashed #334155;
            border-radius: 10px;
        """)
        preview_layout.addWidget(self.capture_preview_lbl, 1)

        self.capture_info_lbl = QLabel("저장 위치: ./captures/")
        self.capture_info_lbl.setWordWrap(True)
        self.capture_info_lbl.setStyleSheet("color: #94a3b8; font-size: 12px; border: none; background: transparent;")
        preview_layout.addWidget(self.capture_info_lbl)

        bottom_layout.addWidget(preview_card, 2)
        layout.addLayout(bottom_layout, 1)

        return page

    def _create_stat_banner_card(self, parent_layout: QHBoxLayout, title: str, val: str, desc: str, color: str) -> QLabel:
        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background-color: #111827;
                border: 1px solid #1f2937;
                border-radius: 14px;
            }
        """)
        box = QVBoxLayout(card)
        box.setContentsMargins(20, 16, 20, 16)
        box.setSpacing(6)

        lbl_title = QLabel(title)
        lbl_title.setStyleSheet("color: #cbd5e1; font-size: 14px; font-weight: 600; border: none; background: transparent;")

        lbl_val = QLabel(val)
        lbl_val.setStyleSheet(f"color: {color}; font-size: 28px; font-weight: 800; border: none; background: transparent;")

        lbl_desc = QLabel(desc)
        lbl_desc.setStyleSheet("color: #64748b; font-size: 12px; border: none; background: transparent;")

        box.addWidget(lbl_title)
        box.addWidget(lbl_val)
        box.addWidget(lbl_desc)
        parent_layout.addWidget(card, 1)
        return lbl_val

    # ---------------------------------------------------------
    # 미착용자 자동 확대(Zoom-in) 사진 촬영 및 기록 메서드
    # ---------------------------------------------------------
    def _capture_zoomed_violators(self, raw_frame, violator_boxes: list, timestamp_str: str) -> str:
        """탐지된 미착용자 영역을 여유 여백과 함께 확대 크롭해 이미지 파일로 저장"""
        if raw_frame is None or not violator_boxes:
            return ""

        h_img, w_img = raw_frame.shape[:2]
        saved_paths = []

        for idx, (x1, y1, x2, y2) in enumerate(violator_boxes, start=1):
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)

            # 인물 얼굴과 상반신이 함께 잘 잡히도록 바운딩 박스 기준 여유 여백 부여 (Zoom-in 프레이밍)
            pad_x = int(bw * 0.35)
            pad_top = int(bh * 0.35)
            pad_bottom = int(bh * 0.55)

            cx1 = max(0, x1 - pad_x)
            cy1 = max(0, y1 - pad_top)
            cx2 = min(w_img, x2 + pad_x)
            cy2 = min(h_img, y2 + pad_bottom)

            cropped = raw_frame[cy1:cy2, cx1:cx2].copy()
            if cropped.size == 0:
                continue

            # 작은 영역일 경우 선명하게 확대(Upscale)
            ch, cw = cropped.shape[:2]
            if cw < 480:
                scale = 480.0 / cw
                cropped = cv2.resize(
                    cropped, (int(cw * scale), int(ch * scale)), interpolation=cv2.INTER_CUBIC
                )

            # 사진 하단에 촬영 시각 워터마크 바 추가
            ch, cw = cropped.shape[:2]
            cv2.rectangle(cropped, (0, ch - 32), (cw, ch), (20, 20, 30), -1)
            cv2.putText(
                cropped, f"VIOLATION CAPTURE | {timestamp_str}", (10, ch - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 130, 255), 2
            )

            file_tag = QDateTime.currentDateTime().toString("yyyyMMdd_HHmmss")
            filename = f"violation_{file_tag}_{idx}.jpg"
            save_path = os.path.join(self.capture_dir, filename)

            cv2.imwrite(save_path, cropped)
            saved_paths.append(save_path)

        return saved_paths[0] if saved_paths else ""

    def _show_capture_preview(self, image_path: str, info_text: str = ""):
        if image_path and os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            scaled = pixmap.scaled(
                self.capture_preview_lbl.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.capture_preview_lbl.setPixmap(scaled)
            self.capture_info_lbl.setText(info_text or f"파일명: {os.path.basename(image_path)}")

    def _on_history_row_clicked(self, row: int, _col: int):
        if 0 <= row < len(self.violation_records):
            rec = self.violation_records[row]
            self._show_capture_preview(
                rec["image_path"],
                f"발생 일시: {rec['timestamp']}  |  파일: {os.path.basename(rec['image_path'])}"
            )

    def _record_violation_event(self, wearing: int, not_wearing: int, raw_frame=None, violator_boxes=None):
        """위반 발생 시 확대 사진 촬영 + 통계 수치 + 위반 이력 테이블 동시 갱신"""
        self.total_violation_events += 1

        now_dt = QDateTime.currentDateTime()
        full_time = now_dt.toString("yyyy-MM-dd  HH:mm:ss")
        short_time = now_dt.toString("HH:mm:ss")

        # 미착용자 확대 캡처 실행
        saved_image_path = self._capture_zoomed_violators(raw_frame, violator_boxes or [], full_time)
        filename_display = os.path.basename(saved_image_path) if saved_image_path else "저장 실패"

        record = {
            "no": self.total_violation_events,
            "timestamp": full_time,
            "location": self.current_location_name,
            "status": f"안전모 미착용 ({not_wearing}명)",
            "image_path": saved_image_path,
            "filename": filename_display
        }
        self.violation_records.insert(0, record)

        # 통계 카드 업데이트
        self.stat_events_lbl.setText(f"{self.total_violation_events}건")
        self.stat_last_time_lbl.setText(short_time)
        self.table_count_lbl.setText(f"총 {self.total_violation_events}건의 위반 기록")

        # 미리보기 패널에 방금 찍힌 확대 사진 반영
        if saved_image_path:
            self._show_capture_preview(
                saved_image_path,
                f"최근 촬영: {full_time}  |  파일: {filename_display}"
            )

        # 테이블 최상단에 행 추가
        self.history_table.insertRow(0)
        items = [
            (str(record["no"]), "#94a3b8"),
            (record["timestamp"], "#f8fafc"),
            (record["location"], "#cbd5e1"),
            (record["status"], "#f87171"),
            (record["filename"], "#38bdf8")
        ]
        for col, (text, color_hex) in enumerate(items):
            cell = QTableWidgetItem(text)
            cell.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            cell.setForeground(QColor(color_hex))
            self.history_table.setItem(0, col, cell)

    def _open_capture_folder(self):
        os.makedirs(self.capture_dir, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(self.capture_dir))

    def _clear_history(self):
        self.violation_records.clear()
        self.total_violation_events = 0

        self.stat_events_lbl.setText("0건")
        self.stat_last_time_lbl.setText("-")
        self.table_count_lbl.setText("총 0건의 위반 기록")
        self.history_table.setRowCount(0)
        self.capture_preview_lbl.clear()
        self.capture_preview_lbl.setText("미착용자 감지 시\n인물 확대 사진이 여기에 표시됩니다.")
        self.capture_info_lbl.setText("저장 위치: ./captures/")

    def _export_history_csv(self):
        if not self.violation_records:
            QMessageBox.information(self, "알림", "내보낼 위반 이력 데이터가 없습니다.")
            return

        default_name = f"safety_violations_{QDateTime.currentDateTime().toString('yyyyMMdd_HHmmss')}.csv"
        file_path, _ = QFileDialog.getSaveFileName(self, "위반 이력 CSV 저장", default_name, "CSV Files (*.csv)")
        if not file_path:
            return

        try:
            with open(file_path, mode="w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(["순번", "위반 발생 일시", "감지 위치", "경고 상태", "확대 캡처 파일"])
                for r in self.violation_records:
                    writer.writerow([r["no"], r["timestamp"], r["location"], r["status"], r["filename"]])
            QMessageBox.information(self, "저장 완료", "위반 이력이 CSV 파일로 저장되었습니다.")
        except Exception as e:
            QMessageBox.warning(self, "저장 실패", f"파일 저장 중 오류가 발생했습니다:\n{e}")

    # ---------------------------------------------------------
    # 전체 화면 경고 테마 전환 (우측 메인 캔버스 빨간색 전환)
    # ---------------------------------------------------------
    def _set_alert_theme(self, is_alert: bool):
        if self.is_alert_active == is_alert:
            return
        self.is_alert_active = is_alert

        if is_alert:
            self.monitoring_page.setStyleSheet("""
                QFrame#monitoringPage {
                    background-color: #dc2626;
                    border: none;
                }
            """)
            self.subtitle_lbl.setStyleSheet("font-size: 13px; color: #fee2e2; background: transparent;")
            self.warning_box.setStyleSheet("""
                QFrame#warningBox {
                    background-color: #7f1d1d;
                    border: 2px solid #fecaca;
                    border-radius: 14px;
                }
            """)
        else:
            self.monitoring_page.setStyleSheet("""
                QFrame#monitoringPage {
                    background-color: #0b1120;
                    border: none;
                }
            """)
            self.subtitle_lbl.setStyleSheet("font-size: 13px; color: #94a3b8; background: transparent;")
            self.warning_box.setStyleSheet("""
                QFrame#warningBox {
                    background-color: #059669;
                    border: 1px solid #34d399;
                    border-radius: 14px;
                }
            """)

    # ---------------------------------------------------------
    # UI 보조 위젯 생성 함수
    # ---------------------------------------------------------
    def _create_kpi_box(self, grid: QGridLayout, col: int, title: str, initial_val: str, color: str) -> QLabel:
        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background-color: #1e293b;
                border: 1px solid #334155;
                border-radius: 10px;
            }
        """)
        box = QVBoxLayout(card)
        box.setContentsMargins(10, 12, 10, 12)
        box.setSpacing(4)
        box.setAlignment(Qt.AlignmentFlag.AlignCenter)

        lbl_title = QLabel(title)
        lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_title.setStyleSheet("color: #94a3b8; font-size: 12px; border: none; background: transparent;")

        lbl_val = QLabel(initial_val)
        lbl_val.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_val.setStyleSheet(f"color: {color}; font-size: 20px; font-weight: 800; border: none; background: transparent;")

        box.addWidget(lbl_title)
        box.addWidget(lbl_val)
        grid.addWidget(card, 0, col)
        return lbl_val

    def _create_placeholder_page(self, text: str) -> QWidget:
        page = QWidget()
        page.setStyleSheet("background-color: #0b1120;")
        layout = QVBoxLayout(page)
        lbl = QLabel(text)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet("color: #64748b; font-size: 20px; font-weight: bold;")
        layout.addWidget(lbl)
        return page

    def _action_btn_style(self, bg: str, hover: str) -> str:
        return f"""
            QPushButton {{
                background-color: {bg};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 0 16px;
                font-size: 14px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background-color: {hover};
            }}
        """

    # ---------------------------------------------------------
    # 카메라 & AI 추론 제어
    # ---------------------------------------------------------
    def _update_clock(self):
        now = QDateTime.currentDateTime().toString("yyyy-MM-dd  HH:mm:ss")
        self.clock_label.setText(f"🕒 {now}")

    def _toggle_stream(self):
        if not self.is_monitoring:
            self._start_detection()
        else:
            self._stop_detection()

    def _start_detection(self):
        self.is_monitoring = True
        self.btn_toggle_stream.setText("⏹  감지 중지")
        self.btn_toggle_stream.setStyleSheet(self._action_btn_style("#dc2626", "#b91c1c"))
        self.video_label.setText("● YOLO 모델 및 웹캠 로딩 중...\n(최초 실행 시 모델 다운로드로 5~10초 정도 걸릴 수 있습니다)")
        self._add_log("시스템: 카메라 및 YOLO 모델 연결을 시작합니다.", "#38bdf8")

        self.detection_thread = DetectionThread(cam_id=0, model_path=self.model_path)
        self.detection_thread.frame_ready.connect(self.update_video_frame)
        self.detection_thread.detection_ready.connect(self.update_detection_data)
        self.detection_thread.error_occurred.connect(self._on_detection_error)
        self.detection_thread.start()

    def _stop_detection(self):
        self.is_monitoring = False
        if self.detection_thread is not None:
            self.detection_thread.stop()
            self.detection_thread = None

        self._set_alert_theme(False)
        self.btn_toggle_stream.setText("▶  감지 시작")
        self.btn_toggle_stream.setStyleSheet(self._action_btn_style("#2563eb", "#1d4ed8"))
        self.video_label.clear()
        self.video_label.setText("모니터링이 중지되었습니다.\n[감지 시작] 버튼을 누르면 다시 연결됩니다.")
        self.fps_label.setText("FPS: 0.0  |  Confidence: 0.50")
        self._add_log("시스템: 모니터링을 중지했습니다.", "#94a3b8")

    def _on_detection_error(self, error_msg: str):
        self._add_log(f"오류: {error_msg}", "#f87171")
        self._stop_detection()
        self.video_label.setText(f"⚠ {error_msg}")

    def _toggle_sound(self):
        self.sound_enabled = not self.sound_enabled
        if self.sound_enabled:
            self.btn_sound.setText("🔔 경고음 ON")
            self.btn_sound.setStyleSheet(self._action_btn_style("#334155", "#475569"))
        else:
            self.btn_sound.setText("🔕 경고음 OFF")
            self.btn_sound.setStyleSheet(self._action_btn_style("#1e293b", "#334155"))

    def _add_log(self, message: str, color_hex: str = "#e2e8f0"):
        time_str = QDateTime.currentDateTime().toString("HH:mm:ss")
        item = QListWidgetItem(f"[{time_str}] {message}")
        item.setForeground(QColor(color_hex))
        self.log_list.insertItem(0, item)
        if self.log_list.count() > 50:
            self.log_list.takeItem(50)

    def update_video_frame(self, q_image: QImage, fps: float = 0.0):
        if not self.is_monitoring:
            return
        pixmap = QPixmap.fromImage(q_image)
        scaled = pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.video_label.setPixmap(scaled)
        self.fps_label.setText(f"FPS: {fps:.1f}  |  Confidence: 0.50")

    def update_detection_data(self, wearing: int, not_wearing: int, raw_frame=None, violator_boxes=None):
        total = wearing + not_wearing
        rate = (wearing / total * 100.0) if total > 0 else 100.0

        self.total_val_lbl.setText(f"{total}명")
        self.wearing_val_lbl.setText(f"{wearing}명")
        self.not_wearing_val_lbl.setText(f"{not_wearing}명")
        self.rate_percent_lbl.setText(f"{rate:.1f}%")
        self.rate_bar.setValue(int(rate))

        if not_wearing > 0:
            self._set_alert_theme(True)
            self.warning_title.setText(f"⚠ 안전장비 미착용 ({not_wearing}명)")
            self.warning_message.setText("미착용 작업자가 감지되었습니다. 안전 조치가 필요합니다.")
            self.warning_message.setStyleSheet("color: #fee2e2; font-size: 13px; border: none; background: transparent;")

            now = time.time()
            if (now - self.last_warning_log_time) >= self.log_cooldown_sec:
                self.last_warning_log_time = now
                self._add_log(f"경고: 안전모 미착용 작업자 {not_wearing}명 감지! (확대 캡처 저장)", "#f87171")
                self._record_violation_event(
                    wearing=wearing,
                    not_wearing=not_wearing,
                    raw_frame=raw_frame,
                    violator_boxes=violator_boxes
                )
                if self.sound_enabled:
                    QApplication.beep()
        else:
            self._set_alert_theme(False)
            self.warning_title.setText("✅ 전원 안전모 착용 양호")
            self.warning_message.setText("감지된 모든 작업자가 안전모를 정상 착용 중입니다.")
            self.warning_message.setStyleSheet("color: #d1fae5; font-size: 13px; border: none; background: transparent;")

    def closeEvent(self, event):
        self._stop_detection()
        if self.location_thread is not None and self.location_thread.isRunning():
            self.location_thread.wait(500)
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())