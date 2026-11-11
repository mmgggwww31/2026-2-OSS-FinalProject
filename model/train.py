"""[feat/model] YOLO 안전모 탐지 모델 학습"""
import argparse
from pathlib import Path

from ultralytics import YOLO

DEFAULT_DATA = "CV-CBI-MINING SAFETY.v3i.yolov12/data.yaml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--weights", default="yolo11n.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--fraction", type=float, default=1.0,
                    help="사용할 데이터 비율 (시험용은 0.1)")
    ap.add_argument("--name", default="helmet_v1")
    args = ap.parse_args()

    data = Path(args.data).resolve()

    # 모델 선언: 사전학습된 YOLO 가중치를 불러온다
    model = YOLO(args.weights)

    # 학습
    model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        fraction=args.fraction,
        workers=0,  # Windows 데이터 로딩 에러 방지
        project="runs",
        name=args.name,
    )


if __name__ == "__main__":
    main()