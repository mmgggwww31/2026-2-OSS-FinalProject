
from ultralytics import YOLO
import os
import csv

# 테스트 데이터 경로
DATA_YAML = r"C:\Users\user\Downloads\CV-CBI-MINING SAFETY.v3i.yolov12\data.yaml"

# 평가 대상 모델과 결과 저장 폴더
MODELS = {
    "helmet_v2-3": "models/helmet_v2-3.pt",
    "helmet_v3i": "models/helmet_v3i.pt",
    "helmet_yolo26_pt": "models/helmet_yolo26_pt.pt"
}

results = []

for model_name, model_path in MODELS.items():
    print(f"\n{'=' * 50}")
    print(f"{model_name} 평가 시작")
    print(f"{'=' * 50}")

    model = YOLO(model_path)

    metrics = model.val(
        data=DATA_YAML,
        split="test",
        imgsz=640,
        project=f"analysis/{model_name}/evaluation",
        name="results",
        exist_ok=True
    )

    result = {
        "model": model_name,
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map)
    }

    results.append(result)

    # 모델별 평가 결과 저장
    output_path = f"analysis/{model_name}/evaluation/summary.csv"

    with open(output_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=result.keys())
        writer.writeheader()
        writer.writerow(result)

    print(f"{model_name} 평가 완료")
    print(f"mAP50-95: {result['mAP50-95']:.4f}")

# 전체 모델 비교 결과 저장
baseline = results[0]["mAP50-95"]

comparison_path = "analysis/comparison/model_comparison.csv"

with open(comparison_path, "w", newline="", encoding="utf-8-sig") as file:
    fields = [
        "model", "precision", "recall",
        "mAP50", "mAP50-95", "improvement_percent"
    ]

    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()

    for result in results:
        improvement = (
            (result["mAP50-95"] - baseline) / baseline * 100
            if baseline != 0 else 0
        )

        writer.writerow({
            **result,
            "improvement_percent": round(improvement, 2)
        })

print("\n===== 최종 성능 비교 =====")

for result in results:
    improvement = (
        (result["mAP50-95"] - baseline) / baseline * 100
        if baseline != 0 else 0
    )

    print(
        f"{result['model']}: "
        f"mAP50-95 {result['mAP50-95']:.4f} "
        f"({improvement:+.2f}%)"
    )

print("\n모든 평가가 완료되었습니다.")
