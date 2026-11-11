"""[feat/data] train에서 일부를 떼어 valid 폴더 만들기"""
import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--ratio", type=float, default=0.15)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

root = Path(args.root)
img_dir = root / "train" / "images"
lbl_dir = root / "train" / "labels"
valid = root / "valid"
if valid.exists():
    raise SystemExit("valid 폴더가 이미 있어요. 중단합니다.")

# Roboflow는 한 원본을 여러 장으로 증강하므로, 같은 원본끼리 묶어서 나눈다
groups = defaultdict(list)
for p in img_dir.iterdir():
    if p.suffix.lower() in IMG_EXTS:
        groups[p.name.split(".rf.")[0]].append(p)

keys = sorted(groups)
random.Random(args.seed).shuffle(keys)
target = int(sum(len(v) for v in groups.values()) * args.ratio)

(valid / "images").mkdir(parents=True)
(valid / "labels").mkdir(parents=True)

moved = 0
for k in keys:
    if moved >= target:
        break
    for img in groups[k]:
        shutil.move(str(img), valid / "images" / img.name)
        lbl = lbl_dir / f"{img.stem}.txt"
        if lbl.exists():
            shutil.move(str(lbl), valid / "labels" / lbl.name)
        moved += 1
print(f"valid로 이동: {moved}장")