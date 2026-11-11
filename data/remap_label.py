"""[feat/data] 라벨 클래스 번호 변환 + 클래스별 박스 개수 세기"""
import argparse
from collections import Counter
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="train/valid/test가 들어 있는 폴더")
    ap.add_argument("--remap", default=None,
                    help='"원본:새번호,..." 목록에 없는 클래스는 삭제. 없으면 개수만 셈')
    args = ap.parse_args()

    remap = None
    if args.remap:
        remap = {int(a): int(b) for a, b in (p.split(":") for p in args.remap.split(","))}

    for split in ("train", "valid", "test"):
        folder = Path(args.root) / split / "labels"
        if not folder.exists():
            continue
        counts = Counter()
        for txt in folder.glob("*.txt"):
            out = []
            for line in txt.read_text().splitlines():
                parts = line.split()
                if not parts:
                    continue
                cls = int(parts[0])
                if remap is not None:
                    if cls not in remap:
                        continue
                    cls = remap[cls]
                counts[cls] += 1
                out.append(" ".join([str(cls)] + parts[1:]))
            if remap is not None:
                txt.write_text("\n".join(out))
        print(split, dict(sorted(counts.items())))


if __name__ == "__main__":
    main()