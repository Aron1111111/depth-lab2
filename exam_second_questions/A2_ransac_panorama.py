"""A2 / A2: Panorama alignment with RANSAC.
Run: python A2_ransac_panorama.py
"""
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"; OUT.mkdir(exist_ok=True)


def make_demo():
    rng = np.random.default_rng(7)
    img = np.full((420, 760, 3), 245, np.uint8)
    cv2.rectangle(img, (30, 35), (730, 385), (40, 120, 190), 4)
    for _ in range(35):
        x, y = int(rng.integers(50, 710)), int(rng.integers(55, 365))
        cv2.circle(img, (x, y), 5, (20, 40, 80), -1)
    cv2.putText(img, "RANSAC PANORAMA", (160, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 70, 140), 3)
    left = img[:, :520].copy()
    right = img[:, 240:].copy()
    return left, right


def main():
    left, right = make_demo()
    cv2.imwrite(str(OUT / "input_left.png"), left)
    cv2.imwrite(str(OUT / "input_right.png"), right)
    sift = cv2.SIFT_create()
    k1, d1 = sift.detectAndCompute(cv2.cvtColor(left, cv2.COLOR_BGR2GRAY), None)
    k2, d2 = sift.detectAndCompute(cv2.cvtColor(right, cv2.COLOR_BGR2GRAY), None)
    if d1 is None or d2 is None:
        raise RuntimeError("没有检测到足够的局部特征 / Недостаточно локальных признаков")
    pairs = cv2.BFMatcher().knnMatch(d2, d1, k=2)
    good = [m[0] for m in pairs if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
    if len(good) < 4:
        raise RuntimeError("匹配点少于4对 / Меньше четырёх соответствий")
    src = np.float32([k2[m.queryIdx].pt for m in good])
    dst = np.float32([k1[m.trainIdx].pt for m in good])
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if H is None:
        raise RuntimeError("RANSAC 单应矩阵估计失败 / Не удалось оценить гомографию")
    h, w = left.shape[:2]
    pano = cv2.warpPerspective(right, H, (w + right.shape[1], h))
    pano[:h, :w] = np.maximum(pano[:h, :w], left)
    cv2.imwrite(str(OUT / "panorama_ransac.png"), pano)
    inlier_ratio = float(mask.mean())
    print(f"good matches / 有效匹配: {len(good)}")
    print(f"inlier ratio / 内点比例: {inlier_ratio:.3f}")
    print(f"saved / 已保存: {OUT / 'panorama_ransac.png'}")


if __name__ == "__main__":
    main()
