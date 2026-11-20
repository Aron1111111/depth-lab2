"""C2 / C2: Erosion, dilation, opening and closing.
Run: python C2_morphology.py
"""
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"; OUT.mkdir(exist_ok=True)


def main():
    rng = np.random.default_rng(11)
    image = np.zeros((360, 520), np.uint8)
    cv2.rectangle(image, (90, 80), (430, 280), 255, -1)
    cv2.circle(image, (260, 180), 55, 0, -1)  # hole to fill
    for _ in range(80):  # white noise to remove
        x, y = int(rng.integers(0, 520)), int(rng.integers(0, 360))
        image[y, x] = 255
    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(image, kernel, iterations=1)
    dilation = cv2.dilate(image, kernel, iterations=1)
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    result = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    for name, value in {"input": image, "erosion": erosion, "dilation": dilation,
                        "opening": opening, "closing": closing, "result": result}.items():
        cv2.imwrite(str(OUT / f"{name}.png"), value)
    print("腐蚀、膨胀、开运算、闭运算结果已保存 / Результаты сохранены")
    print(f"output / 输出目录: {OUT}")


if __name__ == "__main__":
    main()
