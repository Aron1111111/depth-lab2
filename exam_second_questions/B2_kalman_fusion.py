"""B2 / B2: One-dimensional Kalman fusion of two sensors.
Run: python B2_kalman_fusion.py
"""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"; OUT.mkdir(exist_ok=True)


def update(x, p, z, r):
    k = p / (p + r)
    x = x + k * (z - x)
    p = (1 - k) * p
    return x, p


def main():
    # Camera/depth-like sensor is more precise than the second sensor.
    sensor_a = [10.2, 10.1, 9.9, 10.0, 10.1]
    sensor_b = [9.8, 10.0, 10.2, 9.7, 10.3]
    x, p, q = 0.0, 100.0, 0.01
    r_a, r_b = 0.04, 0.25
    rows = []
    for i, (za, zb) in enumerate(zip(sensor_a, sensor_b)):
        p += q  # prediction / прогноз
        x, p = update(x, p, za, r_a)
        x, p = update(x, p, zb, r_b)
        rows.append({"step": i, "sensor_a": za, "sensor_b": zb, "fused": x, "variance": p})
    (OUT / "kalman_metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("step | sensor A | sensor B | fused | variance")
    for row in rows:
        print(f"{row['step']:>4} | {row['sensor_a']:.2f}     | {row['sensor_b']:.2f}     | {row['fused']:.3f} | {row['variance']:.4f}")
    print(f"saved / 已保存: {OUT / 'kalman_metrics.json'}")


if __name__ == "__main__":
    main()
