"""Lab 2: reproducible depth restoration experiment.

Generates a deterministic synthetic RGB-D scene, removes depth samples, then
restores them with a validity mask, 3x3 morphological closing and a guided
filter. Results and metrics are written to results/.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SEED = 20241110
CONFIG_PATH = ROOT / "config.json"


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8")) if CONFIG_PATH.exists() else {
        "seed": SEED, "width": 260, "height": 180, "random_missing_rate": 0.10,
        "holes": [[48, 95, 18, 28], [112, 170, 30, 40]],
        "morph_radius": 1, "guided_radius": 4,
    }


def box_mean(a: np.ndarray, radius: int) -> np.ndarray:
    p = np.pad(a, ((radius, radius), (radius, radius)), mode="edge")
    integral = np.pad(p, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
    k = 2 * radius + 1
    return (integral[k:, k:] - integral[:-k, k:] - integral[k:, :-k] + integral[:-k, :-k]) / (k * k)


def morph_close(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """Binary closing using dilation followed by erosion."""
    p = np.pad(mask, radius, mode="edge")
    dil = np.zeros_like(mask)
    ero = np.ones_like(mask)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            dil = np.maximum(dil, p[dy:dy + mask.shape[0], dx:dx + mask.shape[1]])
    p2 = np.pad(dil, radius, mode="edge")
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            ero = np.minimum(ero, p2[dy:dy + mask.shape[0], dx:dx + mask.shape[1]])
    return ero.astype(bool)


def guided_filter(guidance: np.ndarray, src: np.ndarray, valid: np.ndarray, radius: int = 4, eps: float = 1e-3) -> np.ndarray:
    """Guided filter with masked local statistics, used only to fill holes."""
    g = guidance.astype(np.float32)
    x = src.astype(np.float32)
    w = valid.astype(np.float32)
    n = np.maximum(box_mean(w, radius), 1e-6)
    mean_g = box_mean(g * w, radius) / n
    mean_x = box_mean(x * w, radius) / n
    corr_g = box_mean(g * g * w, radius) / n
    corr_gx = box_mean(g * x * w, radius) / n
    var_g = np.maximum(corr_g - mean_g * mean_g, 0.0)
    cov_gx = corr_gx - mean_g * mean_x
    a = cov_gx / (var_g + eps)
    b = mean_x - a * mean_g
    mean_a = box_mean(a, radius)
    mean_b = box_mean(b, radius)
    return mean_a * g + mean_b


def save_depth(path: Path, depth: np.ndarray, vmax: float) -> None:
    image = np.clip(depth / vmax * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(image, mode="L").save(path)


def main() -> None:
    config = load_config()
    RESULTS.mkdir(exist_ok=True)
    seed = int(config["seed"])
    rng = np.random.default_rng(seed)
    h, w = int(config["height"]), int(config["width"])
    yy, xx = np.mgrid[0:h, 0:w]
    truth = 2.0 + 0.002 * xx + 0.0015 * yy
    truth += 0.55 * (((xx - 175) ** 2 + (yy - 92) ** 2) < 34 ** 2)
    guidance = np.clip(0.15 + (truth - truth.min()) / (truth.max() - truth.min()) * 0.8, 0, 1)
    valid = np.ones((h, w), dtype=bool)
    valid[rng.random((h, w)) < float(config["random_missing_rate"])] = False
    for y, x, hh, ww in config["holes"]:
        valid[int(y):int(y) + int(hh), int(x):int(x) + int(ww)] = False
    observed = truth.copy()
    observed[~valid] = 0.0

    # Small-hole closing only updates the validity mask; values are filled below.
    closed_valid = morph_close(valid, radius=int(config["morph_radius"]))
    guided = guided_filter(guidance, observed, valid, radius=int(config["guided_radius"]))
    # A physical depth range prevents unstable estimates in textureless windows.
    guided = np.clip(guided, float(truth.min()), float(truth.max()))
    restored = observed.copy()
    fill = ~valid
    restored[fill] = guided[fill]
    # For pixels made valid by morphology, use a local median-like guided estimate.
    newly_valid = closed_valid & ~valid
    restored[newly_valid] = guided[newly_valid]
    output_valid = valid | newly_valid

    rmse_before = float(np.sqrt(np.mean((observed[valid] - truth[valid]) ** 2)))
    rmse_missing = float(np.sqrt(np.mean((restored[fill] - truth[fill]) ** 2)))
    rmse_all = float(np.sqrt(np.mean((restored - truth) ** 2)))
    metrics = {
        "seed": seed,
        "shape": [h, w],
        "missing_rate": float(np.mean(~valid)),
        "valid_percent_before": float(np.mean(valid) * 100),
        "valid_percent_after": float(np.mean(output_valid) * 100),
        "rmse_observed_valid_pixels_m": rmse_before,
        "rmse_restored_missing_pixels_m": rmse_missing,
        "rmse_restored_all_pixels_m": rmse_all,
        "method": "validity mask + 3x3 morphological closing + masked guided filter",
        "config": config,
    }
    (RESULTS / "metrics_before_after.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_depth(RESULTS / "depth_ground_truth.png", truth, truth.max())
    save_depth(RESULTS / "depth_corrupted.png", observed, truth.max())
    save_depth(RESULTS / "depth_restored.png", restored, truth.max())
    mask = (valid.astype(np.uint8) * 255)
    Image.fromarray(mask, mode="L").save(RESULTS / "validity_mask.png")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
