"""
Raspberry Pi 5 Latency Benchmark — YOLOV10N INT8
Usage: python bench_yolov10n_int8.py
Requirements: pip install onnxruntime numpy Pillow
"""

import os, time, json
import numpy as np
from PIL import Image
import onnxruntime as ort

MODEL      = "yolov10n_int8.onnx"
IMGSZ      = 832
WARMUP     = 10
IMAGE_DIR  = "images"
OUTPUT     = "results_yolov10n_int8.json"

def preprocess(path, imgsz):
    img = Image.open(path).convert("RGB").resize((imgsz, imgsz))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))[np.newaxis, ...]

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(here, MODEL)
    image_dir  = os.path.join(here, IMAGE_DIR)

    images = sorted([
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    print(f"Model: {MODEL}")
    print(f"Images: {len(images)} | Resolution: {IMGSZ}x{IMGSZ} | Warmup: {WARMUP}")
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"Model size: {size_mb:.1f} MB")
    print()

    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    # Warmup
    warmup_img = preprocess(images[0], IMGSZ)
    for i in range(WARMUP):
        sess.run(None, {input_name: warmup_img})
        print(f"  Warmup {i+1}/{WARMUP}")

    # Timed runs
    latencies = []
    for i, img_path in enumerate(images):
        img_tensor = preprocess(img_path, IMGSZ)
        start = time.perf_counter()
        sess.run(None, {input_name: img_tensor})
        end = time.perf_counter()
        ms = (end - start) * 1000
        latencies.append(ms)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(images)} — last: {ms:.1f} ms")

    avg    = np.mean(latencies)
    std    = np.std(latencies)
    median = np.median(latencies)
    fps    = 1000.0 / avg

    print()
    print(f"===== {MODEL} =====")
    print(f"Avg:    {avg:.1f} ms")
    print(f"Median: {median:.1f} ms")
    print(f"Std:    {std:.1f} ms")
    print(f"FPS:    {fps:.2f}")
    print(f"Size:   {size_mb:.1f} MB")

    results = {
        "model": MODEL,
        "size_mb": round(size_mb, 1),
        "images": len(images),
        "imgsz": IMGSZ,
        "warmup": WARMUP,
        "avg_ms": round(avg, 1),
        "median_ms": round(median, 1),
        "std_ms": round(std, 1),
        "fps": round(fps, 2),
        "onnxruntime_version": ort.__version__,
        "latencies": [round(l, 1) for l in latencies],
    }

    out_path = os.path.join(here, OUTPUT)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT}")

if __name__ == "__main__":
    main()
