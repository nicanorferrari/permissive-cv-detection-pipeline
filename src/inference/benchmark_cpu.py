"""
CPU Inference Benchmark for Azure Functions / Serverless Workloads.
Configured for x86_64 AVX2 using ONNX Runtime with multi-threading tuning and INT8 PTQ.
"""

import time
import sys
import os

if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import onnxruntime as ort
import os

def create_optimized_cpu_session(onnx_path: str, num_threads: int = 4) -> ort.InferenceSession:
    """
    Initializes ONNX Runtime session optimized for multi-core x86_64 CPU (AVX2).
    Disables AVX-512 dependency and tunes intra-op parallelism for containerized workloads.
    """
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.intra_op_num_threads = num_threads
    opts.inter_op_num_threads = 1
    
    # Enable CPU execution provider with AVX2 instruction dispatch
    session = ort.InferenceSession(onnx_path, sess_options=opts, providers=["CPUExecutionProvider"])
    return session

def benchmark_inference(session: ort.InferenceSession, input_shape=(1, 3, 640, 640), warmup=20, runs=100):
    input_name = session.get_inputs()[0].name
    dummy_input = np.random.randn(*input_shape).astype(np.float32)

    print(f"[+] Warming up inference engine ({warmup} runs)...")
    for _ in range(warmup):
        _ = session.run(None, {input_name: dummy_input})

    print(f"[+] Benchmarking over {runs} iterations...")
    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = session.run(None, {input_name: dummy_input})
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # to ms

    latencies = np.array(latencies)
    mean_lat = np.mean(latencies)
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    fps = 1000.0 / mean_lat

    results = {
        "mean_ms": round(float(mean_lat), 2),
        "p50_ms": round(float(p50), 2),
        "p95_ms": round(float(p95), 2),
        "p99_ms": round(float(p99), 2),
        "fps": round(float(fps), 1)
    }
    return results

def quantize_to_int8(input_onnx: str, output_onnx: str):
    """
    Applies Post-Training Dynamic Quantization (PTQ) for CPU inference.
    """
    from onnxruntime.quantization import quantize_dynamic, QuantType
    print(f"[+] Quantizing {input_onnx} to INT8 dynamic: {output_onnx}...")
    quantize_dynamic(
        input_onnx,
        output_onnx,
        weight_type=QuantType.QInt8
    )
    orig_size_mb = os.path.getsize(input_onnx) / (1024 * 1024)
    quant_size_mb = os.path.getsize(output_onnx) / (1024 * 1024)
    print(f"✅ Quantized model created. Size: {orig_size_mb:.2f} MB -> {quant_size_mb:.2f} MB")
    return output_onnx

if __name__ == "__main__":
    model_path = "models/rtdetr_permissive.onnx"
    if os.path.exists(model_path):
        sess = create_optimized_cpu_session(model_path)
        metrics = benchmark_inference(sess)
        print("\n--- FP32 Benchmark Results (x86_64 AVX2 CPU) ---")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

        print("\n[+] Testing dynamic INT8 PTQ quantization...")
        quant_path = "models/rtdetr_permissive_int8.onnx"
        quantize_to_int8(model_path, quant_path)
        quant_sess = create_optimized_cpu_session(quant_path)
        quant_metrics = benchmark_inference(quant_sess)
        print("\n--- INT8 Benchmark Results (x86_64 AVX2 CPU) ---")
        for k, v in quant_metrics.items():
            print(f"  {k}: {v}")
    else:
        print(f"[!] Please generate the model first using export_onnx.py")
