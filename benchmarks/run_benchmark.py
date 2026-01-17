import time
import numpy as np
import cv2
from awlf_fast import FastDenoiser

def benchmark():
    print("="*40)
    print("AWLF-Fast Performance Benchmark")
    print("="*40)
    
    # Create synthetic 16-bit thermal noise
    h, w = 512, 640
    img = np.random.randint(2000, 8000, (h, w), dtype=np.uint16)
    
    # Initialize
    print("Initializing engine (JIT compiling)...")
    denoiser = FastDenoiser()
    denoiser.warmup() # Trigger compilation
    
    # Run loop
    num_frames = 100
    print(f"Processing {num_frames} frames ({w}x{h}, 16-bit)...")
    
    start = time.time()
    for _ in range(num_frames):
        _ = denoiser.process(img)
    end = time.time()
    
    total_time = end - start
    fps = num_frames / total_time
    latency = (total_time / num_frames) * 1000
    
    print("-" * 40)
    print(f"Total Time: {total_time:.4f} s")
    print(f"Average FPS: {fps:.2f}")
    print(f"Latency:     {latency:.2f} ms")
    print("-" * 40)
    
    if fps > 30:
        print("✅ Status: REAL-TIME READY")
    else:
        print("⚠️ Status: Optimization Needed")

if __name__ == "__main__":
    benchmark()
