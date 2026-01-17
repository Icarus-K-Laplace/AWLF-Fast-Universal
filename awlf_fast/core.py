
```python
import numpy as np
from numba import njit, prange, float32, uint16, uint8

@njit(fastmath=True, cache=True)
def get_adaptive_weight(center_val, neighbor_val, dist_sq, sensitivity):
    """
    Compute weight based on intensity difference and spatial distance.
    Inline function for Numba optimization.
    """
    diff = abs(float(center_val) - float(neighbor_val))
    # Avoid division by zero and expensive exp functions if possible
    # Using a rational approximation for speed: 1 / (1 + x)
    return 1.0 / (1.0 + (diff * diff * dist_sq) / (sensitivity * sensitivity) + 1e-6)

@njit(parallel=True, fastmath=True)
def fast_process_16bit(img, output, h, w, win_size, sensitivity):
    """
    JIT-compiled core loop for 16-bit images.
    Uses OpenMP (prange) for multi-core execution.
    """
    pad = win_size // 2
    
    # Iterate over all pixels in parallel
    for r in prange(pad, h - pad):
        for c in range(pad, w - pad):
            center_val = img[r, c]
            
            # 1. Fast Noise Detection (Min-Max)
            # Unroll loop manually for small windows (3x3 or 5x5) is faster, 
            # but generic loop is safer for variable sizes.
            is_noise = True
            local_min = 65535
            local_max = 0
            
            # First pass: Statistics
            for i in range(-pad, pad + 1):
                for j in range(-pad, pad + 1):
                    val = img[r + i, c + j]
                    if val < local_min: local_min = val
                    if val > local_max: local_max = val
            
            # If center is not an extreme, skip (keep original)
            if local_min < center_val < local_max:
                output[r, c] = center_val
                continue
                
            # 2. Adaptive Restoration
            sum_weights = 0.0
            sum_values = 0.0
            
            for i in range(-pad, pad + 1):
                for j in range(-pad, pad + 1):
                    # Skip center pixel in restoration if it is noise
                    if i == 0 and j == 0:
                        continue
                        
                    val = img[r + i, c + j]
                    dist_sq = i*i + j*j
                    
                    w_val = get_adaptive_weight(center_val, val, dist_sq, sensitivity)
                    
                    sum_weights += w_val
                    sum_values += w_val * val
            
            # 3. Write result
            if sum_weights > 1e-6:
                output[r, c] = uint16(sum_values / sum_weights)
            else:
                # Fallback to median-like heuristic (average of min and max)
                output[r, c] = uint16((local_min + local_max) / 2)

class FastDenoiser:
    """
    High-performance wrapper for the JIT kernel.
    """
    def __init__(self, window_size=5, sensitivity=10.0):
        self.window_size = window_size
        self.sensitivity = float(sensitivity)
        self._warmed_up = False

    def process(self, img):
        """
        Process an image (8-bit or 16-bit).
        """
        # Ensure contiguous memory array for Numba
        img = np.ascontiguousarray(img)
        h, w = img.shape
        output = np.zeros_like(img)
        
        if img.dtype == np.uint16:
            fast_process_16bit(img, output, h, w, self.window_size, self.sensitivity)
        elif img.dtype == np.uint8:
            # Cast to 16-bit for processing to maintain precision, then cast back
            # (Or implement a separate fast_process_8bit kernel for max speed)
            img_16 = img.astype(np.uint16)
            out_16 = np.zeros_like(img_16)
            fast_process_16bit(img_16, out_16, h, w, self.window_size, self.sensitivity)
            output = np.clip(out_16, 0, 255).astype(np.uint8)
        else:
            raise ValueError(f"Unsupported dtype: {img.dtype}. Use uint8 or uint16.")
            
        return output
        
    def warmup(self):
        """Run a dummy pass to trigger JIT compilation."""
        dummy = np.zeros((100, 100), dtype=np.uint16)
        self.process(dummy)
        self._warmed_up = True
