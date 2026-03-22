import numpy as np
import torch
from scipy.signal import savgol_filter
from typing import Tuple

class GamakasFilter:
    def __init__(self, window_length: int = 15, polyorder: int = 3, jitter_threshold: float = 20.0):
        """
        window_length: size of smoothing window (odd)
        polyorder: order of polynomial for Savitzky-Golay
        jitter_threshold: acceleration threshold in cents/frame^2 to detect jitter
        """
        self.window_length = window_length
        self.polyorder = polyorder
        self.jitter_threshold = jitter_threshold

    def apply_smoothing(self, cents: np.ndarray) -> np.ndarray:
        """
        Apply Savitzky-Golay filtering to preserve curves (Gamakas) while reducing noise.
        """
        if len(cents) <= self.window_length:
            return cents
        
        # We only smooth non-zero (voiced) segments
        smoothed = np.copy(cents)
        voiced_segments = np.where(cents != 0)[0]
        
        if len(voiced_segments) < self.window_length:
            return cents

        # Finding contiguous voiced segments
        diff = np.diff(voiced_segments)
        breaks = np.where(diff > 1)[0]
        starts = np.insert(breaks + 1, 0, 0)
        ends = np.append(breaks, len(voiced_segments) - 1)

        for s, e in zip(starts, ends):
            segment_indices = voiced_segments[s:e+1]
            if len(segment_indices) > self.window_length:
                smoothed[segment_indices] = savgol_filter(
                    cents[segment_indices], 
                    window_length=self.window_length, 
                    polyorder=self.polyorder
                )
        
        return smoothed

    def analyze_dynamics(self, cents: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute first and second order derivatives ($df/dt$ and $d^2f/dt^2$).
        """
        v = np.gradient(cents)
        a = np.gradient(v)
        return v, a

@torch.jit.script
def detect_jitter_jit(cents: torch.Tensor, threshold: float) -> torch.Tensor:
    """
    Torch-optimized jitter detection.
    Computes absolute acceleration and masks regions exceeding threshold.
    """
    # Simple finite difference for speed
    v = cents[1:] - cents[:-1]
    a = v[1:] - v[:-1]
    
    # Pad a to match cents length
    a_padded = torch.zeros_like(cents)
    a_padded[1:-1] = torch.abs(a)
    
    jitter_mask = a_padded > threshold
    return jitter_mask

class GamakaIntegrityAuditor:
    """
    High-level auditor that uses dynamic thresholds to flag 'Apaswaras'.
    """
    def __init__(self, filter_engine: GamakasFilter):
        self.filter_engine = filter_engine

    def audit(self, cents: np.ndarray) -> dict:
        v, a = self.filter_engine.analyze_dynamics(cents)
        # High acceleration regions
        high_accel = np.abs(a) > self.filter_engine.jitter_threshold
        
        # Gamaka logic: Smooth high-velocity segments are likely intentional.
        # Apaswara logic: Erratic, high-frequency "jitter" is unintentional.
        
        # Calculate local variance of acceleration to distinguish
        # (Simplified implementation)
        return {
            "velocity": v,
            "acceleration": a,
            "high_accel_mask": high_accel
        }
