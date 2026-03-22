import numpy as np
from scipy.signal import find_peaks
import torch
from typing import Tuple, Optional

class TonicRefiner:
    def __init__(self, bin_resolution: float = 10.0, refinement_range: float = 50.0):
        """
        bin_resolution: resolution of histogram in cents (default 10 cents)
        refinement_range: range in cents to look for refinement around a peak
        """
        self.bin_resolution = bin_resolution
        self.refinement_range = refinement_range

    def get_cents(self, f_observed: np.ndarray, f_tonic: float) -> np.ndarray:
        """
        Convert Hz to Cents relative to tonic.
        Cents = 1200 * log2(f_observed / f_tonic)
        """
        mask = (f_observed > 0)
        cents = np.zeros_like(f_observed)
        cents[mask] = 1200 * np.log2(f_observed[mask] / f_tonic)
        return cents

    def identify_tonic(self, f0: np.ndarray, voiced_mask: np.ndarray) -> float:
        """
        Histogram-based Tonic Refinement (Koduri et al. 2014).
        Identifies the Adhara Shadjam (S) from a distribution of detected f0s.
        """
        voiced_f0 = f0[voiced_mask & (f0 > 0)]
        if len(voiced_f0) == 0:
            return 0.0

        # Convert to log domain for histogramming (cents relative to A4=440Hz as reference)
        ref_freq = 440.0
        cents_ref = 1200 * np.log2(voiced_f0 / ref_freq)
        
        # Create histogram with resolution (e.g., 10 cents)
        # We wrap around to 1200 cents (octave equivalent)
        cents_wrapped = cents_ref % 1200
        bins = np.arange(0, 1200 + self.bin_resolution, self.bin_resolution)
        hist, bin_edges = np.histogram(cents_wrapped, bins=bins, density=True)

        # Finding peak
        peaks, properties = find_peaks(hist, height=0.001, distance=100/self.bin_resolution)
        if len(peaks) == 0:
            # Fallback to mean/median if no clear peak
            best_cents = np.median(cents_wrapped)
        else:
            # Pick the highest peak
            best_peak_idx = peaks[np.argmax(properties['peak_heights'])]
            best_cents = (bin_edges[best_peak_idx] + bin_edges[best_peak_idx+1]) / 2.0

        # Find the absolute frequency corresponding to this peak
        # Since it's wrapped, we need to find the octave. 
        # Typically the tonic is the most prominent low-frequency peak or its octaves.
        # For simplicity, we choose the median octave of the voiced segments.
        median_cents = np.median(cents_ref)
        octave_shift = np.round((median_cents - best_cents) / 1200) * 1200
        final_tonic_cents = best_cents + octave_shift
        tonic_freq = ref_freq * (2 ** (final_tonic_cents / 1200))
        
        return float(tonic_freq)

@torch.jit.script
def convert_to_cents_tensor(f0: torch.Tensor, tonic: float) -> torch.Tensor:
    """
    Scriptable tensor operation for cents conversion.
    """
    mask = f0 > 0
    cents = torch.zeros_like(f0)
    # Using log2 for cents
    cents = torch.where(mask, 1200.0 * torch.log2(f0 / tonic), torch.zeros_like(f0))
    return cents
