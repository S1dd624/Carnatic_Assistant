import os
import librosa
import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple, Optional

class AudioProcessor:
    def __init__(self, sr: int = 22050, fmin: float = 80.0, fmax: float = 600.0, frame_length: int = 2048):
        self.sr = sr
        self.fmin = librosa.note_to_hz('E2') if fmin is None else fmin
        self.fmax = librosa.note_to_hz('C5') if fmax is None else fmax
        self.frame_length = frame_length
        self.hop_length = frame_length // 4

    def extract_f0_single(self, audio_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract f0 using pYIN (Salamon & Gomez, 2012).
        Returns: (f0, voiced_flag, voiced_probs)
        """
        try:
            y, _ = librosa.load(audio_path, sr=self.sr)
            # pYIN extraction
            f0, voiced_flag, voiced_probs = librosa.pyin(
                y, 
                fmin=self.fmin, 
                fmax=self.fmax, 
                sr=self.sr, 
                frame_length=self.frame_length,
                hop_length=self.hop_length
            )
            return f0, voiced_flag, voiced_probs
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return np.array([]), np.array([]), np.array([])

    def process_batch(self, audio_paths: List[str], max_workers: int = 12) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Multiprocessed wrapper for f0 extraction.
        """
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(self.extract_f0_single, audio_paths))
        return results

@torch.jit.script
def normalize_f0_tensor(f0: torch.Tensor, voiced_mask: torch.Tensor) -> torch.Tensor:
    """
    Example of a scriptable function for tensor operations (CPU optimized).
    Converts f0 to log-frequency space where voiced.
    """
    log_f0 = torch.zeros_like(f0)
    # Avoid log(0)
    safe_f0 = torch.where(voiced_mask, f0, torch.ones_like(f0))
    log_f0 = torch.log2(safe_f0)
    # Mask unvoiced
    log_f0 = torch.where(voiced_mask, log_f0, torch.zeros_like(log_f0))
    return log_f0
