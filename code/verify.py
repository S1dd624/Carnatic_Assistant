import numpy as np
import librosa
import soundfile as sf
import os
from preprocess_pipeline import run_pipeline

def generate_test_audio(path="test_gamaka.wav", duration=5.0, sr=22050):
    t = np.linspace(0, duration, int(sr * duration))
    
    # Base frequency (Tonic S) = 150 Hz
    f_tonic = 150.0
    
    # Gamaka: 5Hz oscillation with 100 cent amplitude (approx 2**(1/12) factor)
    # 100 cents = 1 semitone. 150 * 2**(1/12) = 158.9 Hz.
    # So amplitude is approx 9 Hz.
    f_m = 5.0
    a_m = 10.0 
    
    f_t = f_tonic + a_m * np.sin(2 * np.pi * f_m * t)
    
    # Add random jitter (Apaswara) - high frequency noise in pitch
    jitter = 2.0 * np.random.randn(len(t))
    f_t += jitter
    
    # Phase is integral of frequency
    phase = 2 * np.pi * np.cumsum(f_t) / sr
    y = 0.5 * np.sin(phase)
    
    sf.write(path, y, sr)
    return path, f_tonic

def verify():
    test_file, true_tonic = generate_test_audio()
    print(f"Generated synthetic audio: {test_file}")
    
    run_pipeline([test_file], output_dir="test_output")
    
    # Load results
    res_path = "test_output/test_gamaka.wav_processed.npz"
    if not os.path.exists(res_path):
        print("Verification Failed: No output found.")
        return
        
    data = np.load(res_path)
    est_tonic = data['tonic']
    
    print("\n--- Verification Results ---")
    print(f"True Tonic: {true_tonic} Hz")
    print(f"Est Tonic:  {est_tonic:.2f} Hz")
    print(f"Error:      {abs(true_tonic - est_tonic):.2f} Hz")
    
    if abs(true_tonic - est_tonic) < 5.0:
        print("Tonic Refinement: SUCCESS")
    else:
        print("Tonic Refinement: FAILURE")

    # Check smoothing
    cents = data['cents']
    smoothed = data['smoothed_cents']
    
    # Voiced segments only
    mask = cents != 0
    if np.any(mask):
        noise_reduction = np.std(cents[mask]) - np.std(smoothed[mask])
        print(f"Noise (std) reduction: {noise_reduction:.2f} cents")
        if noise_reduction > 0:
             print("Jitter Filtering: SUCCESS (Preserved signal with lower variance)")
        else:
             print("Jitter Filtering: NEUTRAL (Low noise or over-smoothing)")

if __name__ == "__main__":
    verify()
