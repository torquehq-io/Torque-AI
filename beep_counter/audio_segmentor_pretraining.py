import librosa
import numpy as np
from scipy.io.wavfile import write

def segment_audio(audio_path, timestamps, output_dir):
    y, sr = librosa.load(audio_path, sr=22050)

    for i, (start, end, label) in enumerate(timestamps):
        start_sample = int(start * sr)
        end_sample = int(end * sr)

        segment = y[start_sample:end_sample]
        output_path = f"{output_dir}/segment_{i}_{label}.wav"
        write(output_path, sr, segment.astype(np.float32))

def convert_to_spectrogram(audio_path, n_fft=2048):
    y, sr = librosa.load(audio_path, sr=22050)
    
    # If the audio is shorter than n_fft, pad it with zeros
    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)), constant_values=(0, 0))
    
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    spectrogram = librosa.power_to_db(spectrogram)
    return spectrogram



audio_path = '/home/torque/github/main/Torque-AI/beep_counter/out_audio.aac'
output_dir = '/home/torque/github/main/Torque-AI/beep_counter/data/segments'
timestamps = [
    (0, 3, 'beep'),
    (5, 39, 'non-beep'),
    (40, 43, 'beep'),
    (48, 102, 'non-beep'),
    (103, 106, 'beep'),
    (110, 125, 'non-beep'),
    (128, 134, 'beep'),
    (138, 158, 'non-beep')
]



segment_audio(audio_path, timestamps, output_dir)

for i, (start, end, label) in enumerate(timestamps):
    segment_path = f"{output_dir}/segment_{i}_{label}.wav"
    spectrogram = convert_to_spectrogram(segment_path)
    # Now `spectrogram` is a 2D array representing the spectrogram of the audio segment.
