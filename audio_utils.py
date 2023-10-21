
import os
import numpy as np
import librosa
from pydub import AudioSegment
from pydub.utils import make_chunks


def load_audio(file_path):
    audio = AudioSegment.from_wav(file_path)
    return audio


def pad_audio(samples, max_length):
    if len(samples) < max_length:
        padded_samples = np.pad(samples, (0, max_length - len(samples)), mode='constant')
    else:
        padded_samples = samples[:max_length]
    return padded_samples


def extract_mfcc(samples, sample_rate, n_mfcc):
    mfcc = librosa.feature.mfcc(y=samples, sr=sample_rate, n_mfcc=n_mfcc)
    return mfcc

def preprocess_audio(file_path, n_mfcc, max_length):
    audio = load_audio(file_path)
    padded_samples = pad_audio(np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0, max_length)
    mfcc = extract_mfcc(padded_samples, audio.frame_rate, n_mfcc)
    processed_data = np.mean(mfcc, axis=1).reshape(1, -1, n_mfcc)  # Mengubah dimensi menjadi (1, n_mfcc)
    return processed_data