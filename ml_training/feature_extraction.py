import librosa
import numpy as np
import pandas as pd
import os
from pydub import AudioSegment, silence
import speech_recognition as sr

def extract_features(file_path):
    # Load audio
    y, sample_rate = librosa.load(file_path, sr=16000)

    # 1. MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)

    # 2. RMS Energy
    rms = librosa.feature.rms(y=y)[0]
    rms_mean = np.mean(rms)

    # 3. Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    zcr_mean = np.mean(zcr)

    # 4. Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sample_rate)
    chroma_mean = np.mean(chroma)

    # 5. Tempo (use old API to avoid error)
    onset_env = librosa.onset.onset_strength(y=y, sr=sample_rate)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sample_rate)[0]

    # 6. Duration
    duration = librosa.get_duration(y=y, sr=sample_rate)

    # 7. Silence detection
    audio = AudioSegment.from_file(file_path)
    silent_chunks = silence.detect_silence(audio, min_silence_len=500, silence_thresh=-40)
    pause_ratio = len(silent_chunks) / (duration / 60 + 1e-6)  # avoid div-by-zero

    # 8. Speech to Text → WPM
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(file_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            word_count = len(text.split())
            wpm = (word_count / duration) * 60 if duration > 0 else 0
    except:
        word_count = 0
        wpm = 0

    # Combine features
    return {
        'mfcc1': mfcc_mean[0],
        'mfcc2': mfcc_mean[1],
        'mfcc3': mfcc_mean[2],
        'rms': rms_mean,
        'zcr': zcr_mean,
        'chroma': chroma_mean,
        'tempo': tempo,
        'duration': duration,
        'pause_ratio': pause_ratio,
        'wpm': wpm
    }


if __name__ == "__main__":
    directory = 'sample_audio/'
    data = []

    for filename in os.listdir(directory):
        if filename.lower().endswith(('.wav', '.mp3')):
            path = os.path.join(directory, filename)
            print(f"🔄 Processing {filename}")
            try:
                features = extract_features(path)
                features['file'] = filename
                data.append(features)
            except Exception as e:
                print(f"❌ Error with {filename}: {e}")

    df = pd.DataFrame(data)
    df.to_csv('extracted_features.csv', index=False)
    print("✅ Features saved to extracted_features.csv")
