import streamlit as st
import librosa
import numpy as np
import joblib
import os
import pandas as pd
from pydub import AudioSegment, silence
import speech_recognition as sr_module
from sklearn.preprocessing import LabelEncoder

model = joblib.load("ml_training/emotion_classifier.pkl")
label_encoder = joblib.load("ml_training/label_encoder.pkl")

st.set_page_config(page_title="AI Public Speaking Evaluator", layout="centered")
st.title("🎤 AI Public Speaking Evaluator")

uploaded_file = st.file_uploader("Upload your .wav or .mp3 file", type=["wav", "mp3"])

def extract_features(file_path):
    y, sample_rate = librosa.load(file_path, sr=16000)

    mfcc = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)

    rms = np.mean(librosa.feature.rms(y=y)[0])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y)[0])
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sample_rate))
    onset_env = librosa.onset.onset_strength(y=y, sr=sample_rate)

    try:
        from librosa.feature.rhythm import tempo
        tempo_val = tempo(onset_envelope=onset_env, sr=sample_rate)[0]
    except:
        tempo_val = librosa.beat.tempo(onset_envelope=onset_env, sr=sample_rate)[0]

    duration = librosa.get_duration(y=y, sr=sample_rate)

    audio = AudioSegment.from_file(file_path)
    silent_chunks = silence.detect_silence(audio, min_silence_len=500, silence_thresh=-40)
    pause_ratio = len(silent_chunks) / (duration / 60)

    recognizer = sr_module.Recognizer()
    with sr_module.AudioFile(file_path) as source:
        try:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            word_count = len(text.split())
            wpm = (word_count / duration) * 60
        except:
            word_count = 0
            wpm = 0

    return {
        'mfcc1': mfcc_mean[0],
        'mfcc2': mfcc_mean[1],
        'mfcc3': mfcc_mean[2],
        'rms': rms,
        'zcr': zcr,
        'chroma': chroma,
        'tempo': tempo_val,
        'duration': duration,
        'pause_ratio': pause_ratio,
        'wpm': wpm
    }

if uploaded_file is not None:
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())

    st.audio("temp.wav")

    st.subheader("🔍 Analyzing your speech...")
    features = extract_features("temp.wav")
    input_df = pd.DataFrame([features])
    predicted_emotion = model.predict(input_df)[0]
    decoded_emotion = label_encoder.inverse_transform([predicted_emotion])[0]

    st.success(f"🎭 Detected Emotion: **{decoded_emotion.capitalize()}**")

    st.subheader("📊 Detailed Feature Analysis")
    st.markdown(f"- **Speech Speed (WPM)**: `{round(features['wpm'])}`")
    st.markdown(f"- **Pauses per Minute**: `{round(features['pause_ratio'], 2)}`")
    st.markdown(f"- **Volume Stability (RMS)**: `{round(features['rms'], 5)}`")
    st.markdown(f"- **Pitch Estimate (MFCC1)**: `{round(features['mfcc1'], 2)}`")
    st.markdown(f"- **Estimated Tempo (BPM)**: `{round(features['tempo'], 2)}`")

    st.subheader("📈 Categorized Feedback for Improvement")

    # 1. Speech Speed
    if features['wpm'] < 80:
        st.warning("📉 You're speaking too slowly. Try increasing your pace for clarity and engagement.")
    elif features['wpm'] > 180:
        st.warning("🚀 You're speaking too fast. Try slowing down to help the audience follow along.")
    else:
        st.success("✅ Your speaking pace is well-balanced.")

    # 2. Pausing
    if features['pause_ratio'] < 2:
        st.info("⏸️ Consider adding more natural pauses to improve pacing.")
    elif features['pause_ratio'] > 10:
        st.warning("🔁 You're pausing too frequently. Practice smoother transitions.")
    else:
        st.success("✅ Good control over pauses.")

    # 3. Volume / RMS
    if features['rms'] < 0.01:
        st.warning("🔉 Your speech volume is quite low. Try to speak louder or closer to the mic.")
    elif features['rms'] > 0.05:
        st.info("📢 Your volume is a bit high. Watch for distortion.")
    else:
        st.success("✅ Clear and consistent volume.")

    # 4. Pitch (MFCC1)
    if features['mfcc1'] < -500:
        st.info("🎵 Try using more vocal variety. The pitch sounds flat or monotone.")
    elif features['mfcc1'] > 0:
        st.warning("🎼 Too much pitch variation may sound inconsistent.")
    else:
        st.success("✅ Your pitch variation is healthy.")

    # 5. Tempo
    if features['tempo'] < 60:
        st.info("⏰ Your delivery is slow-paced. Try adding a bit more rhythm.")
    elif features['tempo'] > 180:
        st.warning("🎬 Tempo is high. Reduce rush to maintain clarity.")
    else:
        st.success("✅ Good speaking tempo.")

    # Visualizations
    st.subheader("📊 Voice Visualizations")
    from matplotlib import pyplot as plt
    import librosa.display

    y, sr = librosa.load("temp.wav", sr=16000)

    # Waveform
    st.markdown("**Waveform**")
    fig, ax = plt.subplots()
    librosa.display.waveshow(y, sr=sr, ax=ax)
    st.pyplot(fig)

    # MFCC
    st.markdown("**MFCC Heatmap**")
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(mfcc, x_axis='time', ax=ax)
    fig.colorbar(img, ax=ax)
    ax.set_title('MFCC')
    st.pyplot(fig)

    # Tempo Plot
    st.markdown("**Onset Strength (Tempo Curve)**")
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    fig, ax = plt.subplots()
    ax.plot(onset_env)
    ax.set_title("Onset Strength")
    st.pyplot(fig)
