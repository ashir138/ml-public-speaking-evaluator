# 🎤 ML-Based Public Speaking Evaluator

An intelligent web application that evaluates public speaking performance using Machine Learning and audio signal processing. Upload your voice, and get real-time feedback on your **emotion**, **speech speed**, **volume stability**, **pitch variation**, and **pausing behavior** — all visualized beautifully with actionable suggestions.

---

## 🚀 Features

- 🎭 Emotion prediction (Sad, Happy, Neutral, etc.)
- 🗣️ Speech speed analysis (Words Per Minute)
- 🔉 Volume consistency (RMS energy)
- 🎵 Pitch variation using MFCC
- ⏸️ Pause detection & pacing score
- 📊 Visual graphs: waveform, MFCC heatmap, tempo plot
- 🧠 Categorized, human-like feedback to improve your delivery

---

## 🧠 Technologies Used

| Type           | Tools / Libraries                            |
|----------------|-----------------------------------------------|
| **Language**   | Python                                        |
| **Web UI**     | Streamlit                                     |
| **ML Model**   | Scikit-learn (`RandomForestClassifier`)       |
| **Audio**      | Librosa, PyDub, SpeechRecognition             |
| **Data Viz**   | Matplotlib, Librosa.display                   |
| **Deployment** | Localhost (Streamlit)                         |

---


---

## 🧪 How It Works

1. User uploads `.wav`  file
2. Audio is preprocessed using Librosa & PyDub
3. Features are extracted (MFCCs, RMS, tempo, pause ratio, etc.)
4. ML model predicts the **emotion**
5. App provides:
   - Visual feedback (waveform, MFCC, tempo)
   - Categorized suggestions to improve delivery

---

## 📸 Visualization

[LinkedIn](https://www.linkedin.com/posts/ali-ashir-3bb590257_aliashir-machinelearning-publicspeaking-activity-7335010123599216641-ALKs?utm_source=share&utm_medium=member_desktop&rcm=ACoAAD9Ehz8B3F06Xk6O3RVLyYiiMwgY_wGWhSo)

---

## 📬 Feedback / Contributions
Feel free to open issues or PRs to improve the project!
Want to deploy it live or add mic input? Let’s collaborate.

---

## 📦 Installation

```bash
git clone https://github.com/your-username/ml-public-speaking-evaluator.git
cd ml-public-speaking-evaluator
pip install -r requirements.txt
streamlit run app.py


