import pandas as pd

# Emotion code map from RAVDESS
emotion_map = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# Load existing features CSV
df = pd.read_csv("extracted_features.csv")

# Extract emotion code from filename and map it
def get_emotion_from_filename(filename):
    try:
        code = filename.split("-")[2]
        return emotion_map.get(code, "unknown")
    except:
        return "unknown"

df['emotion'] = df['file'].apply(get_emotion_from_filename)

# Save updated CSV
df.to_csv("extracted_features.csv", index=False)
print("✅ Emotion column added to extracted_features.csv")
