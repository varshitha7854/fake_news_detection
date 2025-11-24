import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
# Corrected and necessary import for the evaluation metric
from sklearn.metrics import accuracy_score 

# --- 1. Load Dataset ---
# NOTE: By default this looks for `news.csv` next to this file. You can
# override by providing a full path in the CSV_PATH variable below.
CSV_PATH = os.path.join(os.path.dirname(__file__), "news.csv")
try:
    data = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    print(f"ERROR: news.csv not found at: {CSV_PATH}")
    print("Please ensure 'news.csv' is present or update the CSV_PATH variable.")
    raise

# Basic validation
if 'text' not in data.columns or 'label' not in data.columns:
    raise ValueError("news.csv must contain 'text' and 'label' columns.")

# --- 2. Text Preprocessing and Feature Extraction ---
# TfidfVectorizer converts text into a matrix of TF-IDF features (numerical data).
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)

# Convert the 'text' column values to string type before fitting/transforming
X = tfidf.fit_transform(data['text'].values.astype('U'))

# Encode labels to integers (safer for sklearn estimators)
le = LabelEncoder()
y = le.fit_transform(data['label'].values.astype('U'))

# --- 3. Split Data ---
# Split data into 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 4. Train Model ---
print("\nTraining RandomForestClassifier...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print("Training complete.")

# --- 5. Predict and Evaluate ---
y_pred = rf.predict(X_test)

# Calculate and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print("-" * 30)
print(f"Model Accuracy on Test Set: {accuracy * 100:.2f}%")
print("-" * 30)

# Optional: show label mapping used by LabelEncoder
label_mapping = {int(idx): label for idx, label in enumerate(le.classes_)}
print("Label mapping (encoded -> original):", label_mapping)