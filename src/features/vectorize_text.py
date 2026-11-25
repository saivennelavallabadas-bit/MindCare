
from pathlib import Path
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(".")
INPUT = ROOT / "data/processed/text_documents.csv"
OUTPUT = ROOT / "data/processed"
MODEL_DIR = ROOT / "models"

OUTPUT.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

data = pd.read_csv(INPUT)
data["participant_text"] = data["participant_text"].fillna("")

train_text = data.loc[data["split"] == "train", "participant_text"]

vectorizer = TfidfVectorizer(
    max_features=300,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

vectorizer.fit(train_text)

features = vectorizer.transform(data["participant_text"])

feature_names = vectorizer.get_feature_names_out()

tfidf_data = pd.DataFrame(
    features.toarray(),
    columns=feature_names
)

tfidf_data.insert(0, "participant_id", data["participant_id"])
tfidf_data.insert(1, "split", data["split"])

tfidf_data.to_csv(OUTPUT / "text_tfidf_features.csv", index=False)

joblib.dump(vectorizer, MODEL_DIR / "tfidf_vectorizer.joblib")

print("=" * 60)
print("TEXT TF-IDF VECTORIZATION")
print("=" * 60)
print("Participants:", len(data))
print("Training participants:", len(train_text))
print("Vocabulary size:", len(feature_names))
print("Feature shape:", features.shape)
print("Output:", OUTPUT / "text_tfidf_features.csv")
print("Vectorizer:", MODEL_DIR / "tfidf_vectorizer.joblib")
print("=" * 60)
