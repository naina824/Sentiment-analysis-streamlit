import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

# =====================================================
# FIXED PATH HANDLING (VERY IMPORTANT)
# =====================================================

# Absolute path of project root (SentimentAnalysisProject)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Models directory (always in project root)
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Dataset path
DATA_PATH = os.path.join(BASE_DIR, "movie_data.csv")

print("Project root:", BASE_DIR)
print("Dataset path:", DATA_PATH)
print("Models will be saved to:", MODELS_DIR)

# Create models folder if not exists
os.makedirs(MODELS_DIR, exist_ok=True)

# =====================================================
# 1. Load Dataset
# =====================================================

data = pd.read_csv(DATA_PATH)

# Features & labels
X = data["review"]        # text column
y = data["sentiment"]     # 0 = negative, 1 = positive

print("\nDataset shape:", data.shape)
print("Label distribution:\n", y.value_counts())

# =====================================================
# 2. Train-Test Split
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================================================
# 3. TF-IDF Vectorization
# =====================================================

tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=30000,
    ngram_range=(1, 2)
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# =====================================================
# 4. Train Naive Bayes
# =====================================================

nb = MultinomialNB()
nb.fit(X_train_tfidf, y_train)
nb_pred = nb.predict(X_test_tfidf)

# =====================================================
# 5. Train SVM
# =====================================================

svm = LinearSVC()
svm.fit(X_train_tfidf, y_train)
svm_pred = svm.predict(X_test_tfidf)

# =====================================================
# 6. Evaluation
# =====================================================

nb_acc = accuracy_score(y_test, nb_pred)
svm_acc = accuracy_score(y_test, svm_pred)

print("\n================ Naive Bayes Results ================")
print("Accuracy:", nb_acc)
print(classification_report(y_test, nb_pred))

print("\n================ SVM Results ================")
print("Accuracy:", svm_acc)
print(classification_report(y_test, svm_pred))
nb_report = classification_report(y_test, nb_pred, output_dict=True)
svm_report = classification_report(y_test, svm_pred, output_dict=True)

# =====================================================
# 7. Confusion Matrices
# =====================================================

cm_nb = confusion_matrix(y_test, nb_pred)
cm_svm = confusion_matrix(y_test, svm_pred)

plt.figure(figsize=(5, 4))
sns.heatmap(
    cm_nb,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Negative", "Positive"],
    yticklabels=["Negative", "Positive"]
)
plt.title("Naive Bayes Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

plt.figure(figsize=(5, 4))
sns.heatmap(
    cm_svm,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=["Negative", "Positive"],
    yticklabels=["Negative", "Positive"]
)
plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# =====================================================
# 8. Save Models & Metrics (GUARANTEED)
# =====================================================

pickle.dump(nb, open(os.path.join(MODELS_DIR, "nb_model.pkl"), "wb"))
pickle.dump(svm, open(os.path.join(MODELS_DIR, "svm_model.pkl"), "wb"))
pickle.dump(tfidf, open(os.path.join(MODELS_DIR, "tfidf.pkl"), "wb"))

metrics = {
    "nb_accuracy": nb_acc,
    "svm_accuracy": svm_acc,
    "nb_report": nb_report,
    "svm_report": svm_report,
    "cm_nb": cm_nb,
    "cm_svm": cm_svm,
}

pickle.dump(metrics, open(os.path.join(MODELS_DIR, "metrics.pkl"), "wb"))

print("\n✅ Models and metrics saved successfully!")
