
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, hamming_loss, f1_score
from gensim.models import Word2Vec, KeyedVectors

# ---------- Load Dataset ----------
df = pd.read_csv("data_amazon.csv")

# Detect review column
text_col = [c for c in df.columns if "review" in c.lower() or "text" in c.lower()][0]

# Choose low-cardinality label columns
candidate_labels = [c for c in df.columns if c != text_col and df[c].nunique() <= 20]
chosen_labels = candidate_labels[:5]  # adjust if needed

df['labels'] = df.apply(lambda r: [f"{c}:{r[c]}" for c in chosen_labels if not pd.isna(r[c])], axis=1)
df = df[df['labels'].map(len) > 0].reset_index(drop=True)

# ---------- Text Cleaning ----------
STOPWORDS = set("""
i me my we our us you your he she they them it this that the a an and or but if
""".split())

def preprocess(text):
    text = re.sub("[^a-zA-Z ]", " ", text.lower())
    tokens = [w for w in text.split() if w not in STOPWORDS and len(w) > 2]
    return tokens

df["tokens"] = df[text_col].astype(str).apply(preprocess)

sentences = df["tokens"].tolist()

# ========= Train Word2Vec =========
w2v_size = 100
w2v_model = Word2Vec(sentences, vector_size=w2v_size, window=5, min_count=2, workers=4)

# Document embedding using W2V
def doc_w2v(tokens):
    vecs = [w2v_model.wv[w] for w in tokens if w in w2v_model.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(w2v_size)

# ========= Load GloVe =========
glove_path = "glove.6B.100d.txt"  # change if different

glove = {}
with open(glove_path, "r", encoding="utf-8") as f:
    for line in f:
        vals = line.split()
        word = vals[0]
        vec = np.asarray(vals[1:], dtype="float32")
        glove[word] = vec

glove_dim = len(vec)

def doc_glove(tokens):
    vecs = [glove[w] for w in tokens if w in glove]
    return np.mean(vecs, axis=0) if vecs else np.zeros(glove_dim)

# ---------- Build Feature Matrices ----------
X_w2v = np.vstack(df["tokens"].apply(doc_w2v))
X_glove = np.vstack(df["tokens"].apply(doc_glove))

mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(df["labels"])

# Train-Test
Xw_train, Xw_test, Y_train, Y_test = train_test_split(X_w2v, Y, test_size=0.2, random_state=42)
Xg_train, Xg_test, _, _ = train_test_split(X_glove, Y, test_size=0.2, random_state=42)

# ---------- Classification ----------
clf = OneVsRestClassifier(LogisticRegression(max_iter=2000))

# Train both
clf_w2v = clf.fit(Xw_train, Y_train)
clf_glove = clf.fit(Xg_train, Y_train)

# Predictions
Y_pred_w2v = clf_w2v.predict(Xw_test)
Y_pred_glove = clf_glove.predict(Xg_test)

# ---------- Evaluation ----------
def metrics(name, y_true, y_pred):
    print("\n", "="*10, name, "="*10)
    print("Subset accuracy :", accuracy_score(y_true, y_pred))
    print("Hamming loss    :", hamming_loss(y_true, y_pred))
    print("F1 Micro        :", f1_score(y_true, y_pred, average='micro'))
    print("F1 Macro        :", f1_score(y_true, y_pred, average='macro'))

metrics("WORD2VEC", Y_test, Y_pred_w2v)
metrics("GLOVE", Y_test, Y_pred_glove)

# ============================================================
# ROC CURVE + AUC COMPARISON FOR WORD2VEC vs GLOVE
# ============================================================

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Prediction probabilities required for ROC
Y_score_w2v  = clf_w2v.predict_proba(Xw_test)
Y_score_glv  = clf_glove.predict_proba(Xg_test)

label_names = mlb.classes_
n_classes = len(label_names)

# ================= Label‑wise ROC Curves ===================
for i in range(n_classes):
    fpr_w2v, tpr_w2v, _ = roc_curve(Y_test[:, i], Y_score_w2v[:, i])
    fpr_glv, tpr_glv, _ = roc_curve(Y_test[:, i], Y_score_glv[:, i])

    auc_w2v = auc(fpr_w2v, tpr_w2v)
    auc_glv = auc(fpr_glv, tpr_glv)

    # ---- Plot for each label ----
    plt.figure(figsize=(7,5))
    plt.plot(fpr_w2v, tpr_w2v, label=f"Word2Vec AUC = {auc_w2v:.3f}", linewidth=2)
    plt.plot(fpr_glv, tpr_glv, label=f"GloVe AUC = {auc_glv:.3f}", linewidth=2)

    plt.plot([0,1],[0,1],"k--", alpha=0.6)
    plt.title(f"ROC Curve — Label: {label_names[i]}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.show()


# ================= Micro‑Averaged ROC =====================
fpr_w2v, tpr_w2v, _ = roc_curve(Y_test.ravel(), Y_score_w2v.ravel())
fpr_glv, tpr_glv, _ = roc_curve(Y_test.ravel(), Y_score_glv.ravel())

auc_micro_w2v = auc(fpr_w2v, tpr_w2v)
auc_micro_glv = auc(fpr_glv, tpr_glv)

plt.figure(figsize=(8,6))
plt.plot(fpr_w2v, tpr_w2v, label=f"Word2Vec Micro AUC = {auc_micro_w2v:.3f}", linewidth=2)
plt.plot(fpr_glv, tpr_glv, label=f"GloVe Micro AUC = {auc_micro_glv:.3f}", linewidth=2)
plt.plot([0,1],[0,1],'k--', alpha=0.6)

plt.title("Micro‑Averaged ROC Curve — Word2Vec vs GloVe")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.show()
