import urllib.request
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score, roc_auc_score

URL = ("https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu-llama3-annotations/"
       "resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet")
urllib.request.urlretrieve(URL, "edu_scores.parquet")
df = pd.read_parquet("edu_scores.parquet", columns=["text", "score"])
perm = np.random.default_rng(0).permutation(len(df))
test, train = perm[: len(df) // 6], perm[len(df) // 6:]

hv = HashingVectorizer(n_features=2**20, alternate_sign=False, ngram_range=(1, 2), norm=None)
tf = TfidfTransformer(sublinear_tf=True)
X_train = tf.fit_transform(hv.transform(df["text"].iloc[train]))
X_test = tf.transform(hv.transform(df["text"].iloc[test]))
y_train, y_test = df["score"].values[train], df["score"].values[test]
model = Ridge(alpha=1.0).fit(X_train, y_train)
pred = model.predict(X_test)

print("rows", len(df), "| share of test pages scored 3 or more", round(float(np.mean(y_test >= 3)), 4))
print("mean squared error", round(float(np.mean((pred - y_test) ** 2)), 4),
      "| predicting the mean", round(float(np.mean((y_train.mean() - y_test) ** 2)), 4))
print("AUC for score >= 3", round(roc_auc_score(y_test >= 3, pred), 4))
for cut in (3.0, 2.0):
    print(f"keep pred >= {cut}: F1 {f1_score(y_test >= 3, pred >= cut):.4f}, pages kept {int((pred >= cut).sum())}")
