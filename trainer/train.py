import os, pandas as pd, joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
def train_commit_classifier(data_path, model_path):
    df = pd.read_csv(data_path)
    pipe = Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1,2), token_pattern=r'\w+')), ('clf', LogisticRegression(C=10.0, max_iter=1000))])
    pipe.fit(df['diff_summary'], df['commit_type'])
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipe, model_path)
