import joblib
class CommitPredictor:
    def __init__(self, m): self.p = joblib.load(m)
    def predict_commit(self, t):
        pr = self.p.predict_proba([t])[0]
        ct = self.p.classes_[pr.argmax()]
        return {'suggested_title': f'{ct}(core): {t[:35]}', 'confidence': float(pr.max())}
