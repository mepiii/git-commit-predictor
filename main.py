import os, argparse
from trainer.train import train_commit_classifier
from predictor.predict import CommitPredictor
def main():
    b = os.path.dirname(os.path.abspath(__file__))
    d, m = os.path.join(b, 'dataset', 'git_commits.csv'), os.path.join(b, 'models', 'commit_classifier.joblib')
    if not os.path.exists(m): train_commit_classifier(d, m)
    p = CommitPredictor(m)
    print('Git Commit Predictor Demo:')
    for text in ['Added login validation endpoint', 'Fixed null pointer exception']:
        res = p.predict_commit(text)
        print(' ', res['suggested_title'])
if __name__ == '__main__': main()
