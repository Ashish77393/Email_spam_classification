"""Quick test helper to verify saved model artifacts and try a sample prediction."""
from pathlib import Path
import joblib


def main():
    root = Path(__file__).resolve().parents[1]
    vect_path = root / 'ml_models' / 'spam_vectorizer.joblib'
    clf_path = root / 'ml_models' / 'spam_clf.joblib'

    if not vect_path.exists() or not clf_path.exists():
        print('Model files are missing. Run scripts/train_spam_model.py first.')
        return

    vect = joblib.load(vect_path)
    clf = joblib.load(clf_path)

    samples = [
        'Free gift card — click to claim your prize now',
        "Hi team, quick reminder about the budget review meeting tomorrow",
    ]

    X = vect.transform(samples)
    preds = clf.predict(X)
    print('Samples:')
    for s, p in zip(samples, preds):
        print('-', p, ':', s)


if __name__ == '__main__':
    main()
