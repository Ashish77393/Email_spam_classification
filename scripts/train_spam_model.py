"""Small training script that builds a TF-IDF vectorizer + classifier and saves them.

Run this from the project root (same directory as manage.py).

python scripts/train_spam_model.py

The script will write files under: <project-root>/ml_models/spam_vectorizer.joblib and spam_clf.joblib
"""
from pathlib import Path
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def get_sample_dataset():
    # Small in-repo example dataset. Replace with larger dataset as needed.
    spam = [
        "Congratulations, you won a free prize! Claim now!",
        "You are a winner — click here to get your reward",
        "Lowest mortgage rates, apply now and get approved",
        "Buy cheap meds online, no prescription needed",
        "Limited time offer! Get 50% off today",
        "Earn money fast with this easy program",
    ]

    ham = [
        "Hi John, are we still on for tomorrow's meeting?",
        "Please find attached the minutes from today's meeting",
        "Dinner at my place on Friday — let's catch up",
        "Your Amazon order has shipped and is on its way",
        "Can you review the report and send feedback?",
        "We've scheduled the call for 3pm — joining link included",
    ]

    texts = spam + ham
    labels = ['spam'] * len(spam) + ['ham'] * len(ham)
    return texts, labels


def main():
    root = Path(__file__).resolve().parents[1]
    out_dir = root / 'ml_models'
    out_dir.mkdir(exist_ok=True)

    texts, labels = get_sample_dataset()

    print('Training TF-IDF vectorizer...')
    vect = TfidfVectorizer(stop_words='english', max_features=2000)
    X = vect.fit_transform(texts)

    print('Training classifier (LogisticRegression)...')
    clf = LogisticRegression(solver='liblinear')
    clf.fit(X, labels)

    vect_path = out_dir / 'spam_vectorizer.joblib'
    clf_path = out_dir / 'spam_clf.joblib'

    print('Saving vectorizer to', vect_path)
    joblib.dump(vect, vect_path)

    print('Saving classifier to', clf_path)
    joblib.dump(clf, clf_path)

    print('Done — created model artifacts under', out_dir)


if __name__ == '__main__':
    main()
