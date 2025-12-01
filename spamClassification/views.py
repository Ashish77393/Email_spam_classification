from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
import json
import os

# ML imports are optional. If scikit-learn/joblib aren't available, the view will fall back to the
# simple heuristic implemented below.
try:
    from joblib import load
    import numpy as np
except Exception:
    load = None
    np = None


def Home(request):
    return render(request, 'Home.html')


def classify(request):
    """Simple, local heuristic classification endpoint for demo/testing.

    Accepts POST JSON {"text": "..."} and returns JSON:
      {"label": "spam"|"ham", "confidence": float(0..1), "explanation": "..."}

    This is a placeholder — replace with real model inference later.
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)

    try:
        payload = json.loads(request.body.decode('utf-8') or '{}')
        text = (payload.get('text') or '').strip()
    except Exception:
        return JsonResponse({'error': 'invalid JSON'}, status=400)

    if not text:
        return JsonResponse({'error': 'no text provided'}, status=400)

    # Attempt to use a saved vectorizer and classifier. They should be saved in
    # <BASE_DIR>/ml_models/spam_vectorizer.joblib and spam_clf.joblib
    model = None
    vect = None

    if load is not None:
        model_dir = os.path.join(settings.BASE_DIR, 'ml_models')
        vect_path = os.path.join(model_dir, 'spam_vectorizer.joblib')
        clf_path = os.path.join(model_dir, 'spam_clf.joblib')
        try:
            if os.path.exists(vect_path) and os.path.exists(clf_path):
                vect = load(vect_path)
                model = load(clf_path)
        except Exception:
            # If loading fails, continue to heuristic fallback below.
            model = None
            vect = None

    if model is not None and vect is not None:
        # Use the vectorizer + model to predict. Support predict_proba and decision_function.
        try:
            X = vect.transform([text])
            label = 'ham'
            confidence = 0.5

            # Determine spam probability if possible
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X)[0]
                # find index of 'spam' if present in classes_, else use index 1
                classes = getattr(model, 'classes_', None)
                spam_idx = 1 if classes is None else int(list(classes).index('spam') if 'spam' in classes else 1)
                spam_prob = float(probs[spam_idx])
                label = 'spam' if spam_prob >= 0.5 else 'ham'
                confidence = float(spam_prob if label == 'spam' else 1.0 - spam_prob)
            elif hasattr(model, 'decision_function'):
                score = float(model.decision_function(X)[0])
                # map large positive -> spam, use sigmoid to convert to probability
                prob = 1.0 / (1.0 + np.exp(-score)) if np is not None else (0.5 if score >= 0 else 0.5)
                label = 'spam' if prob >= 0.5 else 'ham'
                confidence = float(prob if label == 'spam' else 1.0 - prob)
            else:
                pred = model.predict(X)[0]
                label = 'spam' if pred == 'spam' or str(pred) == '1' else 'ham'
                confidence = 0.9

            # Try to craft a short explanation using linear coef if available
            explanation = ''
            try:
                if hasattr(model, 'coef_') and hasattr(vect, 'get_feature_names_out'):
                    coefs = model.coef_[0]
                    names = vect.get_feature_names_out()
                    # get top positive contributing features
                    top_idx = list(np.argsort(coefs)[-6:][::-1])
                    top_feats = [names[i] + f'({coefs[i]:.2f})' for i in top_idx if coefs[i] > 0]
                    if top_feats:
                        explanation = 'Top positive features: ' + ', '.join(top_feats)
            except Exception:
                explanation = explanation or 'Model returned a prediction.'

            return JsonResponse({'label': label, 'confidence': confidence, 'explanation': explanation})
        except Exception:
            # If model fails at inference time, silently fall back to heuristic below
            model = None

    # If we reach here, either model not available or inference failed — use simple heuristic
    spam_keywords = ['free', 'buy', 'click', 'win', 'winner', 'prize', 'deal', 'offer', 'money', 'urgent', 'loan']
    found = []
    lowered = text.lower()
    for w in spam_keywords:
        if w in lowered:
            found.append(w)

    if found:
        # confidence increases with number of matches (capped 0.99)
        confidence = min(0.5 + 0.12 * len(found), 0.99)
        label = 'spam'
        explanation = 'Found suspicious keywords: ' + ', '.join(found)
    else:
        confidence = 0.88
        label = 'ham'
        explanation = 'No suspicious keywords found (heuristic).'

    return JsonResponse({'label': label, 'confidence': confidence, 'explanation': explanation})