


import json
import os
import re
import joblib
import numpy as np
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.shortcuts import render

def Home(request):
    return render(request, 'Home.html')

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Load Model + Vectorizer once for speed
# MODEL_PATH = os.path.join(settings.BASE_DIR, "ml_models", "model1.pkl")
# VECT_PATH  = os.path.join(settings.BASE_DIR, "ml_models", "vector.pkl")

model = joblib.load('model1.pkl')
vectorizer = joblib.load('vector.pkl')


@csrf_exempt
@csrf_exempt
def classify(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    try:
        data = json.loads(request.body.decode("utf-8"))
        text = data.get("text", "").strip()
    except:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    if not text:
        return JsonResponse({"error": "No text provided"}, status=400)

    cleaned = clean_text(text)
    X = vectorizer.transform([cleaned])

    pred = model.predict(X)[0]   # ML model result
    label = "spam" if pred == 1 else "ham"

    # ------------------------------
    #  🔥 Keyword Boosting System
    # ------------------------------
    spam_keywords = [
    # Financial & winning
    "win money", "prize", "lottery", "bonus", "reward", "cash",
    "credit", "profit", "investment", "guaranteed return",

    # Scam / clickbait
    "click here", "open link", "access your review", "see your report",
    "button below", "verify account", "urgent attention", "limited time",
    "action required", "immediate response", "review is ready",

    # Resume/Email generic spam
    "resume review", "your review is ready", "see your review",
    "cv review", "profile shortlisting", "job shortlisted",

    # Marketing promotions
    "special offer", "free trial", "discount available", "exclusive deal"
]

    for keyword in spam_keywords:
        if keyword in cleaned:
            label = "spam"
            boosted = True
            break
    else:
        boosted = False

    # ------------------------------
    # Confidence calculation
    # ------------------------------
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(X)[0][1])  # spam probability      
        if boosted:
            prob = max(prob, 0.95)   # if keyword found → make spam more confident
        confidence = prob if label == "spam" else 1 - prob
    else:
        confidence = 0.90 if label == "spam" else 0.80

    return JsonResponse({
        "label": label,
        "confidence": round(confidence, 3),
        "keyword_matched": keyword if boosted else None,
        "message": "Boosted using rule + ML" if boosted else "ML probability based"
    })
