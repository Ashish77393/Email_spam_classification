Deployment notes for Render / Heroku-style platforms

1. Ensure requirements.txt contains `gunicorn` and other runtime deps (already included).

2. Start command / Procfile

   - Render uses the `start` command configured in the service or a `Procfile`.
   - This project needs the Django WSGI application. Use:

     web: gunicorn spamClassification.wsgi:application --bind 0.0.0.0:$PORT

3. Allowed hosts

   - Add your site domain to `ALLOWED_HOSTS` in `spamClassification/settings.py` or set the environment variable `DJANGO_ALLOWED_HOSTS`.

4. Static files

   - We added WhiteNoise to the middleware and set STATIC_ROOT and STATICFILES_STORAGE в settings. On deploy, ensure `python manage.py collectstatic --noinput` runs (Render does this automatically in many setups).

5. Optional - model artifacts
   - If you plan to use the ML model, run `python scripts/train_spam_model.py` locally and commit `ml_models/` artifacts or store them in remote storage (S3) and load on startup.
