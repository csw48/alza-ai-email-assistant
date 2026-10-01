import os

# Nastaviť pred importom app.config – N8N_API_KEY je povinný.
os.environ.setdefault("N8N_API_KEY", "test-secret")
os.environ.pop("GEMINI_API_KEY", None)
