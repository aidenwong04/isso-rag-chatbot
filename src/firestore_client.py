"""The one Firestore client for this process.

Request logs now, rate-limit counters next: both need Firestore, and neither
belongs in the endpoint file, so the client lives here and is created once
per process, the same way query.py holds the Gemini client and the index.

Created on first use rather than at import, so `import main` works without
Google configured, and so .env (which sets FIRESTORE_EMULATOR_HOST locally)
is loaded before the client decides where to connect. The project is pinned
in config.py rather than taken from gcloud's default, which on this laptop
is a different app's project.
"""

import os

from dotenv import load_dotenv
from google.api_core.retry import Retry
from google.cloud import firestore

from config import (DATA_DIR, FIRESTORE_PROJECT)

_client_cache = None

def get_firestore_client():
    global _client_cache
    if _client_cache is None:
        load_dotenv(DATA_DIR.parent / ".env")
        firestore_emulator_host = os.environ.get("FIRESTORE_EMULATOR_HOST")
        if firestore_emulator_host:
            print(f"Connecting to Firestore emulator at {firestore_emulator_host}", flush=True)
        else:
            print(f"Connecting to PRODUCTION Firestore ({FIRESTORE_PROJECT})", flush=True)
        _client_cache = firestore.Client(project=FIRESTORE_PROJECT)
    return _client_cache

# The write happens inside the request, before the response is sent: Cloud
# Run throttles CPU once a response is out, so a write queued for after it
# could stall indefinitely. The cap keeps a slow or unreachable Firestore to
# at most a few seconds on top of an answer, never a failed one.
WRITE_TIMEOUT_SECONDS = 3
_write_retry = Retry(initial=0.2, maximum=1.0, timeout=WRITE_TIMEOUT_SECONDS)


def log_request(row):
    """Add one row to the `requests` collection. Never raises.

    A failure is printed (reaching Cloud Logging on Cloud Run) and dropped:
    losing a metrics row is better than failing the student's request.
    """
    try:
        get_firestore_client().collection("requests").add(
            row, timeout=WRITE_TIMEOUT_SECONDS, retry=_write_retry
        )
    except Exception as error:  # noqa: BLE001 - logging must not break /chat
        print(f"Failed to log request to Firestore: {type(error).__name__}: {error}", flush=True)
