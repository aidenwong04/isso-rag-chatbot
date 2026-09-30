import json
import os
import time
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ConfigDict
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from query import answer

load_dotenv()

app = FastAPI()

# The Gemini free tier has a fixed daily quota shared by every user of this
# app, so one person in a loop is an outage for everyone.
#
# These counters live in process memory: on Cloud Run they reset whenever an
# instance scales to zero and are not shared between instances. They are a
# placeholder until the counters move to Firestore.
#
# Keyed on client IP, but behind Firebase Hosting and Cloud Run that is the
# proxy's address unless uvicorn is told which X-Forwarded-For entry to trust.
# Verify request.client.host against a known IP after the first deploy.
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.get("/healthz")
async def read_health():
    return {"Health": "alive"}

@app.get("/hello")
async def read_hello():
    return {"Hello": "World"}

# need endpoint for taking in a query
class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    message: str = Field(min_length = 1, max_length = 1000, description = "The user's query message.")

logging_enabled = os.getenv("LOGGING_ENABLED", "False").lower() == "true"

# The body parameter cannot be called `request`: slowapi looks up a parameter
# of that name and requires it to be the starlette Request.
@app.post("/chat")
@limiter.limit("5/minute;50/day")
def read_query(request: Request, body: ChatRequest):
    received_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    started = time.perf_counter()
    result = answer(body.message)
    total_ms = int((time.perf_counter() - started) * 1000)

    # One row per answered request: the pipeline's fields plus what only the
    # endpoint can see. total_ms starts inside the handler, so it includes
    # the first request's index load but not time spent waiting for a
    # threadpool worker.
    if logging_enabled:
        log_rows(
            "query",
            [{"timestamp": received_at, **result.to_log_record(),
            "total_ms": total_ms}],
        )
    return {"response": result.text}

@app.exception_handler(RequestValidationError)
def handle_validation_error(request: Request, exc: RequestValidationError):
    log_rejections(
        [
            {
                "path": request.url.path,
                "reason": error["type"],
                # Stripped, because the length limits are checked after
                # stripping. None when the input was not a string (missing
                # field, bad JSON).
                "length": (
                    len(error["input"].strip())
                    if isinstance(error.get("input"), str)
                    else None
                ),
                "limit": (error.get("ctx") or {}).get(
                    "max_length", (error.get("ctx") or {}).get("min_length")
                ),
            }
            for error in exc.errors()
        ]
    )
    return JSONResponse(
        status_code=422,
        content={"message": "Invalid request data"},
    )

@app.exception_handler(RateLimitExceeded)
def handle_rate_limit(request: Request, exc: RateLimitExceeded):
    # Logged for the same reason rejections are: a throttled request never
    # reaches answer(), so without a row a limit set too low is invisible.
    log_rejections(
        [
            {
                "path": request.url.path,
                "reason": "rate_limited",
                "length": None,
                "limit": str(exc.limit.limit),
            }
        ]
    )
    return JSONResponse(
        status_code=429,
        content={"message": "Too many questions in a short time. Please wait a moment and try again."},
    )

def log_rejections(rows):
    """Log one row per refused request.

    Always on, regardless of LOGGING_ENABLED, because a row never holds the
    message text - only why the request was refused and how long it was.
    Refused requests never reach answer(), so without these rows a limit that
    is too tight would be invisible.
    """
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    log_rows("rejection", [{"timestamp": timestamp, **row} for row in rows])

def log_rows(stream, rows):
    """Print one JSON object per row to stdout, tagged with its stream.

    Cloud Run forwards stdout to Cloud Logging, which parses each JSON line
    into a searchable entry (filter on jsonPayload.log="rejection"). Files are
    not an option there: the container runs as a user that cannot write under
    /app, and the filesystem is gone when the instance stops.
    """
    for row in rows:
        print(json.dumps({"log": stream, **row}, ensure_ascii=False), flush=True)
