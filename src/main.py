import json
import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ConfigDict
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from config import DATA_DIR
from query import answer

load_dotenv()

app = FastAPI()

REJECTIONS_LOG_PATH = DATA_DIR / "logs" / "rejections.jsonl"

# The Gemini free tier has a fixed daily quota shared by every user of this
# app, so one person in a loop is an outage for everyone. The limit is per
# client IP, which means uvicorn has to be told which proxy to believe:
#
#   uvicorn main:app --host 0.0.0.0 --port $PORT --forwarded-allow-ips='100.0.0.0/8'
#
# 100.0.0.0/8 is Railway's internal proxy range. Without it every request
# looks like it came from the proxy and shares one bucket; with '*' instead,
# uvicorn takes the leftmost X-Forwarded-For entry, which the caller can set
# to anything and so can bypass the limit.
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
@limiter.limit("10/minute;100/day")
def read_query(request: Request, body: ChatRequest):
    result = answer(body.message, log=logging_enabled)
    return {"response": result.text}

# def, not async def: Starlette runs a sync handler in the threadpool, so the
# file write does not block the event loop.
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
    """Append one row per refused request to data/logs/rejections.jsonl.

    Always on, regardless of LOGGING_ENABLED, because a row never holds the
    message text - only why the request was refused and how long it was.
    Refused requests never reach answer(), so without this file a limit that
    is too tight would be invisible.
    """
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    REJECTIONS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REJECTIONS_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(
            "".join(
                json.dumps({"timestamp": timestamp, **row}) + "\n" for row in rows
            )
        )
