import json
import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ConfigDict

from config import DATA_DIR
from query import answer

load_dotenv()

app = FastAPI()

REJECTIONS_LOG_PATH = DATA_DIR / "logs" / "rejections.jsonl"

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

@app.post("/chat")
def read_query(request: ChatRequest):
    result = answer(request.message, log=logging_enabled)
    return {"response": result.text}

# def, not async def: Starlette runs a sync handler in the threadpool, so the
# file write does not block the event loop.
@app.exception_handler(RequestValidationError)
def handle_validation_error(request: Request, exc: RequestValidationError):
    log_rejections(request, exc.errors())
    return JSONResponse(
        status_code=422,
        content={"message": "Invalid request data"},
    )

def log_rejections(request, errors):
    """Append one row per validation error to data/logs/rejections.jsonl.

    Always on, regardless of LOGGING_ENABLED, because a row never holds the
    message text. Each error's "input" is the raw text the user sent, so the
    row is built from the fields we want rather than by copying the error.
    Rejected requests never reach answer(), so without this a limit that is
    too tight would be invisible.
    """
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = []
    for error in errors:
        value = error.get("input")
        context = error.get("ctx") or {}
        rows.append({
            "timestamp": timestamp,
            "path": request.url.path,
            "reason": error["type"],
            # Stripped, because the length limits are checked after stripping.
            # None when the input was not a string (missing field, bad JSON).
            "length": len(value.strip()) if isinstance(value, str) else None,
            "limit": context.get("max_length", context.get("min_length")),
        })

    REJECTIONS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REJECTIONS_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write("".join(json.dumps(row) + "\n" for row in rows))
