# The /chat API, as run on Cloud Run.
#
# Only what serving needs goes in: the code and the embedded corpus. The
# corpus ships inside the image on purpose, so code and corpus deploy and
# roll back together as one Cloud Run revision. Refreshing the corpus means
# re-embedding, committing, and redeploying.
#
# Cloud Run needs linux/amd64. Building locally on an Apple Silicon Mac
# produces arm64 by default, so pass --platform:
#
#   docker build --platform linux/amd64 -t isso-rag-chatbot .
#   docker run --rm -p 8080:8080 -e GEMINI_API_KEY isso-rag-chatbot

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies first, so a code-only change reuses this cached layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY data/embeddings/chunks.json data/embeddings/chunks.json

RUN useradd --system --no-create-home app
USER app

# main.py and query.py use flat imports, so run from src/.
WORKDIR /app/src

# Cloud Run sets PORT (8080 unless configured). exec makes uvicorn PID 1 so
# it receives SIGTERM directly and shuts down cleanly when an instance stops.
#
# On Cloud Run every request arrives from a proxy at a 169.254.x.x link-local
# address, with the real client appended to X-Forwarded-For. Trusting only
# that range makes uvicorn take the rightmost address that is not the proxy,
# so request.client.host (and the per-IP rate limit) is the real client. "*"
# would take the leftmost entry instead, which any client can forge. Outside
# Cloud Run no peer is in that range, so the header is ignored.
CMD exec uvicorn main:app --host 0.0.0.0 --port "${PORT:-8080}" \
    --forwarded-allow-ips 169.254.0.0/16
