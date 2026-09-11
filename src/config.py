"""Settings shared by the pipeline.

The model ids live here rather than in each script because a corpus
experiment is only meaningful if everything except the corpus is held fixed.
Embedding with one model and querying with another produces wrong similarity
scores and raises nothing, and changing the model between two corpus versions
makes their comparison measure the model instead of the corpus.

RETRIEVAL_DOCUMENT and RETRIEVAL_QUERY are the two halves of Gemini's
asymmetric embedding scheme: chunks are indexed with the first, queries
embedded with the second. Mixing them up also fails silently.
"""

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSIONS = 768
GENERATION_MODEL = "gemini-3.1-flash-lite"

DOCUMENT_TASK_TYPE = "RETRIEVAL_DOCUMENT"
QUERY_TASK_TYPE = "RETRIEVAL_QUERY"

DEFAULT_TOP_K = 3
