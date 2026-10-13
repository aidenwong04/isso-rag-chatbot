"""
discovery - the first stage in the pipeline.

It takes a set of queries from (agent/queries/dev/queries.json)

for each query in that set:
1. calls retrieve() - retrieving relevant chunks for that query.
2. returns a set of alerts. there should be nk alerts, 
   n is number of queries, k is how many chunks to retrieve per query.

"""

from dataclasses import dataclass
from pathlib import Path 
import json
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT / "src"))

from query import RetrievedChunk, retrieve

@dataclass
class Alert:
    """One (query, chunk) pair: the unit the agent labels."""

    query_id: str
    query: str
    audience: str
    chunk: RetrievedChunk

    @property
    def alert_id(self):
        # The one place this string is built. labels/dev.json and report.json
        # both join on it.
        return f"{self.query_id}::{self.chunk.chunk_id}"

def discover(file_path, k):
    """ 
    returns a list of alerts [alert1, alert2, alert3]
    file - should be a json file containing the queries.
    k - parameter to pass into retrieve() - (how many chunks to retrieve)
    """

    with open(file_path) as file:
        queries = json.load(file)

    '''
    For one alert, I need: query_id, chunk_id, query, chunk, audience.

    RetrievedAlert contains: _____, chunk_id, ______, chunk (which is text), ______
    RetrievedChunk(
                chunk_id=records[i]["id"],
                score=float(scores[i]),
                source=records[i]["source"],
                url=records[i]["url"],
                heading=records[i]["heading"],
                text=records[i]["text"],
                audience=records[i].get("audience"),
            )

    So I need to store query_id and query, and query's audience.
    '''

    retrievals = []
    for query in queries:
        chunk_list = []
        for chunk in retrieve(query["text"],k,query["audience"]):
            chunk_list.append((query["text"],query["id"],chunk,query["audience"]))
        retrievals.append(chunk_list)

    '''
    retrievals = [
        [(query, query_id, chunkA, query_audience), (query, query_id, chunkB, query_audience), (query, query_id, chunkC, query_audience)],   # for dev-q01
        ...
    ]
    '''

    alerts = []
    for chunks in retrievals:
        for i in range(len(chunks)):
            alerts.append(Alert(query_id=chunks[i][1],chunk=chunks[i][2],query=chunks[i][0],audience=chunks[i][3]))

    '''
    An Alert's shape.

    Alert(
        query_id, query
        chunk = RetreivedChunk
        audience = query_audience
    )

    '''

    return alerts

if __name__ == "__main__":
    DEV_DIR = REPO_ROOT / "agent" / "queries" / "dev"

    alerts = discover(DEV_DIR / "queries.json", 3)
    print(f"{len(alerts)} alerts")
