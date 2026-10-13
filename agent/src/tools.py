"""
tools...

Read
Read Neighbour
Classify

This file defines those tools that the agent can call. 

"""

from dataclasses import dataclass
from pathlib import Path 
import json
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT / "src"))

from query import load_index

# need a lookup structure to get from id to record..

_id_to_record = None

def record_lookup():
    global _id_to_record
    if _id_to_record is None:
        _id_to_record = {}
        # need to load index..
        records, _ = load_index()
        for record in records:
            # A copy without the embedding, so no tool can hand it to the model
            # and no caller can change the cached corpus through it.
            _id_to_record[record["id"]] = {
                key: value for key, value in record.items() if key != "embedding"
            }
    return _id_to_record


def validate_id(chunk_id):
    ''' 
    id should be in the form of 'source_page::chunk_num'

    function returns the error message or success

    examples of malformed chunk_ids:
    7, ["a"], None
    "about-your-visa", "a::b::c"
    "::5", "about-your-visa::"
    "about-your-visa::x", "about-your-visa::-1"

    nonexistent ones:
    "about-your-visa::999"
    '''

    if not isinstance(chunk_id, str):
        return {
            "ok": False,
            "msg": "Malformed chunk id",
            "chunk": None
        }
    components = chunk_id.split("::")
    # split() only ever returns strings, so the chunk number is checked for
    # being all digits rather than for being an int.
    if len(components) != 2 or not components[0] or not components[1].isdigit():
        return {
            "ok": False,
            "msg": "Malformed chunk id",
            "chunk": None
        }

    id_to_record = record_lookup()
    if chunk_id in id_to_record:
        if id_to_record[chunk_id]["archived"]:
            return {
                "ok": False,
                "msg": "Attempted to view an archived page",
                "chunk": None
            }
        return {
            "ok": True,
            "msg": "success",
            "chunk": id_to_record[chunk_id]
        }

    return {
        "ok": False,
        "msg": "Nonexistent chunk_id",
        "chunk": None
    }

def generate_chunk(record):
    '''
    takes in a record, and generates the dict that represents that record.
    '''
    audience = record["audience"]
    if not audience:
        audience = "not stated"

    breadcrumb = list(record["breadcrumb"])
    
    return {
        "chunk_id": record["id"],
        "text": record["text"],
        "heading": record["heading"],
        "breadcrumb": breadcrumb,
        "audience": audience
    }

def refuse(msg):
    return {
        "ok": False,
        "msg": msg,
        "chunks": []
    }

# ----- THE READ TOOL -------

def read(chunk_id):
    '''
    takes in a chunk_id, and returns the chunk's text, heading, breadcrumb, audience if valid.
    else, returns the error message, doesn't crash the program.    
    '''
    id_validation = validate_id(chunk_id)
    if not id_validation["ok"]:
        return refuse(id_validation["msg"])

    chunk = id_validation['chunk']
    return {
        "ok": True,
        "msg": id_validation["msg"],
        "chunks" : [generate_chunk(chunk)]
    }

# ---------------------------

# ----- THE READ NEIGHBORS TOOL -------

def read_neighbors(chunk_id):
    '''
    this one shows the model the chunks that are next to chunk_id.
    '''
    id_validation = validate_id(chunk_id)
    if not id_validation["ok"]:
        return refuse(id_validation["msg"])
    
    # The previous chunk comes first, so the result reads in page order.
    # A neighbor that fails validation is the edge of the page, not an error:
    # "page::-1" is malformed and "page::<n>" is nonexistent, and both are
    # left out.
    page, number = chunk_id.split("::")
    neighbors = []
    for neighbor_id in (f"{page}::{int(number) - 1}", f"{page}::{int(number) + 1}"):
        neighbor_validation = validate_id(neighbor_id)
        if neighbor_validation["ok"]:
            neighbors.append(neighbor_validation["chunk"])

    return {
        "ok": True,
        "msg": id_validation["msg"],
        "chunks": [generate_chunk(neighbor) for neighbor in neighbors]
    }

# ---------------------------

# ----- THE CLASSIFy TOOL -------

def classify(label: int, rationale: str, evidence: list, shown_set: set):
    '''
    evidence is list of chunk IDs to back up the label.

    this tool looks at the label, rationale, set, and returns a confirmation on
    whether the label should be settled.

    refuses when
    - label is not the integer 0 or 1 (the string "1" is refused). 
    - rationale is empty or over 600 characters. 
    - evidence is empty, is not a list of chunk IDs, or cites a chunk that is not in this alert's shown set
    '''
    if type(label) != int:
        return {
            "ok": False,
            "msg": "label is not int"
        }

    if not isinstance(evidence, list):
        return {
            "ok": False,
            "msg": "evidence is not a list"
        }

    if not isinstance(rationale, str):
        return {
            "ok": False,
            "msg": "rationale is not str"
        }

    if label != 0 and label != 1:
        return {
            "ok": False,
            "msg": "label is not 0 or 1"
        }

    rationale = rationale.strip()

    if len(rationale) > 600:
        return {
            "ok": False,
            "msg": "rational exceeded character limit of 600"
        }

    if len(rationale) == 0:
        return {
            "ok": False,
            "msg": "rational is empty"
        }

    if evidence == []:
        return {
            "ok": False,
            "msg": "evidence list is empty"
        }

    for ev in evidence:
        validated = validate_id(ev)
        if not validated["ok"]:
            return {
                "ok": False,
                "msg": "A chunk id in evidence list was not valid: " + validated["msg"]
            }
        if ev not in shown_set:
            return {
                "ok": False,
                "msg": "Evidence not in shown_set"
            }
    
    return {
        "ok": True,
        "msg": "This classification passes checks, okay to settle the row"
    }

# -------------------------------

