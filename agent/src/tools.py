"""
tools...

Read
Read Neighbour
Classify

This file defines those tools that the agent can call. 

"""

from pathlib import Path

from workspace import load_copy

# id -> record, one lookup per workspace, loaded on first use.
_lookups: dict[Path, dict[str, dict]] = {}

def record_lookup(workspace: Path):
    """
    The run's corpus copy (see workspace.py), keyed by chunk id. The tools
    read only this: archived pages and embeddings are not in it at all.
    """
    if workspace not in _lookups:
        _lookups[workspace] = {record["id"]: record for record in load_copy(workspace)}
    return _lookups[workspace]


def validate_id(chunk_id, workspace: Path):
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
    an archived page's chunks, which are not in the workspace copy
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

    id_to_record = record_lookup(workspace)
    if chunk_id in id_to_record:
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

def read(chunk_id, workspace: Path):
    '''
    takes in a chunk_id, and returns the chunk's text, heading, breadcrumb, audience if valid.
    else, returns the error message, doesn't crash the program.    
    '''
    id_validation = validate_id(chunk_id, workspace)
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

def read_neighbors(chunk_id, workspace: Path):
    '''
    this one shows the model the chunks that are next to chunk_id.
    '''
    id_validation = validate_id(chunk_id, workspace)
    if not id_validation["ok"]:
        return refuse(id_validation["msg"])
    
    # The previous chunk comes first, so the result reads in page order.
    # A neighbor that fails validation is the edge of the page, not an error:
    # "page::-1" is malformed and "page::<n>" is nonexistent, and both are
    # left out.
    page, number = chunk_id.split("::")
    neighbors = []
    for neighbor_id in (f"{page}::{int(number) - 1}", f"{page}::{int(number) + 1}"):
        neighbor_validation = validate_id(neighbor_id, workspace)
        if neighbor_validation["ok"]:
            neighbors.append(neighbor_validation["chunk"])

    return {
        "ok": True,
        "msg": id_validation["msg"],
        "chunks": [generate_chunk(neighbor) for neighbor in neighbors]
    }

# ---------------------------

# ----- THE CLASSIFy TOOL -------

def classify(label: int, rationale: str, evidence: list, shown_set: set, workspace: Path):
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
        validated = validate_id(ev, workspace)
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



# ----- TOOL DEFINITIONS -------
# What the model is told about each tool, in the OpenAI Chat Completions
# format. Kept next to the tools so a tool and its description cannot drift
# apart. The schemas describe the arguments; they do not enforce them (no
# "strict" mode), so a bad call still reaches the tool and is refused there,
# where the refusal is counted and traced.

CHUNK_ID_SCHEMA = {
    "type": "string",
    "description": "A chunk id of the form page::index, for example about-your-visa::3.",
}

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Show one chunk: its text, heading, breadcrumb and audience.",
            "parameters": {
                "type": "object",
                "properties": {"chunk_id": CHUNK_ID_SCHEMA},
                "required": ["chunk_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_neighbors",
            "description": (
                "Show the chunks just before and after a chunk on the same page, "
                "for context that was cut off. Returns fewer at the start or end of a page. "
                "To see further, call it again on a neighbor's id."
            ),
            "parameters": {
                "type": "object",
                "properties": {"chunk_id": CHUNK_ID_SCHEMA},
                "required": ["chunk_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "classify",
            "description": (
                "Hand in your judgment for the passage you are working on. "
                "This ends your work on it once it is accepted."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "label": {
                        "type": "integer",
                        "enum": [0, 1],
                        "description": "1 if the passage is relevant to the question, 0 if not.",
                    },
                    "rationale": {
                        "type": "string",
                        "maxLength": 600,
                        "description": "Why, in at most 600 characters, pointing to what the chunk does or does not say.",
                    },
                    "evidence": {
                        "type": "array",
                        "items": CHUNK_ID_SCHEMA,
                        "minItems": 1,
                        "description": "Ids of the chunks your judgment relies on. Only chunks you have been shown for this passage.",
                    },
                },
                "required": ["label", "rationale", "evidence"],
                "additionalProperties": False,
            },
        },
    },
]

# -------------------------------
