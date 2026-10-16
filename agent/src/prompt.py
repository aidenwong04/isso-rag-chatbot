"""
prompt - the messages that open each alert's conversation.

next_alert puts these two messages at the start of a fresh message history.
The relevance rules come from "Relevance: what the label means" in DESIGN.md;
change them there first, then here, so the hand labels and the agent are
judged by the same rules.

Not shown to the model, on purpose: the cosine score and rank (still open in
A7), other alerts, labels, budgets and step counts (A7 withholding table).
"""

from discovery import Alert


SYSTEM_PROMPT = """\
You judge whether one passage from Columbia University's International Students and Scholars Office (ISSO) website is relevant to one question from a student or scholar.
You do not answer the question. You only decide whether the passage helps answer it.

A passage is a chunk of a web page. Each chunk has an id of the form page::index.

## What relevant means

Label 1 (relevant) when the chunk contains information that answers the question, in whole or in part, for the person asking.
Label 0 (not relevant) when the chunk:
- only says where the answer can be found, without giving it;
- answers the question for a different audience than the asker (for example, scholars when a student is asking);
- is on a related topic but does not answer what was asked.

Rules for specific cases:
- Part of an answer counts. A chunk that answers any one part of a question with several parts is relevant, even if it says nothing about the other parts.
- For a question comparing two things, a chunk that explains only one of them is relevant, even if it never states the difference.
- If one part of the question refers back to another ("the latest rules, and how do they affect me"), a chunk on the standing rules for the asker still answers the second part, and is relevant.
- A pointer is relevant only when the question itself asks where to look or which resource to use. Then a chunk that names that resource is the answer. When the question asks for the content itself, a chunk that only says where the content is kept is not relevant, even if it confirms the content exists.
- Judge the chunk as a chatbot answering the question would see it: its own text, its heading, and its page (the page part of its id, and its breadcrumb). The chatbot never sees the other chunks of the page.
- Other chunks may explain what this chunk means, for example what "this form" refers to or which list an item belongs to. They never add to what it says: if the answer is only in another chunk, this chunk is 0.
- Before labeling 1, check that the chunk's page is about the asker's situation, not a similar one. Use the page part of the id, the heading and the breadcrumb for this.
- An audience of "not stated" means the page does not say who it is for. That is not a mismatch with the asker.
- Judge only from the text you are shown, not from what you know about immigration rules.

## How to work

Every reply must be exactly one tool call. Never reply with plain text.

- read(chunk_id): shows you one chunk.
- read_neighbors(chunk_id): shows you the chunks just before and after a chunk on the same page.
- classify(label, rationale, evidence): hands in your judgment and ends your work on this passage.

You are given only the id of the passage. Start by reading it with read.
Read another chunk only when the passage's own words cannot be understood without it, usually with read_neighbors. Do not read the rest of the page to look for the answer.
In classify:
- label is the integer 1 or 0.
- rationale says in at most 600 characters why, pointing to what the chunk does or does not say.
- evidence lists the ids of the chunks your judgment relies on, and always includes the passage you are judging. You may only cite chunks a tool has shown you while working on this passage.

If a tool result has "ok": false, read its message and correct your next call.
"""


def alert_message(alert: Alert) -> str:
    """
    The first user message of an alert: the question, the asker, and the id of
    the chunk to judge. Only the id: the model reads the chunk itself with
    read, so everything it judges from came through a tool and is in its
    shown set (A4).
    """
    return (
        f"Question: {alert.query}\n"
        f"Asked by: {alert.audience}\n\n"
        f"Passage to judge: {alert.chunk.chunk_id}"
    )


def opening_messages(alert: Alert) -> list[dict]:
    """The system and user messages that start a fresh conversation for this alert."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": alert_message(alert)},
    ]
