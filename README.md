# ISSO RAG Chatbot

A RAG (retrieval-augmented generation) chatbot for Columbia's International Students and Scholars Office (ISSO).

Learning project focused on RAG techniques - chunking, embeddings, retrieval, and LLM integration.
If it works out, may reach out to ISSO about actual use.

## Structure

- `data/` - raw and processed ISSO source content (scraped pages, PDFs, etc.)
- `src/` - RAG pipeline code (ingestion, embedding, retrieval, chat)
- `notebooks/` - exploration and experimentation

## Status

Just scaffolded. Next: decide on data source (scrape isso.columbia.edu vs. provided docs) and pick an embedding/retrieval stack.
