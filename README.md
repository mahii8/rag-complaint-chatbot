# RAG Complaint Chatbot — CrediTrust Financial

## Overview
An intelligent complaint-answering chatbot that enables internal teams at
CrediTrust Financial to ask plain-English questions about customer complaints
and receive synthesized, evidence-backed answers in seconds.

## Products Covered
- Credit Cards
- Personal Loans
- Savings Accounts
- Money Transfers

## Architecture
1. **EDA & Preprocessing** — Clean and filter CFPB complaint data
2. **Chunking & Embedding** — Convert narratives to vector embeddings
3. **RAG Pipeline** — Semantic search + LLM generation
4. **Streamlit UI** — Interactive chat interface

## Tasks
| Task | Description | Branch |
|------|-------------|--------|
| Task 1 | EDA and data preprocessing | task-1 |
| Task 2 | Chunking, embedding, vector store | task-2 |
| Task 3 | RAG pipeline and evaluation | task-3 |
| Task 4 | Interactive chat UI | task-4 |

## How to Reproduce
1. Clone the repository
2. Install: pip install -r requirements.txt
3. Download CFPB dataset and place in data/raw/
4. Run notebooks in order
5. Launch app: streamlit run app.py

## Author
Mahi | 10 Academy x Kifiya | Week 7 | June 2026
