import os
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq

# Load environment
load_dotenv('.env')
groq_key = os.getenv('GROQ_API_KEY')

# Page config
st.set_page_config(
    page_title="CrediTrust Complaint Analyst",
    page_icon="🏦",
    layout="wide"
)

# ── Load models (cached so they don't reload on every interaction) ──
@st.cache_resource
def load_resources():
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    chroma_client = chromadb.PersistentClient(path='vector_store/chroma_db')
    collection = chroma_client.get_collection("complaints")
    groq_client = Groq(api_key=groq_key)
    return embed_model, collection, groq_client

embed_model, collection, groq_client = load_resources()

# ── Core RAG functions ──
def retrieve_chunks(query, n_results=5):
    query_embedding = embed_model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        include=['documents', 'metadatas', 'distances']
    )
    chunks = []
    for doc, meta, dist in zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    ):
        chunks.append({
            'text': doc,
            'product': meta['product'],
            'relevance_score': round(1 - dist, 4)
        })
    return chunks

def generate_answer(query, chunks):
    context = ""
    for i, chunk in enumerate(chunks, 1):
        context += f"\n[Complaint {i} — {chunk['product']}]\n{chunk['text']}\n"

    prompt = f"""You are a financial analyst assistant for CrediTrust Financial.
Your job is to answer questions about customer complaints using only the provided complaint excerpts.

COMPLAINT EXCERPTS:
{context}

QUESTION: {query}

Instructions:
- Answer based only on the complaint excerpts above
- Be specific and cite patterns you see across complaints
- If the excerpts don't contain enough info, say so clearly
- Keep your answer concise (3-5 sentences)

ANSWER:"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.3
    )
    return response.choices[0].message.content

# ── UI Layout ──
st.title("🏦 CrediTrust Financial — Complaint Analyst")
st.markdown("Ask questions about customer complaints across Credit Cards, Personal Loans, Savings Accounts, and Money Transfers.")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    n_results = st.slider("Chunks to retrieve", min_value=3, max_value=10, value=5)
    show_sources = st.checkbox("Show source chunks", value=True)
    st.markdown("---")
    st.markdown("**Vector Store:** ChromaDB")
    st.markdown("**Embeddings:** all-MiniLM-L6-v2")
    st.markdown("**LLM:** Groq Llama-3.3-70b")
    st.markdown(f"**Complaints indexed:** 59,766 chunks")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message and show_sources:
            with st.expander("📄 Source chunks"):
                for i, chunk in enumerate(message["sources"], 1):
                    st.markdown(f"**Chunk {i}** | `{chunk['product']}` | relevance: `{chunk['relevance_score']}`")
                    st.text(chunk['text'][:300])
                    st.markdown("---")

# Chat input
if query := st.chat_input("Ask about customer complaints..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching complaints and generating answer..."):
            chunks = retrieve_chunks(query, n_results=n_results)
            answer = generate_answer(query, chunks)

        st.markdown(answer)

        if show_sources:
            with st.expander("📄 Source chunks"):
                for i, chunk in enumerate(chunks, 1):
                    st.markdown(f"**Chunk {i}** | `{chunk['product']}` | relevance: `{chunk['relevance_score']}`")
                    st.text(chunk['text'][:300])
                    st.markdown("---")

    # Save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": chunks
    })