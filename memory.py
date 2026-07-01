"""
memory.py — Persistent vector memory for Echo bot using ChromaDB.

Embeddings are generated via HuggingFace Inference API so no local model
or PyTorch is needed. Works with any chromadb version.
"""

import logging
import os

import chromadb
import requests

log = logging.getLogger(__name__)

CHROMA_PATH = os.path.join(os.path.dirname(__file__), "echo_memory")
TOP_K = 5

_HF_TOKEN = os.getenv("HF_TOKEN", "")
_EMBED_URL = (
    "https://api-inference.huggingface.co/pipeline/feature-extraction/"
    "sentence-transformers/all-MiniLM-L6-v2"
)
_EMBED_DIM = 384  # all-MiniLM-L6-v2 output dimension

# ── Embedding ─────────────────────────────────────────────────────────────────

def _embed(texts: list) -> list:
    """
    Get sentence embeddings from HuggingFace Inference API.
    Falls back to zero vectors on error so the bot never crashes mid-chat.
    """
    try:
        resp = requests.post(
            _EMBED_URL,
            headers={"Authorization": f"Bearer {_HF_TOKEN}"},
            json={"inputs": texts, "options": {"wait_for_model": True}},
            timeout=30,
        )
        resp.raise_for_status()
        raw = resp.json()

        embeddings = []
        for item in raw:
            if item and isinstance(item[0], list):
                # Token-level output — mean pool to a single sentence vector
                n, dim = len(item), len(item[0])
                embeddings.append([sum(item[j][d] for j in range(n)) / n for d in range(dim)])
            else:
                embeddings.append(item)
        return embeddings
    except Exception as exc:
        log.warning("HF embed failed: %s — using zero vectors", exc)
        return [[0.0] * _EMBED_DIM for _ in texts]


# ── ChromaDB client ───────────────────────────────────────────────────────────

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
        log.info("ChromaDB initialised at %s", CHROMA_PATH)
    return _client


def _get_collection(user_id: str):
    """Return (or create) the ChromaDB collection for a Discord user."""
    return _get_client().get_or_create_collection(
        name=f"user-{user_id}",
        metadata={"hnsw:space": "cosine"},
    )


# ── Public API ────────────────────────────────────────────────────────────────

def add_memory(user_id: str, user_message: str, bot_reply: str) -> None:
    """Embed and store a single conversation exchange."""
    try:
        doc = f"User: {user_message}\nEcho: {bot_reply}"
        doc_id = str(abs(hash(doc)) % (10 ** 16))
        emb = _embed([doc])[0]
        _get_collection(user_id).upsert(
            ids=[doc_id],
            documents=[doc],
            embeddings=[emb],
            metadatas=[{"user_id": user_id}],
        )
    except Exception as exc:
        log.error("add_memory failed for %s: %s", user_id, exc)


def get_relevant_memories(user_id: str, query: str) -> str:
    """
    Return a formatted block of the TOP_K most relevant past exchanges,
    ready to inject into the AI prompt. Returns "" if nothing stored.
    """
    try:
        col = _get_collection(user_id)
        count = col.count()
        if count == 0:
            return ""
        query_emb = _embed([query])[0]
        results = col.query(
            query_embeddings=[query_emb],
            n_results=min(TOP_K, count),
            include=["documents"],
        )
        docs = results.get("documents", [[]])[0]
        if not docs:
            return ""
        return "[Past conversations with this user:\n" + "\n---\n".join(docs) + "\n]"
    except Exception as exc:
        log.error("get_relevant_memories failed for %s: %s", user_id, exc)
        return ""


def delete_user_memories(user_id: str) -> int:
    """Wipe all memories for a user. Returns deleted count."""
    try:
        col = _get_collection(user_id)
        count = col.count()
        _get_client().delete_collection(f"user-{user_id}")
        log.info("Deleted %d memories for user %s", count, user_id)
        return count
    except Exception as exc:
        log.error("delete_user_memories failed for %s: %s", user_id, exc)
        return 0


def get_memory_count(user_id: str) -> int:
    """Return how many memories are stored for a user."""
    try:
        return _get_collection(user_id).count()
    except Exception:
        return 0
