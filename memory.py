"""
memory.py — Persistent vector memory for Echo bot using ChromaDB.

Each Discord user gets their own isolated collection of embeddings.
The bot stores key conversation moments and retrieves the most
semantically relevant ones when building a response prompt.
"""

import logging
import os
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

# Persist to disk so memory survives restarts.
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "echo_memory")

# How many past memories to surface per response.
TOP_K = 5

# Sentence-transformers model used for embeddings.
# "all-MiniLM-L6-v2" is fast, small (~80MB), and accurate enough for this use case.
EMBED_MODEL = "all-MiniLM-L6-v2"

# ── Client singleton ──────────────────────────────────────────────────────────

_client: Optional[chromadb.PersistentClient] = None
_embed_fn: Optional[embedding_functions.SentenceTransformerEmbeddingFunction] = None


def _get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
        log.info("ChromaDB client initialised at %s", CHROMA_PATH)
    return _client


def _get_embed_fn() -> embedding_functions.SentenceTransformerEmbeddingFunction:
    global _embed_fn
    if _embed_fn is None:
        _embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL
        )
        log.info("Embedding function loaded: %s", EMBED_MODEL)
    return _embed_fn


def _collection_name(user_id: str) -> str:
    """Each user gets their own ChromaDB collection."""
    # ChromaDB collection names must be 3-63 chars, alphanumeric + hyphens.
    return f"user-{user_id}"


def _get_collection(user_id: str) -> chromadb.Collection:
    """Return (creating if needed) the collection for a given user."""
    return _get_client().get_or_create_collection(
        name=_collection_name(user_id),
        embedding_function=_get_embed_fn(),
        metadata={"hnsw:space": "cosine"},
    )


# ── Public API ────────────────────────────────────────────────────────────────

def add_memory(user_id: str, user_message: str, bot_reply: str) -> None:
    """
    Store a single conversation exchange as a memory.

    The document is the full exchange so the embedding captures both sides.
    A unique ID prevents duplicates if the same message is added twice.
    """
    try:
        collection = _get_collection(user_id)
        doc = f"User: {user_message}\nEcho: {bot_reply}"
        # Use a hash of the document as the ID to deduplicate.
        doc_id = str(abs(hash(doc)) % (10 ** 16))
        collection.upsert(
            ids=[doc_id],
            documents=[doc],
            metadatas=[{"user_id": user_id}],
        )
        log.debug("Memory stored for user %s (id=%s)", user_id, doc_id)
    except Exception as e:
        log.error("Failed to store memory for user %s: %s", user_id, e)


def get_relevant_memories(user_id: str, query: str) -> str:
    """
    Retrieve the TOP_K most semantically similar past exchanges.

    Returns a formatted string ready to inject into the prompt, or ""
    if nothing relevant was found.
    """
    try:
        collection = _get_collection(user_id)
        count = collection.count()
        if count == 0:
            return ""

        results = collection.query(
            query_texts=[query],
            n_results=min(TOP_K, count),
            include=["documents"],
        )
        docs = results.get("documents", [[]])[0]
        if not docs:
            return ""

        joined = "\n---\n".join(docs)
        return f"[Past conversations with this user:\n{joined}\n]"
    except Exception as e:
        log.error("Failed to retrieve memories for user %s: %s", user_id, e)
        return ""


def delete_user_memories(user_id: str) -> int:
    """
    Wipe all stored memories for a user. Returns the number of documents deleted.
    Used by the /forget_me command.
    """
    try:
        collection = _get_collection(user_id)
        count = collection.count()
        _get_client().delete_collection(name=_collection_name(user_id))
        log.info("Deleted %d memories for user %s", count, user_id)
        return count
    except Exception as e:
        log.error("Failed to delete memories for user %s: %s", user_id, e)
        return 0


def get_memory_count(user_id: str) -> int:
    """Return how many memories are stored for a user."""
    try:
        return _get_collection(user_id).count()
    except Exception:
        return 0
