

from __future__ import annotations

import uuid
from dataclasses import dataclass

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from grid07.config   import settings, get_logger
from grid07.personas import BOTS, Bot

log = get_logger(__name__)

# ChromaDB collection name for persona vectors
_COLLECTION_NAME = "bot_personas"
_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ══════════════════════════════════════════════════════════════════════════════
# § 1  DATA MODEL
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class RouteMatch:
    bot    : Bot
    score  : float
    matched: bool


# ══════════════════════════════════════════════════════════════════════════════
# § 2  PERSONA ROUTER (ChromaDB)
# ══════════════════════════════════════════════════════════════════════════════

class PersonaRouter:
    """
    Routes incoming posts to matching bots via ChromaDB vector similarity.

    Initialisation (once at startup):
        - Creates an in-memory ChromaDB client
        - Embeds each bot's persona_vector_text with sentence-transformers
        - Stores all persona vectors in a single ChromaDB collection

    Per-post routing:
        - Embeds the post with the same model
        - Queries ChromaDB → returns distances for all bots
        - Converts distances to similarity scores
        - Applies threshold → returns RouteMatch list

    Note on ChromaDB distances:
        ChromaDB returns cosine DISTANCE (0 = identical, 2 = opposite).
        We convert to similarity:  similarity = 1 - (distance / 2)
        This maps the result to [0.0, 1.0] matching standard cosine similarity.
    """

    def __init__(self) -> None:
        self._bot_list  : list[Bot] = list(BOTS.values())
        self._embed_fn              = SentenceTransformerEmbeddingFunction(
            model_name = _EMBEDDING_MODEL
        )

        # in-memory ChromaDB — no disk, clean state on every run
        self._client     = chromadb.Client()
        self._collection = self._client.create_collection(
            name               = _COLLECTION_NAME,
            embedding_function = self._embed_fn,
            metadata           = {"hnsw:space": "cosine"},
        )

        # embed and store all persona vectors
        self._collection.add(
            ids        = [bot.id for bot in self._bot_list],
            documents  = [bot.persona_vector_text for bot in self._bot_list],
            metadatas  = [{"name": bot.name} for bot in self._bot_list],
        )

        log.info(
            "PersonaRouter ready — %d bots indexed | model: %s | threshold: %.2f",
            len(self._bot_list),
            _EMBEDDING_MODEL,
            settings.SIMILARITY_THRESHOLD,
        )

    # ── private ───────────────────────────────────────────────────────────────

    def _query(self, post_content: str) -> list[RouteMatch]:
        """
        Embed post, query ChromaDB, convert distances → RouteMatch list.
        Results sorted by score descending (best match first).
        """
        threshold = settings.SIMILARITY_THRESHOLD
        n         = len(self._bot_list)

        results = self._collection.query(
            query_texts = [post_content],
            n_results   = n,
        )

        ids       = results["ids"][0]
        distances = results["distances"][0]

        # ChromaDB cosine distance → similarity score
        # distance=0 means identical → similarity=1.0
        # distance=2 means opposite  → similarity=0.0
        matches: list[RouteMatch] = []
        for bot_id, distance in zip(ids, distances):
            score   = round(1 - (distance / 2), 4)
            bot     = BOTS[bot_id]
            matches.append(RouteMatch(
                bot     = bot,
                score   = score,
                matched = score >= threshold,
            ))

        matches.sort(key=lambda r: r.score, reverse=True)

        matched_ids = [r.bot.id for r in matches if r.matched]
        log.info(
            "Post routed — matched: %s",
            matched_ids if matched_ids else "none",
        )
        log.debug(
            "All scores — %s",
            " | ".join(f"{r.bot.id}: {r.score:.4f}" for r in matches),
        )

        return matches

    # ── public API ────────────────────────────────────────────────────────────

    def route_post_to_bots(
        self,
        post_content : str,
        threshold    : float | None = None,
    ) -> list[RouteMatch]:
        """
        Score all bots against the post. Returns ALL results sorted by score.
        Matches threshold from settings unless overridden.

        Named to match the assignment spec:
            route_post_to_bots(post_content: str, threshold: float = 0.85)
        """
        if threshold is not None:
            original          = settings.SIMILARITY_THRESHOLD
            settings.SIMILARITY_THRESHOLD = threshold
            results           = self._query(post_content)
            settings.SIMILARITY_THRESHOLD = original
            return results

        return self._query(post_content)

    def get_matches(
        self,
        post_content : str,
        threshold    : float | None = None,
    ) -> list[RouteMatch]:
        """
        Return ONLY bots that matched (score >= threshold).
        Used by main.py and tests.
        """
        return [
            r for r in self.route_post_to_bots(post_content, threshold)
            if r.matched
        ]