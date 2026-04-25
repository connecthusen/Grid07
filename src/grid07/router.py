
from __future__ import annotations

import uuid
from dataclasses import dataclass

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from grid07.config   import settings, get_logger
from grid07.personas import BOTS, Bot

log = get_logger(__name__)

_COLLECTION_NAME = "bot_personas"
_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# RouteMatch — result of matching a post to a bot
# in: bot(Bot), score(float), matched(bool)
@dataclass(frozen=True)
class RouteMatch:
    bot    : Bot
    score  : float
    matched: bool


# PersonaRouter — routes posts to bots via ChromaDB cosine similarity
# embeds all bot personas at startup, then scores incoming posts against them
class PersonaRouter:

    # sets up ChromaDB, embeds all bot persona vectors
    def __init__(self) -> None:
        self._bot_list  : list[Bot] = list(BOTS.values())
        self._embed_fn              = SentenceTransformerEmbeddingFunction(
            model_name = _EMBEDDING_MODEL
        )

        self._client     = chromadb.Client()
        self._collection = self._client.create_collection(
            name               = _COLLECTION_NAME,
            embedding_function = self._embed_fn,
            metadata           = {"hnsw:space": "cosine"},
        )

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

    # embeds post, queries ChromaDB, converts distance → similarity score
    # ChromaDB gives cosine distance (0=identical, 2=opposite) → similarity = 1 - (distance/2)
    # in: post_content(str) | out: list[RouteMatch] sorted by score desc
    def _query(self, post_content: str) -> list[RouteMatch]:
        threshold = settings.SIMILARITY_THRESHOLD
        n         = len(self._bot_list)

        results = self._collection.query(
            query_texts = [post_content],
            n_results   = n,
        )

        ids       = results["ids"][0]
        distances = results["distances"][0]

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

    # returns all bots with their scores — matched flag set based on threshold
    # in: post_content(str), threshold(float|None) | out: list[RouteMatch]
    def route_post_to_bots(
        self,
        post_content : str,
        threshold    : float | None = None,
    ) -> list[RouteMatch]:
        if threshold is not None:
            original          = settings.SIMILARITY_THRESHOLD
            settings.SIMILARITY_THRESHOLD = threshold
            results           = self._query(post_content)
            settings.SIMILARITY_THRESHOLD = original
            return results

        return self._query(post_content)

    # returns only bots that passed the threshold
    # in: post_content(str), threshold(float|None) | out: list[RouteMatch]
    def get_matches(
        self,
        post_content : str,
        threshold    : float | None = None,
    ) -> list[RouteMatch]:
        return [
            r for r in self.route_post_to_bots(post_content, threshold)
            if r.matched
        ]
