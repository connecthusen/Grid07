
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from grid07.config   import settings, get_logger
from grid07.personas import BOTS, Bot


log = get_logger(__name__)


@dataclass(frozen=True)
class RouteMatch:
    bot    : Bot
    score  : float
    matched: bool


class PersonaRouter:

    def __init__(self) -> None:
        self._bot_list    : list[Bot] = list(BOTS.values())
        persona_texts     : list[str] = [
            bot.persona_vector_text for bot in self._bot_list
        ]

        self._vectorizer : TfidfVectorizer = TfidfVectorizer(
            sublinear_tf = True,
            stop_words   = "english",
            norm         = "l2",
        )

        self._persona_matrix = self._vectorizer.fit_transform(persona_texts)

        log.info(
            "PersonaRouter ready — bots: %d | vocab: %d words | threshold: %.2f",
            len(self._bot_list),
            len(self._vectorizer.vocabulary_),
            settings.SIMILARITY_THRESHOLD,
        )


    def route(
        self,
        post_content : str,
        threshold    : float | None = None,
    ) -> list[RouteMatch]:


        cutoff : float = (
            threshold if threshold is not None
            else settings.SIMILARITY_THRESHOLD
        )

        post_vec = self._vectorizer.transform([post_content])

        raw_scores : np.ndarray = cosine_similarity(
            post_vec, self._persona_matrix
        ).flatten()

        results : list[RouteMatch] = [
            RouteMatch(
                bot     = bot,
                score   = round(float(score), 4),
                matched = float(score) >= cutoff,
            )
            for bot, score in zip(self._bot_list, raw_scores)
        ]

        results.sort(key=lambda r: r.score, reverse=True)

        matched_ids = [r.bot.id for r in results if r.matched]
        log.info(
            "Post routed — matched: %s",
            matched_ids if matched_ids else "none",
        )
        log.debug(
            "All scores — %s",
            " | ".join(f"{r.bot.id}: {r.score:.4f}" for r in results),
        )

        return results


    def get_matches(
        self,
        post_content : str,
        threshold    : float | None = None,
    ) -> list[RouteMatch]:

        return [
            match
            for match in self.route(post_content, threshold)
            if match.matched
        ]