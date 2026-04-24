
from __future__ import annotations

from typing   import Any
from pydantic import BaseModel, Field

from langgraph.graph         import StateGraph, END
from langchain_groq          import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from grid07.config   import settings, get_logger
from grid07.personas import Bot
from grid07.tools    import mock_searxng_search

log = get_logger(__name__)



#  1  DATA MODELS

class GraphState(BaseModel):
    bot            : Any  = None   # Bot dataclass
    search_query   : str  = ""
    search_results : str  = ""
    post_output    : Any  = None   # PostOutput once Node 3 runs

    class Config:
        arbitrary_types_allowed = True


class PostOutput(BaseModel):
    bot_id      : str = Field(description="Bot identifier e.g. Bot_A")
    topic       : str = Field(description="One-line topic the post is about")
    post_content: str = Field(description="The actual post — max 280 characters, opinionated")



# § 2  LLM


def _get_llm() -> ChatGroq:
    return ChatGroq(
        api_key     = settings.GROQ_API_KEY,
        model  = settings.GROQ_MODEL,
        temperature = 0.85,
    )



# 3  NODES

def node_decide_search(state: GraphState) -> dict:

    bot = state.bot
    log.info("[Node 1] %s deciding search query ...", bot.id)

    llm  = _get_llm()
    msgs = [
        SystemMessage(content=bot.system_prompt),
        HumanMessage(content=(
            "You are about to make a social media post.\n"
            "Decide what topic you want to post about today based on your persona.\n"
            "Write a SHORT search query (5-8 words) to find recent news on that topic.\n\n"
            "Respond with ONLY the search query — nothing else."
        )),
    ]

    response     = llm.invoke(msgs)
    search_query = response.content.strip().strip('"').strip("'")

    log.info("[Node 1] %s → query: %r", bot.id, search_query)
    return {"search_query": search_query}


def node_web_search(state: GraphState) -> dict:

    query = state.search_query
    log.info("[Node 2] Searching: %r", query)

    results = mock_searxng_search.invoke({"query": query})

    log.info("[Node 2] %d chars of context received", len(results))
    log.debug("[Node 2] Headlines:\n%s", results)
    return {"search_results": results}


def node_draft_post(state: GraphState) -> dict:

    bot     = state.bot
    results = state.search_results

    log.info("[Node 3] %s drafting post ...", bot.id)

    structured_llm = _get_llm().with_structured_output(PostOutput)

    msgs = [
        SystemMessage(content=bot.system_prompt),
        HumanMessage(content=(
            f"Today's relevant headlines:\n\n{results}\n\n"
            f"Write a highly opinionated social media post using these headlines as context.\n\n"
            f"Rules:\n"
            f"- Maximum {settings.MAX_POST_CHARS} characters\n"
            f"- Must reflect your persona strongly\n"
            f"- Reference the real news\n"
            f"- No hashtags\n"
            f"- bot_id must be exactly: {bot.id}"
        )),
    ]

    post: PostOutput = structured_llm.invoke(msgs)

    # hard enforce character limit
    if len(post.post_content) > settings.MAX_POST_CHARS:
        post = PostOutput(
            bot_id       = post.bot_id,
            topic        = post.topic,
            post_content = post.post_content[:settings.MAX_POST_CHARS],
        )

    log.info("[Node 3] %s → topic: %r", bot.id, post.topic)
    log.info("[Node 3] %s → post (%d chars): %s", bot.id, len(post.post_content), post.post_content)
    return {"post_output": post}



#  4  GRAPH ASSEMBLY


def _build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("decide_search", node_decide_search)
    graph.add_node("web_search",    node_web_search)
    graph.add_node("draft_post",    node_draft_post)

    graph.set_entry_point("decide_search")
    graph.add_edge("decide_search", "web_search")
    graph.add_edge("web_search",    "draft_post")
    graph.add_edge("draft_post",    END)

    return graph.compile()


_graph = _build_graph()



#  5  PUBLIC API

def generate_post(bot: Bot) -> PostOutput:

    log.info("Content engine starting — %s (%s)", bot.id, bot.name)

    final_state = _graph.invoke(GraphState(bot=bot))
    result      = final_state["post_output"]

    log.info("Content engine done — %s post generated", bot.id)
    return result