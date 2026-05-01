
import asyncio
import logging
import time
from functools import lru_cache

from ddgs import DDGS
from tool_compat import RunContext, function_tool

from ec_faq import answer_ec_faq
from knowledge_store import answer_faq
from website_context import search_context

_FAQ_HINTS = (
    "where",
    "lab",
    "room",
    "floor",
    "building",
    "seminar",
    "workshop",
    "project coordinator",
    "how many students",
)


def _normalize_query(query: str) -> str:
    return " ".join(query.strip().split())


@lru_cache(maxsize=512)
def _cached_context_query(query: str) -> str:
    return search_context(query)


@lru_cache(maxsize=512)
def _cached_ec_faq(query: str) -> str:
    return answer_ec_faq(query)


@lru_cache(maxsize=512)
def _cached_faq_fastpath(query: str) -> tuple[str | None, float]:
    return answer_faq(query)


@function_tool()
async def search_web(
    context: RunContext,  # type: ignore[valid-type]
    query: str,
) -> str:
    """
    Search the web using DuckDuckGo.
    """
    try:
        # Run sync DDGS search in thread pool to avoid blocking
        results = await asyncio.wait_for(
            asyncio.to_thread(lambda: list(DDGS().text(query, max_results=3))),
            timeout=5,
        )
        text = "\n".join(r.get("body", "") for r in results) if results else "No results found."
        logging.info("Search results for %r: %s", query, text)
        return text
    except asyncio.TimeoutError:
        logging.warning("Web search timed out for %r", query)
        return "Web search timed out. Please try a shorter query."
    except Exception as e:
        logging.error("Error searching the web for %r: %s", query, e)
        return f"An error occurred while searching the web for '{query}'."


@function_tool()
async def query_college_info(
    context: RunContext,  # type: ignore[valid-type]
    query: str,
) -> str:
    """
    Search the college website (RIT Kottayam - www.rit.ac.in) for information.
    Use this tool whenever the user asks anything about the college — such as
    principal, departments, courses, placement, contact info, admission, faculty,
    library, events, or any other college-related question.
    """
    try:
        t0 = time.perf_counter()
        q = _normalize_query(query)
        q_lower = q.lower()

        # If the LLM routed an FAQ-style question here, answer it directly.
        if any(hint in q_lower for hint in _FAQ_HINTS):
            faq_answer, faq_score = _cached_faq_fastpath(q)
            if faq_answer and faq_score >= 0.62:
                logging.info("College query %r answered via FAQ fast-path in %.2f ms", q, (time.perf_counter() - t0) * 1000)
                return faq_answer

        result = _cached_context_query(q)
        logging.info("College query %r completed in %.2f ms", q, (time.perf_counter() - t0) * 1000)
        logging.info("College website query %r: found %d chars", query, len(result))
        return result
    except Exception as e:
        logging.error("Error querying college website for %r: %s", query, e)
        return f"Could not retrieve college information for '{query}'."


@function_tool()
async def show_timetable(
    context: RunContext,  # type: ignore[valid-type]
) -> str:
    """
    Display the class timetable image on the user's screen.
    Call this tool whenever the user asks to see the timetable or their schedule.
    """
    # Keep the spoken reply natural while including the trigger word
    # that the frontend Timetable component listens for.
    return "I'm displaying your class timetable on the screen right now."


@function_tool()
async def query_ec_faq(
    context: RunContext,  # type: ignore[valid-type]
    query: str,
) -> str:
    """
    Answer fixed Electronics and Communication (EC) department FAQs.

    Backed by a small JSON file (KMS/ec_faq.json) so the LLM only sends
    the user's question as input, keeping token usage and latency low.
    """
    t0 = time.perf_counter()
    q = _normalize_query(query)
    result = _cached_ec_faq(q)
    logging.info("EC FAQ query %r completed in %.2f ms", q, (time.perf_counter() - t0) * 1000)
    return result
