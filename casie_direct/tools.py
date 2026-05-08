
import asyncio
import logging
import time
from functools import lru_cache
from pathlib import Path

from ddgs import DDGS
from tool_compat import RunContext, function_tool

from ec_faq import answer_ec_faq
from knowledge_store import answer_faq
from website_context import search_context

TIMETABLE_IMAGE_PATH = Path(__file__).resolve().parent / "KMS" / "timetable.png"
TIMETABLE_DISPLAY_MS = 8000

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


def _emit_ui_event(context: RunContext, event_type: str, **payload) -> bool:
    event_sink = context.metadata.get("event_sink")
    if not callable(event_sink):
        return False

    event = {
        "type": event_type,
        "worker_id": int(context.metadata.get("worker_id", 0) or 0),
        **payload,
    }
    event_sink(event)
    return True


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
    Display KMS/timetable.png on the user's screen for 8 seconds.
    Call this tool whenever the user asks to see the timetable, time table,
    or class schedule.
    """
    if not TIMETABLE_IMAGE_PATH.exists():
        logging.error("Timetable image not found: %s", TIMETABLE_IMAGE_PATH)
        return "I couldn't find the timetable image right now."

    shown = _emit_ui_event(
        context,
        "show_timetable",
        image_path=str(TIMETABLE_IMAGE_PATH),
        duration_ms=TIMETABLE_DISPLAY_MS,
    )
    if shown:
        return "I'm showing your class timetable for 8 seconds."

    return f"The class timetable image is available at {TIMETABLE_IMAGE_PATH}."


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
