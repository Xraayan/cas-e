"""
Website context search backed by SQLite knowledge store.

This module keeps a tiny in-memory context for prompt summary and uses
accuracy-first retrieval from the local database for query answering.
"""

import logging
import re

from knowledge_store import (
    answer_faq,
    ensure_knowledge_base,
    get_context_snapshot,
    list_faculty,
    search_docs,
    search_faculty,
)

logger = logging.getLogger(__name__)

_context: dict | None = None
_context_initialized = False

_NON_NAME_TOPICS = {
    "hostel", "accommodation", "residence", "placement", "admission", "fee", "fees", "exam",
    "result", "library", "canteen", "transport", "location", "address", "website", "course",
    "courses", "facility", "facilities", "building", "lab", "labs", "workshop", "seminar",
}

_PERSON_KEYWORDS = {
    "faculty", "teacher", "professor", "instructor", "staff", "hod", "doctor", "dr", "dr.",
}

_FAQ_HINTS = {
    "where", "lab", "room", "floor", "building", "seminar", "workshop", "project coordinator",
    "how many students", "student count",
}

_FIELD_QUERY_HINTS = {
    "principal", "email", "phone", "contact", "address", "website", "location", "hod email",
}


def _appears_to_be_name_query(query: str) -> bool:
    query_lower = query.lower().strip()
    tokens = [t for t in re.sub(r"[^a-z0-9\s]", " ", query_lower).split() if t]

    if any(h in query_lower for h in _FIELD_QUERY_HINTS):
        return False

    if any(topic in query_lower for topic in _NON_NAME_TOPICS) and not any(k in query_lower for k in _PERSON_KEYWORDS):
        return False

    if any(k in query_lower for k in _PERSON_KEYWORDS):
        return True

    if re.search(r"\b(who is|tell me about|information about|contact)\b", query_lower):
        # If phrase appears without obvious non-person topic, treat as person lookup.
        return not any(topic in query_lower for topic in _NON_NAME_TOPICS)

    # Plain short-name queries such as "Renu Jose" should map to faculty search.
    if 1 <= len(tokens) <= 3:
        if all(token.isalpha() for token in tokens) and not any(topic in query_lower for topic in _NON_NAME_TOPICS):
            return True

    return False


def init_context() -> None:
    """Initialize in-memory summary context and ensure SQLite KB is ready."""
    global _context, _context_initialized

    if _context_initialized:
        return

    ensure_knowledge_base()
    _context = get_context_snapshot()
    _context_initialized = True
    logger.info("Context initialized from SQLite facts")


def get_context() -> dict:
    """Get in-memory context used for compact system-prompt summary."""
    if _context is None:
        init_context()
    return _context or {}


def _format_faculty_result(item: dict) -> str:
    lines = [f"Name: {item.get('name', 'Unknown')}"]

    designation = item.get("designation")
    if designation:
        lines.append(f"Designation: {designation}")

    department = item.get("department")
    if department:
        lines.append(f"Department: {department}")

    discipline = item.get("discipline")
    if discipline:
        lines.append(f"Discipline: {discipline}")

    email = item.get("email")
    if email:
        lines.append(f"Email: {email}")

    phone = item.get("phone")
    if phone:
        lines.append(f"Phone: {phone}")

    return "\n".join(lines)


def _handle_teacher_query(query: str) -> str:
    query_lower = query.lower()

    list_mode = any(word in query_lower for word in ("list", "all", "show", "total", "how many"))

    if any(phrase in query_lower for phrase in ("all faculty", "list faculty", "all teachers", "teacher list", "all staff", "show all")):
        items = list_faculty(mode="all", limit=60)
        if not items:
            return "No faculty information available."

        lines = [f"Total Faculty/Staff: {len(items)}"]
        for idx, item in enumerate(items[:20], start=1):
            lines.append(f"{idx}. {item.get('name', 'Unknown')} - {item.get('designation', 'Staff')}")
        if len(items) > 20:
            lines.append(f"... and {len(items) - 20} more.")
        return "\n".join(lines)

    if list_mode and any(phrase in query_lower for phrase in ("associate professor", "associate professors")):
        items = list_faculty(mode="associate professor", limit=40)
        if not items:
            return "No associate professors found."
        return "\n".join([f"Associate Professors ({len(items)}):"] + [f"{i}. {x['name']}" for i, x in enumerate(items, 1)])

    if list_mode and any(phrase in query_lower for phrase in ("assistant professor", "assistant professors")):
        items = list_faculty(mode="assistant professor", limit=40)
        if not items:
            return "No assistant professors found."
        return "\n".join([f"Assistant Professors ({len(items)}):"] + [f"{i}. {x['name']}" for i, x in enumerate(items, 1)])

    if list_mode and any(phrase in query_lower for phrase in ("instructor", "instructors")):
        items = list_faculty(mode="instructor", limit=40)
        if not items:
            return "No instructors found."
        return "\n".join([f"Instructors ({len(items)}):"] + [f"{i}. {x['name']}" for i, x in enumerate(items, 1)])

    dr_list_patterns = ("who has dr", "with dr", "faculty with dr", "list dr", "all dr", "all doctors", "phd faculty")
    if list_mode and any(phrase in query_lower for phrase in dr_list_patterns):
        items = list_faculty(mode="dr", limit=60)
        if not items:
            return "No faculty found with Dr. title."
        lines = [f"Faculty with Dr. title ({len(items)} found):"]
        for idx, item in enumerate(items, start=1):
            lines.append(f"{idx}. {item.get('name', 'Unknown')} - {item.get('designation', '')}")
        return "\n".join(lines)

    hits = search_faculty(query, limit=5)
    if not hits:
        return "I could not find an exact faculty match. Please share the full or clearer name."

    best = hits[0]
    if best.get("score", 0.0) < 0.80:
        return "I am not fully confident about the faculty match. Please share one more name part."

    if len(hits) == 1 or best.get("score", 0.0) >= 0.95:
        return _format_faculty_result(best)

    lines = [f"Found {len(hits)} possible faculty matches:"]
    for idx, item in enumerate(hits, start=1):
        lines.append(f"{idx}. {item.get('name', 'Unknown')} - {item.get('designation', '')}")
    lines.append("Please tell me which one you mean.")
    return "\n".join(lines)


def _handle_principal_query(data: dict) -> str:
    principal = data.get("principal", {})
    if not principal:
        return "No principal information found."

    lines = []
    if principal.get("name"):
        lines.append(f"Principal: {principal['name']}")
    if principal.get("designation"):
        lines.append(f"Designation: {principal['designation']}")
    if principal.get("email"):
        lines.append(f"Email: {principal['email']}")
    if principal.get("phone"):
        lines.append(f"Phone: {principal['phone']}")

    return "\n".join(lines) if lines else "Principal information not available."


def _handle_contact_query(data: dict) -> str:
    contact = data.get("contact", {})
    ece_hod_email = contact.get("ece_hod_email")
    if not contact:
        return "No contact information found."

    lines = []
    if contact.get("address"):
        lines.append(f"Address: {contact['address']}")
    if contact.get("office_phone"):
        lines.append(f"Office Phone: {contact['office_phone']}")
    if contact.get("college_email"):
        lines.append(f"College Email: {contact['college_email']}")
    if ece_hod_email:
        lines.append(f"ECE HOD Email: {ece_hod_email}")
    if contact.get("website"):
        lines.append(f"Website: {contact['website']}")

    return "\n".join(lines) if lines else "Contact information not available."


def _handle_department_query(query_lower: str, data: dict) -> str:
    departments = data.get("departments", [])
    if not departments:
        hits = search_docs(query_lower, limit=5)
        if not hits:
            return "No department information found."
        lines = ["Relevant department information:"]
        for idx, hit in enumerate(hits[:3], start=1):
            lines.append(f"{idx}. {hit.get('content', '')}")
        return "\n".join(lines)

    dept_keywords = {
        "computer": ["Computer Science", "CSE"],
        "civil": ["Civil"],
        "mechanical": ["Mechanical"],
        "electrical": ["Electrical", "EEE"],
        "electronics": ["Electronics", "ECE"],
        "architecture": ["Architecture"],
        "mca": ["Computer Applications", "MCA"],
    }

    for keyword, names in dept_keywords.items():
        if keyword in query_lower:
            for dept in departments:
                dept_name = dept.get("name", "")
                if any(n.lower() in dept_name.lower() for n in names):
                    lines = [f"Department: {dept_name}"]
                    hod = dept.get("hod", {})
                    if hod and hod.get("name"):
                        lines.append(f"HOD: {hod.get('name')} - {hod.get('designation', '')}")
                    return "\n".join(lines)

    lines = [f"RIT Departments ({len(departments)} total):"]
    for dept in departments:
        hod_name = dept.get("hod", {}).get("name", "TBD")
        lines.append(f"- {dept.get('name', 'Unknown')} (HOD: {hod_name})")
    return "\n".join(lines)


def search_context(query: str, max_results: int = 8) -> str:
    """Search college knowledge with deterministic routing + confidence thresholds."""
    if _context is None:
        init_context()

    data = _context or {}
    if not data:
        return "No college context available."

    query_lower = query.lower()

    if any(word in query_lower for word in ("principal", "director", "head")):
        return _handle_principal_query(data)

    if any(word in query_lower for word in ("contact", "phone", "email", "address", "location", "hod email")):
        return _handle_contact_query(data)

    if any(hint in query_lower for hint in _FAQ_HINTS):
        faq_answer, faq_score = answer_faq(query)
        if faq_answer and faq_score >= 0.62:
            return faq_answer

    if _appears_to_be_name_query(query):
        return _handle_teacher_query(query)

    if any(word in query_lower for word in ("department", "faculty", "hod")):
        return _handle_department_query(query_lower, data)

    doc_hits = search_docs(query, limit=max_results)
    if not doc_hits:
        return "No relevant information found for this query."

    best = doc_hits[0]
    if best.get("score", 0.0) < 0.45:
        return "I am not fully sure about this answer. Please ask with more detail."

    lines = ["Relevant college information:"]
    for idx, hit in enumerate(doc_hits[:3], start=1):
        lines.append(f"{idx}. {hit.get('content', '')}")

    return "\n".join(lines)
