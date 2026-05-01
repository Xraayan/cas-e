import json
import logging
import os
import re
import sqlite3
import threading
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
KMS_DIR = BASE_DIR / "KMS"
WEBSITE_CACHE_FILE = KMS_DIR / "edited.website_cache.json"
FAQ_FILE = KMS_DIR / "ec_faq.json"
DB_FILE = KMS_DIR / "knowledge.db"
RUNTIME_DB_ONLY = os.getenv("CASIE_DB_RUNTIME_ONLY", "1").strip().lower() in {"1", "true", "yes", "on"}

_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "for", "from", "how", "i", "in", "is",
    "it", "me", "my", "of", "on", "or", "please", "tell", "that", "the", "their", "there",
    "this", "to", "was", "what", "when", "where", "which", "who", "with",
}

_NAME_TITLES = {"dr", "dr.", "prof", "prof.", "mr", "mr.", "mrs", "mrs.", "ms", "ms."}

_DB_LOCK = threading.Lock()
_DB_READY = False
_FTS_ENABLED = False


def _normalize(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())
    return " ".join(cleaned.split())


def _tokenize(text: str) -> list[str]:
    tokens = [t for t in _normalize(text).split() if t and t not in _STOP_WORDS]
    return tokens


def _name_parts(name: str) -> tuple[str, str, str]:
    raw_tokens = [t for t in name.split() if t]
    filtered = [t for t in raw_tokens if t.lower() not in _NAME_TITLES]
    if not filtered:
        filtered = raw_tokens

    first = _normalize(filtered[0]) if filtered else ""
    last = _normalize(filtered[-1]) if filtered else ""
    full = _normalize(name)
    return first, last, full


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _create_schema(conn: sqlite3.Connection) -> None:
    global _FTS_ENABLED

    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS faculty (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            normalized_name TEXT NOT NULL,
            first_name TEXT,
            last_name TEXT,
            designation TEXT,
            discipline TEXT,
            email TEXT,
            phone TEXT,
            department TEXT,
            role_group TEXT,
            has_dr INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS faculty_alias (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            faculty_id INTEGER NOT NULL,
            alias TEXT NOT NULL,
            FOREIGN KEY(faculty_id) REFERENCES faculty(id) ON DELETE CASCADE,
            UNIQUE(faculty_id, alias)
        );

        CREATE TABLE IF NOT EXISTS faq (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            normalized_question TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS facts (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS doc_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            section TEXT NOT NULL,
            content TEXT NOT NULL,
            normalized_content TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_faculty_name ON faculty(normalized_name);
        CREATE INDEX IF NOT EXISTS idx_faculty_first ON faculty(first_name);
        CREATE INDEX IF NOT EXISTS idx_faculty_last ON faculty(last_name);
        CREATE INDEX IF NOT EXISTS idx_faculty_designation ON faculty(designation);
        CREATE INDEX IF NOT EXISTS idx_alias_alias ON faculty_alias(alias);
        CREATE INDEX IF NOT EXISTS idx_faq_normalized_question ON faq(normalized_question);
        CREATE INDEX IF NOT EXISTS idx_docs_section ON doc_chunks(section);
        """
    )

    try:
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS faq_fts USING fts5(question, answer, tokenize='porter')"
        )
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts USING fts5(section, content, tokenize='porter')"
        )
        _FTS_ENABLED = True
    except sqlite3.OperationalError:
        _FTS_ENABLED = False
        logger.warning("SQLite FTS5 is unavailable; falling back to LIKE + fuzzy matching")


def _get_file_mtime(path: Path) -> int:
    if not path.exists():
        return 0
    return int(path.stat().st_mtime)


def _meta_get(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None


def _meta_set(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )


def _load_json(path: Path, expected_type: type[Any]) -> Any:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, expected_type):
        raise ValueError(f"Invalid data in {path.name}: expected {expected_type.__name__}")
    return data


def _collect_faculty_rows(website_data: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    def add_person(person: dict[str, Any], dept: str, role_group: str) -> None:
        name = str(person.get("name", "")).strip()
        if not name:
            return
        rows.append(
            {
                "name": name,
                "designation": str(person.get("designation", "")).strip(),
                "discipline": str(person.get("discipline", "")).strip(),
                "email": str(person.get("email", "")).strip(),
                "phone": str(person.get("phone", "")).strip(),
                "department": dept,
                "role_group": role_group,
            }
        )

    for dept in website_data.get("departments", []):
        dept_name = str(dept.get("name", "Unknown")).strip() or "Unknown"
        hod = dept.get("hod")
        if isinstance(hod, dict):
            add_person(hod, dept_name, "hod")
        for item in dept.get("regular_faculty", []):
            if isinstance(item, dict):
                add_person(item, dept_name, "regular")
        for item in dept.get("adhoc_faculty", []):
            if isinstance(item, dict):
                add_person(item, dept_name, "adhoc")

    ece = website_data.get("ece_department", {})
    ece_name = str(ece.get("name", "Electronics and Communication Engineering")).strip() or "ECE"

    hod = ece.get("hod")
    if isinstance(hod, dict):
        add_person(hod, ece_name, "hod")

    for item in ece.get("regular_faculty", []):
        if isinstance(item, dict):
            add_person(item, ece_name, "regular")

    for item in ece.get("support_staff", []):
        if isinstance(item, dict):
            add_person(item, f"{ece_name} (Support)", "support")

    dedup: dict[str, dict[str, str]] = {}
    for row in rows:
        key = _normalize(row["name"])
        if key and key not in dedup:
            dedup[key] = row
    return list(dedup.values())


def _build_doc_chunks(website_data: dict[str, Any]) -> list[tuple[str, str]]:
    chunks: list[tuple[str, str]] = []

    college = website_data.get("college", {})
    principal = website_data.get("principal", {})
    contact = website_data.get("contact", {})
    ece = website_data.get("ece_department", {})

    chunks.append(
        (
            "college_overview",
            " | ".join(
                part
                for part in [
                    f"College: {college.get('name', '')}",
                    f"Short Name: {college.get('short_name', '')}",
                    f"Location: {college.get('location', '')}",
                    f"Website: {college.get('website', '')}",
                    f"Established: {college.get('established', '')}",
                ]
                if part.strip() and not part.endswith(": ")
            ),
        )
    )

    chunks.append(
        (
            "principal",
            " | ".join(
                part
                for part in [
                    f"Principal Name: {principal.get('name', '')}",
                    f"Principal Email: {principal.get('email', '')}",
                    f"Principal Phone: {principal.get('phone', '')}",
                ]
                if part.strip() and not part.endswith(": ")
            ),
        )
    )

    chunks.append(
        (
            "contact",
            " | ".join(
                part
                for part in [
                    f"Address: {contact.get('address', '')}",
                    f"Office Phone: {contact.get('office_phone', '')}",
                    f"College Email: {contact.get('college_email', '')}",
                    f"ECE HOD Email: {contact.get('ece_hod_email', '')}",
                ]
                if part.strip() and not part.endswith(": ")
            ),
        )
    )

    chunks.append(("ece_department", f"ECE Department Name: {ece.get('name', '')}"))
    if ece.get("vision"):
        chunks.append(("ece_vision", f"Vision: {ece.get('vision', '')}"))
    if ece.get("mission"):
        chunks.append(("ece_mission", f"Mission: {ece.get('mission', '')}"))

    programs = ece.get("programs", {})
    for idx, text in enumerate(programs.get("educational_objectives", []), start=1):
        if text:
            chunks.append(("ece_objectives", f"Objective {idx}: {text}"))
    for idx, text in enumerate(programs.get("specific_outcomes", []), start=1):
        if text:
            chunks.append(("ece_outcomes", f"Outcome {idx}: {text}"))

    departments = website_data.get("departments", [])
    for dept in departments:
        name = str(dept.get("name", "")).strip()
        if not name:
            continue
        hod = dept.get("hod", {})
        hod_name = ""
        if isinstance(hod, dict):
            hod_name = str(hod.get("name", "")).strip()
        summary = f"Department: {name}"
        if hod_name:
            summary += f" | HOD: {hod_name}"
        chunks.append(("department", summary))

    return [(section, content) for section, content in chunks if content.strip()]


def _upsert_fact(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO facts(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )


def _rebuild_facts(conn: sqlite3.Connection, website_data: dict[str, Any]) -> None:
    conn.execute("DELETE FROM facts")

    college = website_data.get("college", {})
    principal = website_data.get("principal", {})
    contact = website_data.get("contact", {})
    ece = website_data.get("ece_department", {})
    hod = ece.get("hod", {}) if isinstance(ece.get("hod"), dict) else {}

    facts = {
        "college_name": str(college.get("name", "")).strip(),
        "college_short_name": str(college.get("short_name", "")).strip(),
        "college_location": str(college.get("location", "")).strip(),
        "college_website": str(college.get("website", "")).strip(),
        "principal_name": str(principal.get("name", "")).strip(),
        "principal_designation": str(principal.get("designation", "")).strip(),
        "principal_email": str(principal.get("email", "")).strip(),
        "principal_phone": str(principal.get("phone", "")).strip(),
        "contact_address": str(contact.get("address", "")).strip(),
        "contact_office_phone": str(contact.get("office_phone", "")).strip(),
        "contact_college_email": str(contact.get("college_email", "")).strip(),
        "contact_ece_hod_email": str(contact.get("ece_hod_email", "")).strip(),
        "ece_name": str(ece.get("name", "")).strip(),
        "ece_code": str(ece.get("code", "")).strip(),
        "ece_website": str(ece.get("website", "")).strip(),
        "ece_hod_name": str(hod.get("name", "")).strip(),
        "ece_hod_email": str(hod.get("email", "")).strip(),
    }

    for key, value in facts.items():
        _upsert_fact(conn, key, value)


def _rebuild_database(conn: sqlite3.Connection, website_data: dict[str, Any], faq_data: list[dict[str, Any]]) -> None:
    conn.executescript(
        """
        DELETE FROM faculty_alias;
        DELETE FROM faculty;
        DELETE FROM faq;
        DELETE FROM doc_chunks;
        """
    )

    faculty_rows = _collect_faculty_rows(website_data)
    for row in faculty_rows:
        first_name, last_name, normalized_name = _name_parts(row["name"])
        has_dr = 1 if normalized_name.startswith("dr ") else 0

        cur = conn.execute(
            """
            INSERT INTO faculty(
                name, normalized_name, first_name, last_name, designation,
                discipline, email, phone, department, role_group, has_dr
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["name"],
                normalized_name,
                first_name,
                last_name,
                row["designation"],
                row["discipline"],
                row["email"],
                row["phone"],
                row["department"],
                row["role_group"],
                has_dr,
            ),
        )
        faculty_id = int(cur.lastrowid)

        aliases = {normalized_name}
        stripped_tokens = [t for t in normalized_name.split() if t not in {"dr", "prof", "professor"}]
        stripped_name = " ".join(stripped_tokens)
        if stripped_name:
            aliases.add(stripped_name)
        if first_name:
            aliases.add(first_name)
        if last_name:
            aliases.add(last_name)

        for alias in aliases:
            conn.execute(
                "INSERT OR IGNORE INTO faculty_alias(faculty_id, alias) VALUES (?, ?)",
                (faculty_id, alias),
            )

    for item in faq_data:
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        if not question or not answer:
            continue
        conn.execute(
            "INSERT INTO faq(question, answer, normalized_question) VALUES (?, ?, ?)",
            (question, answer, _normalize(question)),
        )

    for section, content in _build_doc_chunks(website_data):
        conn.execute(
            "INSERT INTO doc_chunks(section, content, normalized_content) VALUES (?, ?, ?)",
            (section, content, _normalize(content)),
        )

    _rebuild_facts(conn, website_data)

    if _FTS_ENABLED:
        conn.executescript(
            """
            DELETE FROM faq_fts;
            INSERT INTO faq_fts(rowid, question, answer)
            SELECT id, question, answer FROM faq;

            DELETE FROM docs_fts;
            INSERT INTO docs_fts(rowid, section, content)
            SELECT id, section, content FROM doc_chunks;
            """
        )


def ensure_knowledge_base(force_refresh: bool = False) -> None:
    global _DB_READY

    if _DB_READY and not force_refresh:
        return

    with _DB_LOCK:
        if _DB_READY and not force_refresh:
            return

        KMS_DIR.mkdir(parents=True, exist_ok=True)
        conn = _connect()
        try:
            _create_schema(conn)

            faculty_count = int(conn.execute("SELECT COUNT(*) AS c FROM faculty").fetchone()["c"])
            faq_count = int(conn.execute("SELECT COUNT(*) AS c FROM faq").fetchone()["c"])
            facts_count = int(conn.execute("SELECT COUNT(*) AS c FROM facts").fetchone()["c"])

            rebuild_needed = force_refresh

            if not rebuild_needed:
                # DB-first runtime mode: if DB already has usable data, avoid JSON reads.
                if faculty_count > 0 and faq_count > 0 and facts_count > 0:
                    conn.commit()
                    _DB_READY = True
                    return

                # One-time migration for older DBs missing facts.
                if faculty_count > 0 and faq_count > 0 and facts_count == 0 and WEBSITE_CACHE_FILE.exists() and FAQ_FILE.exists():
                    rebuild_needed = True

            if not rebuild_needed and not RUNTIME_DB_ONLY:
                website_mtime = _get_file_mtime(WEBSITE_CACHE_FILE)
                faq_mtime = _get_file_mtime(FAQ_FILE)
                stored_website_mtime = int(_meta_get(conn, "website_mtime") or "0")
                stored_faq_mtime = int(_meta_get(conn, "faq_mtime") or "0")
                rebuild_needed = (
                    faculty_count == 0
                    or faq_count == 0
                    or facts_count == 0
                    or website_mtime != stored_website_mtime
                    or faq_mtime != stored_faq_mtime
                )

            if not rebuild_needed and (faculty_count == 0 or faq_count == 0):
                rebuild_needed = True

            if rebuild_needed:
                website_data = _load_json(WEBSITE_CACHE_FILE, dict)
                faq_data = _load_json(FAQ_FILE, list)
                _rebuild_database(conn, website_data, faq_data)
                website_mtime = _get_file_mtime(WEBSITE_CACHE_FILE)
                faq_mtime = _get_file_mtime(FAQ_FILE)
                _meta_set(conn, "website_mtime", str(website_mtime))
                _meta_set(conn, "faq_mtime", str(faq_mtime))
                _meta_set(conn, "rebuilt_at", str(int(time.time())))
                logger.info("Knowledge base rebuilt: %s", DB_FILE)

            conn.commit()
            _DB_READY = True
        finally:
            conn.close()


def _fetch_faculty_by_mode(conn: sqlite3.Connection, mode: str, limit: int) -> list[sqlite3.Row]:
    mode = mode.lower().strip()

    if mode == "dr":
        return conn.execute(
            "SELECT * FROM faculty WHERE has_dr = 1 ORDER BY name LIMIT ?",
            (limit,),
        ).fetchall()

    if mode in {"associate professor", "assistant professor", "instructor", "professor"}:
        return conn.execute(
            "SELECT * FROM faculty WHERE lower(designation) LIKE ? ORDER BY name LIMIT ?",
            (f"%{mode}%", limit),
        ).fetchall()

    return conn.execute("SELECT * FROM faculty ORDER BY name LIMIT ?", (limit,)).fetchall()


def list_faculty(mode: str = "all", limit: int = 50) -> list[dict[str, Any]]:
    ensure_knowledge_base()
    conn = _connect()
    try:
        rows = _fetch_faculty_by_mode(conn, mode=mode, limit=limit)
        return [dict(row) for row in rows]
    finally:
        conn.close()


def _row_to_faculty_result(row: sqlite3.Row, score: float) -> dict[str, Any]:
    return {
        "name": row["name"],
        "designation": row["designation"] or "",
        "discipline": row["discipline"] or "",
        "email": row["email"] or "",
        "phone": row["phone"] or "",
        "department": row["department"] or "",
        "score": round(score, 3),
    }


def search_faculty(query: str, limit: int = 5) -> list[dict[str, Any]]:
    ensure_knowledge_base()
    conn = _connect()

    normalized_query = _normalize(query)
    tokens = _tokenize(query)
    aliases_to_try = [normalized_query] + tokens

    candidates: dict[int, dict[str, Any]] = {}

    try:
        for alias in aliases_to_try:
            if not alias:
                continue
            rows = conn.execute(
                """
                SELECT f.*
                FROM faculty_alias a
                JOIN faculty f ON f.id = a.faculty_id
                WHERE a.alias = ?
                """,
                (alias,),
            ).fetchall()
            for row in rows:
                row_id = int(row["id"])
                score = 1.0 if alias == normalized_query else 0.95
                existing = candidates.get(row_id)
                if existing is None or score > existing["score"]:
                    candidates[row_id] = _row_to_faculty_result(row, score)

        if tokens:
            for token in tokens:
                rows = conn.execute(
                    """
                    SELECT * FROM faculty
                    WHERE first_name LIKE ? OR last_name LIKE ?
                    """,
                    (f"{token}%", f"{token}%"),
                ).fetchall()
                for row in rows:
                    row_id = int(row["id"])
                    score = 0.88
                    existing = candidates.get(row_id)
                    if existing is None or score > existing["score"]:
                        candidates[row_id] = _row_to_faculty_result(row, score)

        # Accuracy-oriented fuzzy fallback only if deterministic lookup was weak.
        top_score = max((item["score"] for item in candidates.values()), default=0.0)
        if top_score < 0.97:
            all_rows = conn.execute("SELECT * FROM faculty").fetchall()
            query_set = set(tokens)

            for row in all_rows:
                name_norm = row["normalized_name"] or ""
                sim = SequenceMatcher(None, normalized_query, name_norm).ratio()

                if query_set:
                    name_tokens = set(name_norm.split())
                    overlap = len(query_set & name_tokens) / len(query_set)
                else:
                    overlap = 0.0

                score = 0.65 * sim + 0.35 * overlap

                if any(t and t in name_norm for t in tokens):
                    score = max(score, 0.82)

                if score >= 0.76:
                    row_id = int(row["id"])
                    existing = candidates.get(row_id)
                    if existing is None or score > existing["score"]:
                        candidates[row_id] = _row_to_faculty_result(row, score)

        sorted_hits = sorted(candidates.values(), key=lambda x: x["score"], reverse=True)
        return sorted_hits[:limit]
    finally:
        conn.close()


def _build_fts_query(tokens: list[str]) -> str:
    safe_tokens = [re.sub(r"[^a-z0-9]", "", tok.lower()) for tok in tokens]
    safe_tokens = [tok for tok in safe_tokens if tok]
    return " OR ".join(safe_tokens)


def search_docs(query: str, limit: int = 4) -> list[dict[str, Any]]:
    ensure_knowledge_base()
    conn = _connect()

    normalized_query = _normalize(query)
    tokens = _tokenize(query)

    try:
        rows: list[sqlite3.Row]
        if _FTS_ENABLED and tokens:
            fts_query = _build_fts_query(tokens)
            rows = conn.execute(
                """
                SELECT d.*
                FROM docs_fts f
                JOIN doc_chunks d ON d.id = f.rowid
                WHERE docs_fts MATCH ?
                LIMIT ?
                """,
                (fts_query, limit * 3),
            ).fetchall()
        else:
            like_term = "%" + "%".join(tokens) + "%" if tokens else f"%{normalized_query}%"
            rows = conn.execute(
                "SELECT * FROM doc_chunks WHERE normalized_content LIKE ? LIMIT ?",
                (like_term, limit * 3),
            ).fetchall()

        scored: list[dict[str, Any]] = []
        query_set = set(tokens)
        for row in rows:
            content_norm = row["normalized_content"]
            sim = SequenceMatcher(None, normalized_query, content_norm).ratio()
            if query_set:
                content_tokens = set(content_norm.split())
                overlap = len(query_set & content_tokens) / len(query_set)
            else:
                overlap = 0.0

            score = 0.55 * sim + 0.45 * overlap
            if score < 0.40:
                continue
            scored.append({
                "section": row["section"],
                "content": row["content"],
                "score": round(score, 3),
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]
    finally:
        conn.close()


def answer_faq(query: str) -> tuple[str | None, float]:
    ensure_knowledge_base()
    conn = _connect()
    normalized_query = _normalize(query)
    tokens = _tokenize(query)

    try:
        direct_patterns = [
            "how many students",
            "digital lab",
            "circuits lab",
            "communication lab",
            "maintenance cell",
            "electronics workshop",
            "lecture hall 1",
            "professors room",
            "systems lab",
            "library",
            "computer lab",
            "hod cabin",
            "staff room",
            "seminar hall",
            "project lab",
            "robotics lab",
            "pg lab",
            "casp lab",
            "project coordinator",
        ]

        for pattern in direct_patterns:
            if pattern in normalized_query:
                row = conn.execute(
                    "SELECT answer FROM faq WHERE normalized_question LIKE ? LIMIT 1",
                    (f"%{pattern}%",),
                ).fetchone()
                if row:
                    return str(row["answer"]), 1.0

        rows: list[sqlite3.Row]
        if _FTS_ENABLED and tokens:
            fts_query = _build_fts_query(tokens)
            rows = conn.execute(
                """
                SELECT q.*
                FROM faq_fts f
                JOIN faq q ON q.id = f.rowid
                WHERE faq_fts MATCH ?
                LIMIT 15
                """,
                (fts_query,),
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM faq").fetchall()

        best_answer: str | None = None
        best_score = 0.0

        query_set = set(tokens)
        for row in rows:
            question_norm = row["normalized_question"]
            sim = SequenceMatcher(None, normalized_query, question_norm).ratio()

            if query_set:
                q_tokens = set(question_norm.split())
                overlap = len(query_set & q_tokens) / len(query_set)
            else:
                overlap = 0.0

            score = 0.55 * sim + 0.45 * overlap

            if "where" in normalized_query and "where" in question_norm:
                score += 0.05
            if "who" in normalized_query and "who" in question_norm:
                score += 0.05

            if score > best_score:
                best_score = score
                best_answer = str(row["answer"])

        if best_answer and best_score >= 0.62:
            return best_answer, round(best_score, 3)

        return None, round(best_score, 3)
    finally:
        conn.close()


def get_context_snapshot() -> dict[str, Any]:
    """Return compact college context from SQLite facts table."""
    ensure_knowledge_base()
    conn = _connect()
    try:
        rows = conn.execute("SELECT key, value FROM facts").fetchall()
        facts = {str(r["key"]): str(r["value"]) for r in rows}

        return {
            "college": {
                "name": facts.get("college_name", ""),
                "short_name": facts.get("college_short_name", ""),
                "location": facts.get("college_location", ""),
                "website": facts.get("college_website", ""),
            },
            "principal": {
                "name": facts.get("principal_name", ""),
                "designation": facts.get("principal_designation", ""),
                "email": facts.get("principal_email", ""),
                "phone": facts.get("principal_phone", ""),
            },
            "contact": {
                "address": facts.get("contact_address", ""),
                "office_phone": facts.get("contact_office_phone", ""),
                "college_email": facts.get("contact_college_email", ""),
                "ece_hod_email": facts.get("contact_ece_hod_email", ""),
            },
            "ece_department": {
                "name": facts.get("ece_name", ""),
                "code": facts.get("ece_code", ""),
                "website": facts.get("ece_website", ""),
                "hod": {
                    "name": facts.get("ece_hod_name", ""),
                    "email": facts.get("ece_hod_email", ""),
                },
            },
        }
    finally:
        conn.close()


def refresh_knowledge_base() -> None:
    """Force rebuild SQLite data from source files."""
    ensure_knowledge_base(force_refresh=True)
