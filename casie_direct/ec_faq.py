from knowledge_store import answer_faq, ensure_knowledge_base


def answer_ec_faq(query: str) -> str:
    """Answer EC FAQ using SQLite retrieval with confidence thresholding."""
    ensure_knowledge_base()

    answer, score = answer_faq(query)
    if answer:
        return answer

    if score >= 0.5:
        return "I found related EC FAQ entries but I am not fully sure. Please ask in a more specific way."

    return "I don't have an exact answer for that EC department question yet."
