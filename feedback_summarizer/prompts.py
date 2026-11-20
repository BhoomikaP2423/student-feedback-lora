def make_prompt(feedback: str) -> str:
    return (
        "Summarize the following student feedback for a course instructor. "
        "Be concise, factual, and constructive. Identify the main issue or strength "
        "and a practical curriculum improvement. Do not invent facts.\n"
        f"Student feedback: {feedback.strip()}\n"
        "Instructor summary:"
    )
