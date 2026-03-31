import re


def rewrite_query(query: str, history: list):
    query = query.lower().strip()

    filler = [
        "give", "explain", "tell", "describe",
        "what is", "what are", "please", "define"
    ]

    for f in filler:
        query = query.replace(f, "")

    query = re.sub(r"\s+", " ", query).strip()

    if history and len(query.split()) <= 3:
        return f"{history[-1]['content'].lower()} {query}"

    return query
