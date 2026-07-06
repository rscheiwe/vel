"""One tool per file. The module exports `tool` (a ToolSpec)."""
from vel import ToolSpec


async def search_notes(query: str = "", ctx: dict = None) -> dict:
    # Toy implementation; a real tool would hit a store.
    return {"query": query, "results": [f"note about {query}"]}


tool = ToolSpec.from_function(search_notes)
