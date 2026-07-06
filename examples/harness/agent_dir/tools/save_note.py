from vel import ToolSpec


async def save_note(text: str = "", ctx: dict = None) -> dict:
    return {"saved": text}


tool = ToolSpec.from_function(save_note)
