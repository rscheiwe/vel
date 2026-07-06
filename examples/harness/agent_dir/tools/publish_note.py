from vel import ToolSpec


async def publish_note(note_id: str = "", ctx: dict = None) -> dict:
    # Consequential — gated via [harness.approval].require_for_tools in agent.toml.
    return {"published": note_id}


tool = ToolSpec.from_function(publish_note)
