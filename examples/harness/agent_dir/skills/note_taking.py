"""One skill per file. The module exports `skill` (a Skill)."""
from vel.harness import Skill

skill = Skill(
    name="note-taking",
    description="Best practices for concise, well-tagged notes.",
    instructions=(
        "When saving a note, add 2-3 lowercase hashtags for retrieval and keep "
        "it to one sentence."
    ),
)
