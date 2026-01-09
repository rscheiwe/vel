# Vel Recipes

Multi-step workflows for common development tasks. Each recipe is a complete procedure.

---

## Available Recipes

| Recipe | Description |
|--------|-------------|
| [add-provider.md](add-provider.md) | Add support for a new LLM provider |
| [add-memory-feature.md](add-memory-feature.md) | Add a new memory system feature |
| [debug-streaming.md](debug-streaming.md) | Debug streaming issues |
| [setup-auto-learning.md](setup-auto-learning.md) | Configure auto-learning pipeline |

---

## Recipe Structure

Each recipe follows this format:

```markdown
# Recipe Name

**Goal:** What this achieves
**Prerequisites:** What you need before starting
**Time Estimate:** Rough duration

---

## Steps

### Step 1: [Action]
[Instructions]

### Step 2: [Action]
[Instructions]

---

## Validation

How to verify success.

---

## Troubleshooting

Common issues and solutions.
```

---

## When to Use Recipes

| Situation | Use Recipe |
|-----------|------------|
| Adding new provider | `add-provider.md` |
| Memory feature development | `add-memory-feature.md` |
| Streaming bugs | `debug-streaming.md` |
| Production learning setup | `setup-auto-learning.md` |

---

## Creating New Recipes

When a multi-step workflow is repeated more than twice:

1. Create `docs/recipes/{workflow-name}.md`
2. Follow the structure above
3. Add to the table in this README
4. Update CLAUDE.md references if needed
