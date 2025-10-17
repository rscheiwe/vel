# run: python scripts/compare_traces.py traces/vel.jsonl traces/ai.jsonl
import sys, json
REQ_ORDER = ["start","text-start","text-delta","text-end","reasoning-start","reasoning-delta","reasoning-end",
             "tool-input-start","tool-input-delta","tool-input-available","tool-output-available",
             "response-metadata","source","file","start-step","finish-step","finish-message","finish","error"]

def load(path):
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]

def squash_deltas(events):
    """Return a copy where we concatenate consecutive text/think deltas for comparison."""
    out = []; buf = None
    for e in events:
        t = e.get("type")
        if t in ("text-delta","reasoning-delta"):
            if buf and buf["type"] == t and buf.get("id")==e.get("id"):
                buf["delta"] = (buf.get("delta","") + e.get("delta",""))
            else:
                if buf: out.append(buf)
                buf = {"type": t, "id": e.get("id"), "delta": e.get("delta","")}
        else:
            if buf: out.append(buf); buf=None
            out.append(e)
    if buf: out.append(buf)
    return out

def strip_optional(e):
    """Drop optional fields for ordering parity."""
    e = dict(e)
    # ignore metadata details, ids in finish, etc.
    if e.get("type") in ("response-metadata","source","file"): e["_drop"]=True
    return e

def normalize(seq):
    return [strip_optional(e) for e in squash_deltas(seq) if e.get("type") != "response-metadata"]

def compare(a, b):
    # Allow 'finish-message' to be optional; drop from either side to compare order-insensitively for that type.
    def drop_optional_finish(seq): return [e for e in seq if e.get("type") != "finish-message"]
    na, nb = normalize(a), normalize(b)
    fa, fb = drop_optional_finish(na), drop_optional_finish(nb)
    ok = True
    types_a = [e.get("type") for e in fa]
    types_b = [e.get("type") for e in fb]
    if types_a != types_b:
        print("❌ Event type order differs:\nA:", types_a, "\nB:", types_b); ok = False
    else:
        print("✅ Event type order matches")

    # Skip text/reasoning content comparison (LLM responses are non-deterministic)
    # Just verify the events exist
    def has_type(seq, t): return any(e.get("type") == t for e in seq)
    for t in ("text-delta", "reasoning-delta"):
        has_a = has_type(fa, t)
        has_b = has_type(fb, t)
        if has_a != has_b:
            print(f"❌ {t} presence mismatch: A={has_a}, B={has_b}"); ok = False

    # Verify finish-step and finish events have required fields
    for event_type in ("finish-step", "finish"):
        events_a = [e for e in fa if e.get("type") == event_type]
        events_b = [e for e in fb if e.get("type") == event_type]
        if len(events_a) != len(events_b):
            print(f"❌ {event_type} count mismatch: A={len(events_a)}, B={len(events_b)}"); ok = False
        elif events_a:
            # Check required fields exist (not comparing values, just structure)
            required_fields = ["finishReason"]
            if event_type == "finish-step":
                required_fields.extend(["usage", "response"])
            elif event_type == "finish":
                required_fields.append("totalUsage")

            for field in required_fields:
                has_a = field in events_a[0]
                has_b = field in events_b[0]
                if has_a != has_b:
                    print(f"❌ {event_type}.{field} presence mismatch: A={has_a}, B={has_b}"); ok = False
                elif has_a:
                    print(f"✅ {event_type}.{field} present in both")

    # Tool args/results shallow compare if present:
    def collect(seq, t, key): return [json.dumps({ "tool": e.get("toolName"), key: e.get(key) }, sort_keys=True)
                                      for e in seq if e.get("type")==t]
    if collect(fa,"tool-input-available","input") != collect(fb,"tool-input-available","input"):
        print("❌ tool-input-available args differ"); ok=False
    if collect(fa,"tool-output-available","output") != collect(fb,"tool-output-available","output"):
        print("❌ tool-output-available out differ"); ok=False
    return ok

def main():
    a = load(sys.argv[1]); b = load(sys.argv[2])
    ok = compare(a,b)
    print("✅ Parity" if ok else "❌ Parity failed")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
