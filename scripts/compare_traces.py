# run: python scripts/compare_traces.py traces/vel.jsonl traces/ai.jsonl
import sys, json
REQ_ORDER = ["start","text-start","text-delta","text-end","reasoning-start","reasoning-delta","reasoning-end",
             "tool-input-start","tool-input-delta","tool-input-available","tool-output-available",
             "response-metadata","source","file","step-start","step-finish","finish-message","finish","error"]

def load(path):
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]

def squash_deltas(events):
    """Return a copy where we concatenate consecutive text/think deltas for comparison."""
    out = []; buf = None
    for e in events:
        if e["t"] in ("text-delta","reasoning-delta"):
            if buf and buf["t"] == e["t"] and buf.get("id")==e.get("id"):
                buf["d"] = (buf.get("d","") + e.get("d",""))
            else:
                if buf: out.append(buf)
                buf = {"t": e["t"], "id": e.get("id"), "d": e.get("d","")}
        else:
            if buf: out.append(buf); buf=None
            out.append(e)
    if buf: out.append(buf)
    return out

def strip_optional(e):
    """Drop optional fields for ordering parity."""
    e = dict(e)
    # ignore metadata details, ids in finish, etc.
    if e["t"] in ("response-metadata","source","file"): e["_drop"]=True
    return e

def normalize(seq):
    return [strip_optional(e) for e in squash_deltas(seq) if e.get("t") != "response-metadata"]

def compare(a, b):
    # Allow 'finish-message' to be optional; drop from either side to compare order-insensitively for that type.
    def drop_optional_finish(seq): return [e for e in seq if e["t"] != "finish-message"]
    na, nb = normalize(a), normalize(b)
    fa, fb = drop_optional_finish(na), drop_optional_finish(nb)
    ok = True
    if [e["t"] for e in fa] != [e["t"] for e in fb]:
        print("❌ Event type order differs:\nA:", [e["t"] for e in fa], "\nB:", [e["t"] for e in fb]); ok = False
    # Compare concatenated text/reasoning payloads:
    def acc_text(seq, t): return "".join(e.get("d","") for e in seq if e["t"]==t)
    for t in ("text-delta","reasoning-delta"):
        if acc_text(fa,t) != acc_text(fb,t):
            print(f"❌ {t} concatenated text differs"); ok = False
    # Tool args/results shallow compare if present:
    def collect(seq, t, key): return [json.dumps({ "tool": e.get("tool"), key: e.get(key) }, sort_keys=True)
                                      for e in seq if e["t"]==t]
    if collect(fa,"tool-input-available","args") != collect(fb,"tool-input-available","args"):
        print("❌ tool-input-available args differ"); ok=False
    if collect(fa,"tool-output-available","out") != collect(fb,"tool-output-available","out"):
        print("❌ tool-output-available out differ"); ok=False
    return ok

def main():
    a = load(sys.argv[1]); b = load(sys.argv[2])
    ok = compare(a,b)
    print("✅ Parity" if ok else "❌ Parity failed")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
