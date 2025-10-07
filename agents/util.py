import orjson
def json_dumps(obj) -> str:
    return orjson.dumps(obj).decode()
