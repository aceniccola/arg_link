def stripjson(inp: str):
    if inp.startswith("```json") and inp.endswith("```"):
        return inp[len("```json"): -len("```")].strip()
    elif inp.startswith("```") and inp.endswith("```"):
        # Handle generic ``` case
        return inp[len("```"): -len("```")].strip()
    else:
        return inp