class GraphRecordBuilderTool:
    def __init__(self, builder):
        self._builder = builder

    def build(self, parsed: dict) -> list[dict]:
        if not parsed.get("ok") or not parsed.get("recipe_info"):
            return []
        rel_path = parsed["rel_path"]
        self._builder.process_recipe_info(parsed["recipe_info"], rel_path)
        self._builder.processed_files.add(rel_path)
        return [{"rel_path": rel_path, "ok": True}]

