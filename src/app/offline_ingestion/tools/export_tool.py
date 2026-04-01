class ExporterTool:
    def __init__(self, builder):
        self._builder = builder

    def export_csv(self, records: list[dict], output_dir: str, output_format: str = "neo4j") -> None:
        if output_format == "neo4j":
            self._builder.export_to_neo4j_csv(output_dir, True)
        else:
            self._builder.merge_all_batches()

