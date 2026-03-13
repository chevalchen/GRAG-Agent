import argparse

from src.app.offline_ingestion.graphs.ingestion_graph import OfflineIngestionGraph


def main(argv=None):
    parser = argparse.ArgumentParser(prog="offline_ingestion", description="LangGraph 离线 ingestion")
    parser.add_argument("recipe_dir", type=str)
    parser.add_argument("-o", "--output-dir", type=str, default="./ai_output")
    parser.add_argument("--output-format", type=str, default="neo4j", choices=["neo4j", "csv"])
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--parse-concurrency", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    graph = OfflineIngestionGraph(
        batch_size=args.batch_size,
        parse_concurrency=args.parse_concurrency,
    )
    graph.invoke(args.recipe_dir, args.output_dir, output_format=args.output_format, resume=args.resume)


if __name__ == "__main__":
    main()

