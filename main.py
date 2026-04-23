import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="RAG Customer Support Assistant"
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Run the document ingestion pipeline before starting Q&A.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Run a single query and exit (non-interactive mode).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Optional ingestion 
    if args.ingest:
        from src.ingest import run_ingestion
        print("\nRunning document ingestion …\n")
        run_ingestion()
        print("\nIngestion complete. Chunks stored in ChromaDB.\n")

    # Load vectorstore and build graph
    from src.retriever import load_vectorstore
    from src.graph import build_graph, run_query

    print("⚙️   Loading vectorstore …")
    vectorstore = load_vectorstore()
    print("⚙️   Compiling LangGraph …")
    graph = build_graph()
    print("System ready.\n")

    # Single-shot mode 
    if args.query:
        _answer_query(args.query, vectorstore, graph)
        return

    # Interactive loop
    print("=" * 60)
    print("  RAG Customer Support Assistant  (type 'exit' to quit)")
    print("=" * 60)

    while True:
        try:
            query = input("\n🔍  Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break

        _answer_query(query, vectorstore, graph)


def _answer_query(query: str, vectorstore, graph) -> None:
    from src.graph import run_query
    print()
    state = run_query(query, vectorstore, graph)
    if state["escalate"]:
        print("[ESCALATED]. Please contact our human agent for further details")
    else:
        print("[ANSWERED]")
    print(f"\n{state['response']}\n")
    print("-" * 60)


if __name__ == "__main__":
    main()
