"""RagZen CLI interface.

Command-line tool for managing RagZen instances, running queries,
ingesting documents, checking health, and running the API server.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from ragzen import RagZen, SecurityContext, __version__
from ragzen.config import RagZenConfig, validate_config


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ragzen",
        description="RagZen — Enterprise-grade, local-first RAG for Python",
    )
    parser.add_argument("--version", action="version", version=f"RagZen {__version__}")
    parser.add_argument("--config", "-c", help="Path to ragzen.yaml config file")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # init
    p_init = subparsers.add_parser("init", help="Initialize a new RagZen directory")
    p_init.add_argument("--path", default=".ragzen", help="Directory path to initialize")

    # config validate
    p_config = subparsers.add_parser("config", help="Configuration commands")
    p_config.add_argument("action", choices=["validate"], help="Action to perform")

    # ingest
    p_ingest = subparsers.add_parser("ingest", help="Ingest a document or directory")
    p_ingest.add_argument("path", help="Path to file or directory")
    p_ingest.add_argument("--tenant", default="default", help="Tenant ID")
    p_ingest.add_argument("--department", default="", help="Department")
    p_ingest.add_argument("--access-level", default="internal", help="Access level")
    p_ingest.add_argument("--idempotency-key", default="", help="Idempotency key")

    # query
    p_query = subparsers.add_parser("query", help="Ask a question using RAG")
    p_query.add_argument("question", help="Question text")
    p_query.add_argument("--tenant", default="default", help="Tenant ID")
    p_query.add_argument("--user", default="cli-user", help="User ID")

    # search
    p_search = subparsers.add_parser("search", help="Search documents")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("--tenant", default="default", help="Tenant ID")
    p_search.add_argument("--top-k", type=int, default=5, help="Top K results")

    # delete
    p_delete = subparsers.add_parser("delete", help="Delete a document")
    p_delete.add_argument("document_id", help="Document ID to delete")
    p_delete.add_argument("--tenant", default="default", help="Tenant ID")

    # stats
    subparsers.add_parser("stats", help="Display instance statistics")

    # health
    subparsers.add_parser("health", help="Run health check")

    # doctor
    subparsers.add_parser("doctor", help="Run diagnostic health checks")

    # migrate
    p_migrate = subparsers.add_parser("migrate", help="Database migration tool")
    p_migrate.add_argument("action", choices=["plan", "apply", "status"], help="Migration action")

    # backup
    p_backup = subparsers.add_parser("backup", help="Create database backup")
    p_backup.add_argument("dest", help="Backup destination path")
    p_backup.add_argument("--no-compress", action="store_true", help="Disable gzip compression")

    # restore
    p_restore = subparsers.add_parser("restore", help="Restore database from backup")
    p_restore.add_argument("source", help="Backup file path")

    # serve
    p_serve = subparsers.add_parser("serve", help="Start FastAPI server")
    p_serve.add_argument("--host", default="127.0.0.1", help="Host address")
    p_serve.add_argument("--port", type=int, default=8000, help="Port number")

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    # Execute subcommands
    if args.command == "init":
        p = Path(args.path)
        p.mkdir(parents=True, exist_ok=True)
        print(f"Initialized RagZen directory at {p.resolve()}")
        return 0

    if args.command == "config" and args.action == "validate":
        cfg = RagZenConfig.from_yaml(args.config) if args.config else RagZenConfig.local_default()
        warnings = validate_config(cfg)
        if warnings:
            print("Configuration Warnings:")
            for w in warnings:
                print(f"  - {w}")
        else:
            print("Configuration is valid.")
        return 0

    engine = RagZen.from_config(args.config) if args.config else RagZen.local()

    if args.command == "ingest":
        job = engine.add(
            args.path,
            metadata={
                "tenant_id": args.tenant,
                "department": args.department,
                "access_level": args.access_level,
            },
            idempotency_key=args.idempotency_key,
        )
        print(json.dumps(job.model_dump(mode="json"), indent=2))
        return 0

    if args.command == "query":
        ctx = SecurityContext(tenant_id=args.tenant, user_id=args.user)
        resp = engine.ask(args.question, security_context=ctx)
        print("\nAnswer:")
        print(resp.answer)
        print("\nSources:")
        for s in resp.sources:
            print(f"  - {s.file_name} (page {s.page}, score: {s.score:.4f})")
        return 0

    if args.command == "search":
        ctx = SecurityContext(tenant_id=args.tenant)
        results = engine.search(args.query, top_k=args.top_k, security_context=ctx)
        print(json.dumps([r.model_dump(mode="json") for r in results], indent=2))
        return 0

    if args.command == "delete":
        deleted = engine.delete(args.document_id, tenant_id=args.tenant)
        if deleted:
            print(f"Deleted document {args.document_id}")
            return 0
        print(f"Document {args.document_id} not found")
        return 1

    if args.command == "stats":
        print(json.dumps(engine.stats(), indent=2))
        return 0

    if args.command == "health" or args.command == "doctor":
        health = engine.health(deep=args.command == "doctor")
        print(json.dumps(health.model_dump(mode="json"), indent=2))
        return 0 if health.healthy else 1

    if args.command == "backup":
        out = engine.backup(args.dest, compress=not args.no_compress)
        print(f"Created backup at {out}")
        return 0

    if args.command == "restore":
        engine.restore(args.source)
        print(f"Restored database from {args.source}")
        return 0

    if args.command == "migrate":
        res = engine.migrate(args.action)
        print(json.dumps(res, indent=2))
        return 0

    if args.command == "serve":
        try:
            import uvicorn

            from ragzen.server.app import create_app
        except ImportError:
            print("Error: FastAPI and Uvicorn are required to run server mode.")
            print('Install with: pip install "ragzen[server]"')
            return 1

        app = create_app(engine)
        uvicorn.run(app, host=args.host, port=args.port)
        return 0

    return 0


def app() -> None:
    """Entry point for project.scripts."""
    sys.exit(main())


if __name__ == "__main__":
    app()
