#!/usr/bin/env python3
# chap08/repl.py.py

from __future__ import annotations
import shlex
import subprocess

from knowledge_store import KnowledgeStore, build_default_store


def print_suggestions(store: KnowledgeStore, query: str) -> None:
    results = store.query(query, k=3)
    if not results:
        print("[ai] No suggestions found.")
        return

    print("[ai] Suggestions:")
    for e in results:
        print(f"  - {e.description}")
        print(f"    $ {e.command}")
    print()


def run_command(line: str) -> None:
    try:
        args = shlex.split(line)
    except ValueError as e:
        print(f"[shell] Parse error: {e}")
        return

    if not args:
        return

    try:
        completed = subprocess.run(args)
        if completed.returncode != 0:
            print(f"[shell] Exit status: {completed.returncode}")
    except FileNotFoundError:
        print(f"[shell] Command not found: {args[0]}")
    except Exception as e:
        print(f"[shell] Error: {e}")


def repl() -> None:
    backend, store = build_default_store()
    print("ai-shell (minimal) -- type '? help' for suggestions,"
          " 'exit' to quit.")

    while True:
        try:
            line = input("ai$ ")
        except EOFError:
            print()
            break

        line = line.strip()
        if not line:
            continue
        if line in ("exit", "quit"):
            break

        if line.startswith("?"):
            query = line[1:].strip()
            if not query:
                print("[ai] Enter a natural-language query after '?'.")
            else:
                print_suggestions(store, query)
        else:
            run_command(line)


if __name__ == "__main__":
    repl()

