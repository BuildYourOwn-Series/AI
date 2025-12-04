#!/usr/bin/env python3
# chap08/repl.py.py

from __future__ import annotations
import shlex
import subprocess

from knowledge_store import KnowledgeStore, build_default_store, extract_flags


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

def print_explanation(store: KnowledgeStore, cmd: str) -> None:
    results = store.query(cmd, k=2)
    if not results:
        print("[ai] No explanation found.")
        return

    print("[ai] Explanation:")
    for e in results:
        print(f"  - {e.description}")
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

def print_flag_suggestions(store: KnowledgeStore, cmd: str) -> None:
    results = store.query(cmd, k=5)
    flags = []

    for e in results:
        flags.extend(extract_flags(e.command))

    if not flags:
        print("[ai] No flag suggestions available.")
        return

    # Rank by simple frequency
    freq = {}
    for fl in flags:
        freq[fl] = freq.get(fl, 0) + 1

    ranked = sorted(freq.items(), key=lambda t: t[1], reverse=True)[:5]

    print("[ai] Common flags:")
    for fl, count in ranked:
        print(f"  {fl}")
    print()

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
        elif line.startswith("^"):
            cmd = line[1:].strip()
            if cmd:
                print_explanation(store, cmd)
            else:
                print("[ai] Try: ^ ls -la")
        else:
            parts = line.split()
            if len(parts) == 1:
                # Bare command: try flag suggestions
                print_flag_suggestions(store, parts[0])
            run_command(line)


if __name__ == "__main__":
    repl()

