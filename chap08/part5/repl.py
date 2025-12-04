#!/usr/bin/env python3
# chap08/repl.py

from __future__ import annotations
import shlex
import subprocess

import readline  # NEW: to capture Tab

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


def print_templates(store: KnowledgeStore, query: str) -> None:
    results = store.query(query, k=3)
    if not results:
        print("[ai] No templates found.")
        return

    print("[ai] Templates:")
    for e in results:
        print(f"  $ {e.command}")
    print()


def print_flag_suggestions(store: KnowledgeStore, cmd: str) -> None:
    results = store.query(cmd, k=5)
    flags: list[str] = []

    for e in results:
        flags.extend(extract_flags(e.command))

    if not flags:
        print(f"[ai] No flag suggestions for {cmd}.")
        return

    freq: dict[str, int] = {}
    for fl in flags:
        freq[fl] = freq.get(fl, 0) + 1

    ranked = sorted(freq.items(), key=lambda t: t[1], reverse=True)[:5]

    print(f"[ai] Common flags for {cmd}:")
    for fl, count in ranked:
        print(f"  {fl}")
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


def make_tab_completer(store: KnowledgeStore):
    """
    Return a readline completer function that uses Tab to trigger
    flag suggestions for the first token on the current line.
    """
    def completer(text: str, state: int):
        # We only act on the first Tab press (state == 0).
        if state > 0:
            return None

        line = readline.get_line_buffer().strip()
        if not line:
            return None

        parts = line.split()
        cmd = parts[0]
        # Trigger suggestions only for bare commands (no spaces yet),
        # so typing "ls<Tab>" works, but "ls -<Tab>" is left alone.
        if len(parts) == 1:
            print()  # move to next line cleanly
            print_flag_suggestions(store, cmd)
            # Reprint the prompt + current buffer so it feels natural.
            # NOTE: the REPL prints "ai$ " itself; we just echo the line.
            print(f"ai$ {line}", end="", flush=True)
        return None  # no actual completion text
    return completer


def repl() -> None:
    backend, store = build_default_store()
    print("ai-shell (minimal) -- type '? help' for suggestions, '^ cmd' "
          "for explanations, 'exit' to quit.")

    # Hook Tab: use our completer instead of filename completion.
    readline.set_completer(make_tab_completer(store))
    readline.parse_and_bind("tab: complete")

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
            if not cmd:
                print("[ai] Try: ^ ls -la")
            else:
                print_explanation(store, cmd)
        elif line.startswith("!"):
            query = line[1:].strip()
            if not query:
                print("[ai] Enter a natural-language query after '!'.")
            else:
                print_templates(store, query)
        else:
            run_command(line)


if __name__ == "__main__":
    repl()

