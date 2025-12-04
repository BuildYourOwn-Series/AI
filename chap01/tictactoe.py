#!/usr/bin/env python3
# tictactoe.py
# A complete, executable Tic-Tac-Toe game: Human vs Computer
# Standard library only; Python >=3.9 recommended.

from __future__ import annotations
from typing import List, Optional, Iterable, Tuple
import random
import sys

Board = List[str]  # 9 cells: 'X', 'O', or ' '

def winner(board: Board) -> Optional[str]:
    wins: Tuple[Tuple[int, int, int], ...] = (
        (0,1,2),(3,4,5),(6,7,8),
        (0,3,6),(1,4,7),(2,5,8),
        (0,4,8),(2,4,6)
    )
    for a,b,c in wins:
        if board[a] == board[b] == board[c] != ' ':
            return board[a]
    return None

def minimax(board: Board, turn: str) -> int:
    w = winner(board)
    if w == 'X': return +1
    if w == 'O': return -1
    if ' ' not in board: return 0

    next_turn = 'O' if turn == 'X' else 'X'
    scores: List[int] = []
    for i, cell in enumerate(board):
        if cell == ' ':
            board[i] = turn
            score = minimax(board, next_turn)
            board[i] = ' '
            scores.append(score)
    return max(scores) if turn == 'X' else min(scores)

def best_move_perfect(board: Board, turn: str) -> int:
    # Returns the index (0..8) of an optimal move for "turn".
    best_score = -2 if turn == 'X' else +2
    move = -1
    for i, cell in enumerate(board):
        if cell == ' ':
            board[i] = turn
            score = minimax(board, 'O' if turn == 'X' else 'X')
            board[i] = ' '
            if (turn == 'X' and score > best_score) or (turn == 'O' and score < best_score):
                best_score, move = score, i
    return move

def best_move_smart(board: Board, turn: str, epsilon: float = 0.15) -> int:
    # ε-greedy: usually optimal, sometimes explores a random legal move.
    legal = [i for i, c in enumerate(board) if c == ' ']
    if not legal: return -1
    if random.random() < epsilon:
        return random.choice(legal)
    return best_move_perfect(board, turn)

def best_move_novice(board: Board, turn: str) -> int:
    # Favor center, then corners, else random edge; block/win if immediate.
    legal = [i for i, c in enumerate(board) if c == ' ']
    if not legal: return -1

    # 1) If we can win now, do it.
    for i in legal:
        board[i] = turn
        if winner(board) == turn:
            board[i] = ' '
            return i
        board[i] = ' '

    # 2) If opponent can win next, block.
    opp = 'O' if turn == 'X' else 'X'
    for i in legal:
        board[i] = opp
        if winner(board) == opp:
            board[i] = ' '
            return i
        board[i] = ' '

    # 3) Preferences
    prefs = [4, 0, 2, 6, 8, 1, 3, 5, 7]  # center, corners, then edges
    for i in prefs:
        if i in legal:
            return i
    return random.choice(legal)

def render(board: Board) -> str:
    # Pretty board with 1-9 hints for empty squares.
    def cell(i: int) -> str:
        return board[i] if board[i] != ' ' else str(i+1)
    rows = [
        f" {cell(0)} │ {cell(1)} │ {cell(2)} ",
        "───┼───┼───",
        f" {cell(3)} │ {cell(4)} │ {cell(5)} ",
        "───┼───┼───",
        f" {cell(6)} │ {cell(7)} │ {cell(8)} ",
    ]
    return "\n".join(rows)

def game_over(board: Board) -> Optional[str]:
    w = winner(board)
    if w: return f"{w} wins!"
    if ' ' not in board: return "It's a draw."
    return None

def ask_int(prompt: str, valid: Iterable[int]) -> Optional[int]:
    valid_set = set(valid)
    while True:
        s = input(prompt).strip().lower()
        if s in {"q", "quit", "exit"}:
            return None
        if s.isdigit():
            k = int(s)
            if k in valid_set:
                return k
        print("  → Please enter a free square number or 'q' to quit.")

def choose(mark_prompt: str, options: Iterable[str]) -> str:
    opts = {o.lower() for o in options}
    while True:
        s = input(mark_prompt).strip().lower()
        if s in opts:
            return s
        print(f"  → Choose one of: {', '.join(sorted(opts))}")

def choose_move_fn(level: str):
    if level == "perfect":
        return best_move_perfect
    if level == "smart":
        return lambda b, t: best_move_smart(b, t, epsilon=0.15)
    if level == "novice":
        return best_move_novice
    raise ValueError("unknown level")

def play_once() -> None:
    print("\n=== Tic-Tac-Toe: Human vs Computer ===")
    level = choose("Difficulty [perfect/smart/novice]: ", {"perfect","smart","novice"})
    human_as = choose("Play as [X/O]? ", {"x","o"}).upper()
    human = "X" if human_as == "X" else "O"
    comp  = "O" if human == "X" else "X"
    move_fn = choose_move_fn(level)

    board: Board = [' '] * 9
    turn = "X"  # X always starts
    print("\nEnter a number 1–9 to place your mark; 'q' to quit.\n")

    while True:
        print(render(board))
        over = game_over(board)
        if over:
            print("\n" + over + "\n")
            break

        if turn == human:
            free = [i+1 for i,c in enumerate(board) if c == ' ']
            k = ask_int(f"\nYour move ({human}) -> ", free)
            if k is None:
                print("Goodbye!")
                sys.exit(0)
            i = k - 1
            board[i] = human
        else:
            i = move_fn(board, comp)
            board[i] = comp
            print(f"\nComputer plays {comp} at {i+1}.\n")

        turn = "O" if turn == "X" else "X"

def main() -> None:
    random.seed()  # for 'smart'/'novice' exploration
    while True:
        play_once()
        again = choose("Play again? [y/n]: ", {"y","n"})
        if again == "n":
            break
    print("Thanks for playing!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Bye!")

