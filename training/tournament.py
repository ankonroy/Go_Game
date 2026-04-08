from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Make project root importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from game_models import GoBoard, Stone
from AI_Agent_Albi.montecarlo import MonteCarloAI as AlbiMonteCarlo
from AI_Agent_Ankon.montecarlo import MonteCarloAI as AnkonMonteCarlo
from AI_Agent_NN.neural_agent import NeuralNetAI


Move = Tuple[int, int]


def stone_to_name(stone: Stone) -> str:
    if stone == Stone.BLACK:
        return "black"
    if stone == Stone.WHITE:
        return "white"
    return "unknown"


def winner_to_int(winner: Optional[Stone]) -> int:
    if winner == Stone.BLACK:
        return 1
    if winner == Stone.WHITE:
        return 2
    return 0


def create_opponent(
    opponent_name: str,
    player: Stone,
    simulations: int,
    time_limit: float,
):
    if opponent_name == "albi_mcts":
        return AlbiMonteCarlo(
            player=player,
            simulations=simulations,
            time_limit=time_limit,
        )

    if opponent_name == "ankon_mcts":
        return AnkonMonteCarlo(
            player=player,
            simulations=simulations,
            time_limit=time_limit,
        )

    raise ValueError(f"Unknown opponent_name: {opponent_name}")


def create_neural_agent(
    player: Stone,
    model_path: str,
    top_k: int,
    simulations: int,
    child_top_k: int,
    c_puct: float,
    max_depth: int,
):
    return NeuralNetAI(
        player=player,
        board_size=19,
        model_path=model_path,
        top_k=top_k,
        simulations=simulations,
        child_top_k=child_top_k,
        c_puct=c_puct,
        max_depth=max_depth,
    )


def play_one_game(
    neural_color: Stone,
    opponent_name: str,
    model_path: str,
    neural_top_k: int,
    neural_simulations: int,
    neural_child_top_k: int,
    neural_c_puct: float,
    neural_max_depth: int,
    opponent_simulations: int,
    opponent_time_limit: float,
    max_game_moves: int = 100,
    verbose: bool = False,
) -> Dict:
    """
    Plays one full game and returns a result dict.
    """
    board = GoBoard(size=19)

    if neural_color == Stone.BLACK:
        black_ai = create_neural_agent(
            player=Stone.BLACK,
            model_path=model_path,
            top_k=neural_top_k,
            simulations=neural_simulations,
            child_top_k=neural_child_top_k,
            c_puct=neural_c_puct,
            max_depth=neural_max_depth,
        )
        white_ai = create_opponent(
            opponent_name=opponent_name,
            player=Stone.WHITE,
            simulations=opponent_simulations,
            time_limit=opponent_time_limit,
        )
        black_name = "neural"
        white_name = opponent_name
    else:
        black_ai = create_opponent(
            opponent_name=opponent_name,
            player=Stone.BLACK,
            simulations=opponent_simulations,
            time_limit=opponent_time_limit,
        )
        white_ai = create_neural_agent(
            player=Stone.WHITE,
            model_path=model_path,
            top_k=neural_top_k,
            simulations=neural_simulations,
            child_top_k=neural_child_top_k,
            c_puct=neural_c_puct,
            max_depth=neural_max_depth,
        )
        black_name = opponent_name
        white_name = "neural"

    consecutive_passes = 0
    moves_played = 0
    start_time = time.time()

    for ply in range(max_game_moves):
        winner = board.get_winner()
        if winner is not None:
            break

        current_ai = black_ai if board.current_player == Stone.BLACK else white_ai
        current_name = black_name if board.current_player == Stone.BLACK else white_name

        move = current_ai.get_best_move(board)

        if move is None:
            if hasattr(board, "pass_turn"):
                board.pass_turn()
                consecutive_passes += 1

                if consecutive_passes >= 2:
                    winner = board.get_winner()
                    if winner is None:
                        winner = None
                    break
                continue

            # no move and no pass support
            winner = None
            break

        consecutive_passes = 0

        if not isinstance(move, (tuple, list)) or len(move) != 2:
            winner = None
            break

        row, col = int(move[0]), int(move[1])
        ok = board.place_stone(row, col)
        if not ok:
            winner = None
            break

        moves_played += 1

        if verbose:
            print(
                f"ply={ply:03d} "
                f"player={stone_to_name(Stone.BLACK if board.current_player == Stone.WHITE else Stone.WHITE)} "
                f"agent={current_name} move=({row}, {col})"
            )

    winner = board.get_winner()
    elapsed = time.time() - start_time

    if winner == neural_color:
        outcome = "neural_win"
    elif winner is None:
        outcome = "draw"
    else:
        outcome = "opponent_win"

    result = {
        "opponent": opponent_name,
        "neural_color": stone_to_name(neural_color),
        "winner": stone_to_name(winner) if winner is not None else "draw",
        "winner_int": winner_to_int(winner),
        "outcome": outcome,
        "moves_played": moves_played,
        "elapsed_sec": elapsed,
    }
    return result


def run_match_block(
    opponent_name: str,
    neural_color: Stone,
    games: int,
    model_path: str,
    neural_top_k: int,
    neural_simulations: int,
    neural_child_top_k: int,
    neural_c_puct: float,
    neural_max_depth: int,
    opponent_simulations: int,
    opponent_time_limit: float,
    max_game_moves: int,
    verbose: bool,
) -> List[Dict]:
    results: List[Dict] = []

    print(
        f"\n=== Running block: neural as {stone_to_name(neural_color)} "
        f"vs {opponent_name} for {games} games ==="
    )

    for i in range(games):
        result = play_one_game(
            neural_color=neural_color,
            opponent_name=opponent_name,
            model_path=model_path,
            neural_top_k=neural_top_k,
            neural_simulations=neural_simulations,
            neural_child_top_k=neural_child_top_k,
            neural_c_puct=neural_c_puct,
            neural_max_depth=neural_max_depth,
            opponent_simulations=opponent_simulations,
            opponent_time_limit=opponent_time_limit,
            max_game_moves=max_game_moves,
            verbose=verbose,
        )
        results.append(result)

        print(
            f"game {i+1}/{games} | "
            f"winner={result['winner']} | "
            f"outcome={result['outcome']} | "
            f"moves={result['moves_played']} | "
            f"time={result['elapsed_sec']:.2f}s"
        )

    return results


def summarize_results(results: List[Dict]) -> Dict:
    outcome_counter = Counter(r["outcome"] for r in results)
    winner_counter = Counter(r["winner"] for r in results)
    opponent_counter = Counter(r["opponent"] for r in results)
    neural_color_counter = Counter(r["neural_color"] for r in results)

    total_games = len(results)
    neural_wins = outcome_counter.get("neural_win", 0)
    opponent_wins = outcome_counter.get("opponent_win", 0)
    draws = outcome_counter.get("draw", 0)

    avg_moves = sum(r["moves_played"] for r in results) / total_games if total_games else 0.0
    avg_time = sum(r["elapsed_sec"] for r in results) / total_games if total_games else 0.0

    summary = {
        "total_games": total_games,
        "neural_wins": neural_wins,
        "opponent_wins": opponent_wins,
        "draws": draws,
        "neural_win_rate": neural_wins / total_games if total_games else 0.0,
        "avg_moves": avg_moves,
        "avg_time_sec": avg_time,
        "winner_counts": dict(winner_counter),
        "opponent_counts": dict(opponent_counter),
        "neural_color_counts": dict(neural_color_counter),
    }
    return summary


def print_summary(title: str, summary: Dict) -> None:
    print(f"\n=== {title} ===")
    print(f"total_games: {summary['total_games']}")
    print(f"neural_wins: {summary['neural_wins']}")
    print(f"opponent_wins: {summary['opponent_wins']}")
    print(f"draws: {summary['draws']}")
    print(f"neural_win_rate: {summary['neural_win_rate']:.3f}")
    print(f"avg_moves: {summary['avg_moves']:.2f}")
    print(f"avg_time_sec: {summary['avg_time_sec']:.2f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run tournaments for the neural-guided Go agent.")
    parser.add_argument(
        "--model-path",
        type=str,
        default="AI_Agent_NN/weights/best_model.keras",
        help="Path to best_model.keras",
    )
    parser.add_argument(
        "--games-per-color",
        type=int,
        default=5,
        help="Games for each color matchup block",
    )
    parser.add_argument(
        "--opponent-simulations",
        type=int,
        default=500,
        help="Monte Carlo baseline simulations",
    )
    parser.add_argument(
        "--opponent-time-limit",
        type=float,
        default=1.0,
        help="Monte Carlo baseline time limit",
    )
    parser.add_argument(
        "--neural-top-k",
        type=int,
        default=5,
        help="Top-k legal moves kept by the neural-guided search",
    )
    parser.add_argument(
        "--neural-simulations",
        type=int,
        default=30,
        help="Restricted search simulations",
    )
    parser.add_argument(
        "--neural-child-top-k",
        type=int,
        default=3,
        help="Child node top-k moves in restricted search",
    )
    parser.add_argument(
        "--neural-c-puct",
        type=float,
        default=1.5,
        help="PUCT exploration constant",
    )
    parser.add_argument(
        "--neural-max-depth",
        type=int,
        default=10,
        help="Max restricted search depth",
    )
    parser.add_argument(
        "--max-game-moves",
        type=int,
        default=100,
        help="Maximum moves before game is treated as unfinished/draw",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default="",
        help="Optional path to save tournament results as JSON",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print every move",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    all_results: List[Dict] = []

    # 5 black + 5 white vs Albi MC
    all_results.extend(
        run_match_block(
            opponent_name="albi_mcts",
            neural_color=Stone.BLACK,
            games=args.games_per_color,
            model_path=args.model_path,
            neural_top_k=args.neural_top_k,
            neural_simulations=args.neural_simulations,
            neural_child_top_k=args.neural_child_top_k,
            neural_c_puct=args.neural_c_puct,
            neural_max_depth=args.neural_max_depth,
            opponent_simulations=args.opponent_simulations,
            opponent_time_limit=args.opponent_time_limit,
            max_game_moves=args.max_game_moves,
            verbose=args.verbose,
        )
    )

    all_results.extend(
        run_match_block(
            opponent_name="albi_mcts",
            neural_color=Stone.WHITE,
            games=args.games_per_color,
            model_path=args.model_path,
            neural_top_k=args.neural_top_k,
            neural_simulations=args.neural_simulations,
            neural_child_top_k=args.neural_child_top_k,
            neural_c_puct=args.neural_c_puct,
            neural_max_depth=args.neural_max_depth,
            opponent_simulations=args.opponent_simulations,
            opponent_time_limit=args.opponent_time_limit,
            max_game_moves=args.max_game_moves,
            verbose=args.verbose,
        )
    )

    # 5 black + 5 white vs Ankon MC
    all_results.extend(
        run_match_block(
            opponent_name="ankon_mcts",
            neural_color=Stone.BLACK,
            games=args.games_per_color,
            model_path=args.model_path,
            neural_top_k=args.neural_top_k,
            neural_simulations=args.neural_simulations,
            neural_child_top_k=args.neural_child_top_k,
            neural_c_puct=args.neural_c_puct,
            neural_max_depth=args.neural_max_depth,
            opponent_simulations=args.opponent_simulations,
            opponent_time_limit=args.opponent_time_limit,
            max_game_moves=args.max_game_moves,
            verbose=args.verbose,
        )
    )

    all_results.extend(
        run_match_block(
            opponent_name="ankon_mcts",
            neural_color=Stone.WHITE,
            games=args.games_per_color,
            model_path=args.model_path,
            neural_top_k=args.neural_top_k,
            neural_simulations=args.neural_simulations,
            neural_child_top_k=args.neural_child_top_k,
            neural_c_puct=args.neural_c_puct,
            neural_max_depth=args.neural_max_depth,
            opponent_simulations=args.opponent_simulations,
            opponent_time_limit=args.opponent_time_limit,
            max_game_moves=args.max_game_moves,
            verbose=args.verbose,
        )
    )

    albi_results = [r for r in all_results if r["opponent"] == "albi_mcts"]
    ankon_results = [r for r in all_results if r["opponent"] == "ankon_mcts"]

    overall_summary = summarize_results(all_results)
    albi_summary = summarize_results(albi_results)
    ankon_summary = summarize_results(ankon_results)

    print_summary("OVERALL SUMMARY", overall_summary)
    print_summary("VS ALBI MCTS", albi_summary)
    print_summary("VS ANKON MCTS", ankon_summary)

    payload = {
        "overall_summary": overall_summary,
        "albi_summary": albi_summary,
        "ankon_summary": ankon_summary,
        "results": all_results,
    }

    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"\nSaved tournament results to: {out_path}")


if __name__ == "__main__":
    main()