"""
Stronger Minimax AI Agent - Albi
A stronger Go AI using Minimax with alpha-beta pruning.
Tuned for the project's first-capture-wins rules.
"""

import random
from typing import Dict, List, Optional, Tuple
from game_models import GoBoard, Stone


class MinimaxAI:
    """
    Stronger minimax agent with:
    - alpha-beta pruning
    - better move ordering
    - immediate tactical checks (win now / block now)
    - board-state memoization
    - group danger evaluation tuned for capture-based play
    """

    def __init__(self, player: Stone, depth: int = 3):
        self.player = player
        self.opponent = Stone.WHITE if player == Stone.BLACK else Stone.BLACK
        self.max_depth = min(depth, 3)
        self.nodes_evaluated = 0
        self.max_nodes = 18000
        self.ttable: Dict[Tuple, float] = {}

    def get_best_move(self, board: GoBoard) -> Optional[Tuple[int, int]]:
        """Get strongest move found for the current position."""
        self.nodes_evaluated = 0
        self.ttable.clear()

        valid_moves = self._get_valid_moves(board)
        if not valid_moves:
            return None
        if len(valid_moves) == 1:
            return valid_moves[0]

        # 1) If we can win immediately, do it.
        winning_move = self._find_immediate_winning_move(board, valid_moves)
        if winning_move is not None:
            return winning_move

        # 2) If opponent has immediate winning threats, try to block them first.
        opponent_winning_moves = self._find_immediate_winning_moves_for_player(board, self.opponent)
        if opponent_winning_moves:
            blocking_moves = self._find_blocking_moves(board, valid_moves, opponent_winning_moves)
            if blocking_moves:
                valid_moves = blocking_moves

        # 3) Order moves well before search.
        ordered_moves = self._order_moves(board, valid_moves)
        if len(ordered_moves) > 28:
            ordered_moves = ordered_moves[:28]

        best_move = ordered_moves[0]
        best_value = float('-inf')
        alpha = float('-inf')
        beta = float('inf')

        for move in ordered_moves:
            if self.nodes_evaluated >= self.max_nodes:
                break

            new_board = self._copy_board(board)
            if not new_board.place_stone(move[0], move[1]):
                continue

            value = self._minimax(new_board, 1, alpha, beta, False)
            if value > best_value:
                best_value = value
                best_move = move
            alpha = max(alpha, best_value)

        return best_move

    def _minimax(self, board: GoBoard, depth: int, alpha: float, beta: float,
                 is_maximizing: bool) -> float:
        self.nodes_evaluated += 1

        winner = self._check_winner(board)
        if winner == self.player:
            return 100000 - depth
        if winner == self.opponent:
            return -100000 + depth

        if self.nodes_evaluated >= self.max_nodes or depth >= self.max_depth:
            return self._evaluate(board)

        cache_key = self._hash_board(board, depth, is_maximizing)
        cached = self.ttable.get(cache_key)
        if cached is not None:
            return cached

        valid_moves = self._get_valid_moves(board)
        if not valid_moves:
            score = self._evaluate(board)
            self.ttable[cache_key] = score
            return score

        ordered_moves = self._order_moves(board, valid_moves)
        if depth >= 1 and len(ordered_moves) > 14:
            ordered_moves = ordered_moves[:14]

        if is_maximizing:
            value = float('-inf')
            for move in ordered_moves:
                new_board = self._copy_board(board)
                if not new_board.place_stone(move[0], move[1]):
                    continue
                value = max(value, self._minimax(new_board, depth + 1, alpha, beta, False))
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
        else:
            value = float('inf')
            for move in ordered_moves:
                new_board = self._copy_board(board)
                if not new_board.place_stone(move[0], move[1]):
                    continue
                value = min(value, self._minimax(new_board, depth + 1, alpha, beta, True))
                beta = min(beta, value)
                if beta <= alpha:
                    break

        self.ttable[cache_key] = value
        return value

    def _evaluate(self, board: GoBoard) -> float:
        """Capture-focused static evaluation for first-capture-wins."""
        score = 0.0

        # Captures dominate this ruleset.
        my_captures = board.captured_stones[self.player]
        opp_captures = board.captured_stones[self.opponent]
        score += my_captures * 5000
        score -= opp_captures * 5000

        my_groups = self._get_all_groups(board, self.player)
        opp_groups = self._get_all_groups(board, self.opponent)

        # Own groups: prefer alive, connected groups.
        for group in my_groups:
            if not group:
                continue
            liberties = board._count_liberties(group[0][0], group[0][1])
            size_bonus = len(group) * 3
            score += size_bonus

            if liberties == 1:
                score -= 180
            elif liberties == 2:
                score -= 40
            else:
                score += liberties * 5

            if len(group) >= 2:
                score += len(group) * 4

        # Opponent groups: reward pressure heavily.
        for group in opp_groups:
            if not group:
                continue
            liberties = board._count_liberties(group[0][0], group[0][1])
            score -= len(group) * 2

            if liberties == 1:
                score += 220
            elif liberties == 2:
                score += 70
            else:
                score -= liberties * 4

        # Mild stone count signal.
        my_stones = 0
        opp_stones = 0
        for row in range(board.size):
            for col in range(board.size):
                if board.board[row][col] == self.player:
                    my_stones += 1
                elif board.board[row][col] == self.opponent:
                    opp_stones += 1
        score += (my_stones - opp_stones) * 2

        return score

    def _order_moves(self, board: GoBoard, moves: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        scored = [(move, self._score_move(board, move)) for move in moves]
        scored.sort(key=lambda item: item[1], reverse=True)
        return [move for move, _ in scored]

    def _score_move(self, board: GoBoard, move: Tuple[int, int]) -> float:
        """Fast heuristic score used only for move ordering/pruning."""
        row, col = move
        score = 0.0

        center = board.size // 2
        distance = abs(row - center) + abs(col - center)
        score += (board.size - distance)

        adjacent_friend = 0
        adjacent_enemy = 0
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if board.is_valid_position(nr, nc):
                if board.board[nr][nc] == self.player:
                    adjacent_friend += 1
                elif board.board[nr][nc] == self.opponent:
                    adjacent_enemy += 1

        score += adjacent_friend * 8
        score += adjacent_enemy * 16

        # Tactical lookahead for this move.
        test_board = self._copy_board(board)
        if test_board.place_stone(row, col):
            winner = self._check_winner(test_board)
            if winner == self.player:
                score += 100000

            my_groups = self._get_all_groups(test_board, self.player)
            opp_groups = self._get_all_groups(test_board, self.opponent)

            for group in opp_groups:
                if group:
                    liberties = test_board._count_liberties(group[0][0], group[0][1])
                    if liberties == 1:
                        score += 150
                    elif liberties == 2:
                        score += 30

            for group in my_groups:
                if group and (row, col) in group:
                    liberties = test_board._count_liberties(group[0][0], group[0][1])
                    if liberties == 1:
                        score -= 220
                    elif liberties == 2:
                        score -= 40

        return score

    def _find_immediate_winning_move(self, board: GoBoard,
                                     moves: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        for move in self._order_moves(board, moves):
            test_board = self._copy_board(board)
            if test_board.place_stone(move[0], move[1]) and self._check_winner(test_board) == self.player:
                return move
        return None

    def _find_immediate_winning_moves_for_player(self, board: GoBoard,
                                                 player: Stone) -> List[Tuple[int, int]]:
        winning = []
        current = board.current_player
        board_for_scan = self._copy_board(board)
        board_for_scan.current_player = player

        for move in self._get_valid_moves_for_player(board_for_scan, player):
            test_board = self._copy_board(board_for_scan)
            test_board.current_player = player
            if test_board.place_stone(move[0], move[1]):
                if test_board.captured_stones[player] > board_for_scan.captured_stones[player]:
                    winning.append(move)

        board.current_player = current
        return winning

    def _find_blocking_moves(self, board: GoBoard, valid_moves: List[Tuple[int, int]],
                             opponent_winning_moves: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        blocking = []
        for move in valid_moves:
            test_board = self._copy_board(board)
            if not test_board.place_stone(move[0], move[1]):
                continue

            opp_threats_after = self._find_immediate_winning_moves_for_player(test_board, self.opponent)
            if len(opp_threats_after) < len(opponent_winning_moves):
                blocking.append(move)

        return self._order_moves(board, blocking)

    def _get_all_groups(self, board: GoBoard, player: Stone) -> List[List[Tuple[int, int]]]:
        groups = []
        counted = set()

        for row in range(board.size):
            for col in range(board.size):
                if board.board[row][col] == player and (row, col) not in counted:
                    group = list(board._get_stone_group(row, col))
                    groups.append(group)
                    counted.update(group)

        return groups

    def _get_valid_moves(self, board: GoBoard) -> List[Tuple[int, int]]:
        moves = []
        for row in range(board.size):
            for col in range(board.size):
                if board.is_empty(row, col) and self._is_legal_move(board, row, col):
                    moves.append((row, col))
        return moves

    def _get_valid_moves_for_player(self, board: GoBoard, player: Stone) -> List[Tuple[int, int]]:
        original = board.current_player
        board.current_player = player
        moves = self._get_valid_moves(board)
        board.current_player = original
        return moves

    def _is_legal_move(self, board: GoBoard, row: int, col: int) -> bool:
        test_board = self._copy_board(board)
        return test_board.place_stone(row, col)

    def _copy_board(self, board: GoBoard) -> GoBoard:
        new_board = GoBoard(size=board.size)
        for row in range(board.size):
            new_board.board[row] = board.board[row][:]
        new_board.current_player = board.current_player
        new_board.move_count = board.move_count
        new_board.captured_stones = board.captured_stones.copy()
        if hasattr(board, 'last_move'):
            new_board.last_move = getattr(board, 'last_move', None)
        return new_board

    def _hash_board(self, board: GoBoard, depth: int, is_maximizing: bool) -> Tuple:
        rows = tuple(tuple(cell.value if hasattr(cell, 'value') else cell for cell in row)
                     for row in board.board)
        current = board.current_player.value if hasattr(board.current_player, 'value') else board.current_player
        return (
            rows,
            current,
            board.captured_stones.get(self.player, 0),
            board.captured_stones.get(self.opponent, 0),
            depth,
            is_maximizing,
        )

    def _check_winner(self, board: GoBoard) -> Optional[Stone]:
        if board.captured_stones[self.player] > 0:
            return self.player
        if board.captured_stones[self.opponent] > 0:
            return self.opponent
        return None



def get_ai_move(board_state, player_color, depth=3):
    player = Stone.BLACK if player_color == 1 else Stone.WHITE

    board = GoBoard(size=len(board_state))
    for row in range(len(board_state)):
        for col in range(len(board_state[row])):
            if board_state[row][col] == 1:
                board.board[row][col] = Stone.BLACK
            elif board_state[row][col] == 2:
                board.board[row][col] = Stone.WHITE

    ai = MinimaxAI(player=player, depth=depth)
    move = ai.get_best_move(board)
    return move if move else (-1, -1)