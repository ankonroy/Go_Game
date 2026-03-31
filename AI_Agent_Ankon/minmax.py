"""
Minimax AI Agent - Ankon
A Go AI using Minimax algorithm with alpha-beta pruning.
"""

import random
from typing import Tuple, List, Optional
from game_models import GoBoard, Stone


class MinimaxAI:
    """
    AI Agent using Minimax with Alpha-Beta pruning.
    Includes safeguards against infinite recursion.
    """
    
    def __init__(self, player: Stone, depth: int = 2):  # Reduced default depth
        """
        Initialize the AI agent.
        
        Args:
            player: The stone color this AI plays as
            depth: Maximum depth for minimax search (keep small for 19x19)
        """
        self.player = player
        self.max_depth = min(depth, 3)  # Cap at depth 3 for safety
        self.opponent = Stone.WHITE if player == Stone.BLACK else Stone.BLACK
        self.nodes_evaluated = 0
        self.max_nodes = 10000  # Safety limit
    
    def get_best_move(self, board: GoBoard) -> Optional[Tuple[int, int]]:
        """Get the best move with safety limits."""
        self.nodes_evaluated = 0
        
        valid_moves = self._get_valid_moves(board)
        
        if not valid_moves:
            return None
        
        if len(valid_moves) == 1:
            return valid_moves[0]
        
        # Prune moves for performance
        if len(valid_moves) > 50:
            valid_moves = self._prune_moves(board, valid_moves, 30)
        
        best_move = None
        best_value = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        
        for move in valid_moves:
            if self.nodes_evaluated > self.max_nodes:
                print(f"Node limit reached ({self.max_nodes})")
                break
            
            new_board = self._copy_board(board)
            if new_board.place_stone(move[0], move[1]):
                value = self._minimax(new_board, 1, alpha, beta, False)
                
                if value > best_value:
                    best_value = value
                    best_move = move
                
                alpha = max(alpha, best_value)
        
        return best_move if best_move else random.choice(valid_moves)
    
    def _prune_moves(self, board: GoBoard, moves: List[Tuple[int, int]], 
                     keep: int) -> List[Tuple[int, int]]:
        """Prune moves based on heuristics."""
        scored_moves = []
        
        for move in moves:
            score = 0
            row, col = move
            
            # Prefer center and corners
            center = board.size // 2
            distance = abs(row - center) + abs(col - center)
            score += (board.size - distance)
            
            # Prefer moves near opponent stones
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = row + dr, col + dc
                if board.is_valid_position(nr, nc):
                    if board.board[nr][nc] == self.opponent:
                        score += 15
                    elif board.board[nr][nc] == self.player:
                        score += 5
            
            scored_moves.append((move, score))
        
        scored_moves.sort(key=lambda x: x[1], reverse=True)
        return [m[0] for m in scored_moves[:keep]]
    
    def _minimax(self, board: GoBoard, depth: int, alpha: float, beta: float, 
                 is_maximizing: bool) -> float:
        """Minimax with depth limiting and node counting."""
        self.nodes_evaluated += 1
        
        # Safety checks
        if self.nodes_evaluated > self.max_nodes:
            return self._evaluate(board)
        
        if depth >= self.max_depth:
            return self._evaluate(board)
        
        # Check terminal states
        winner = self._check_winner(board)
        if winner == self.player:
            return 10000 - depth
        elif winner == self.opponent:
            return -10000 + depth
        
        valid_moves = self._get_valid_moves(board)
        
        if not valid_moves:
            return self._evaluate(board)
        
        # Limit moves at deeper levels
        if depth > 1 and len(valid_moves) > 15:
            valid_moves = self._prune_moves(board, valid_moves, 10)
        
        if is_maximizing:
            max_eval = float('-inf')
            for move in valid_moves:
                new_board = self._copy_board(board)
                if new_board.place_stone(move[0], move[1]):
                    eval_score = self._minimax(new_board, depth + 1, alpha, beta, False)
                    max_eval = max(max_eval, eval_score)
                    alpha = max(alpha, eval_score)
                    if beta <= alpha:
                        break
            return max_eval
        else:
            min_eval = float('inf')
            for move in valid_moves:
                new_board = self._copy_board(board)
                if new_board.place_stone(move[0], move[1]):
                    eval_score = self._minimax(new_board, depth + 1, alpha, beta, True)
                    min_eval = min(min_eval, eval_score)
                    beta = min(beta, eval_score)
                    if beta <= alpha:
                        break
            return min_eval
    
    def _evaluate(self, board: GoBoard) -> float:
        """Evaluate the board position."""
        score = 0
        
        # Captured stones
        my_captures = board.captured_stones[self.player]
        opp_captures = board.captured_stones[self.opponent]
        score += my_captures * 100
        score -= opp_captures * 100
        
        # Count stones
        my_stones = 0
        opp_stones = 0
        
        for row in range(board.size):
            for col in range(board.size):
                if board.board[row][col] == self.player:
                    my_stones += 1
                elif board.board[row][col] == self.opponent:
                    opp_stones += 1
        
        score += my_stones * 4
        score -= opp_stones * 4
        
        # Evaluate groups
        my_groups = self._get_all_groups(board, self.player)
        opp_groups = self._get_all_groups(board, self.opponent)
        
        for group in my_groups:
            if group:
                liberties = board._count_liberties(group[0][0], group[0][1])
                score += liberties * 3
                if len(group) > 1:
                    score += len(group) * 2
        
        for group in opp_groups:
            if group:
                liberties = board._count_liberties(group[0][0], group[0][1])
                if liberties <= 1:
                    score += 15
                elif liberties == 2:
                    score += 5
                score -= liberties * 2
        
        return score
    
    def _get_all_groups(self, board: GoBoard, player: Stone) -> List[List[Tuple[int, int]]]:
        """Get all stone groups for a player."""
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
        """Get all valid moves."""
        moves = []
        for row in range(board.size):
            for col in range(board.size):
                if board.is_empty(row, col):
                    if self._is_legal_move(board, row, col):
                        moves.append((row, col))
        return moves
    
    def _is_legal_move(self, board: GoBoard, row: int, col: int) -> bool:
        """Check if a move is legal."""
        test_board = self._copy_board(board)
        return test_board.place_stone(row, col)
    
    def _copy_board(self, board: GoBoard) -> GoBoard:
        """Create a deep copy of the board."""
        new_board = GoBoard(size=board.size)
        for row in range(board.size):
            for col in range(board.size):
                new_board.board[row][col] = board.board[row][col]
        new_board.current_player = board.current_player
        new_board.move_count = board.move_count
        new_board.captured_stones = board.captured_stones.copy()
        return new_board
    
    def _check_winner(self, board: GoBoard) -> Optional[Stone]:
        """Check if there's a winner."""
        if board.captured_stones[self.player] > 0:
            return self.player
        if board.captured_stones[self.opponent] > 0:
            return self.opponent
        return None


def get_ai_move(board_state, player_color, depth=2):
    """Function to be called from main.py."""
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


if __name__ == "__main__":
    board = GoBoard(size=19)
    ai = MinimaxAI(player=Stone.BLACK, depth=2)
    print("Testing Minimax AI (Ankon)...")
    move = ai.get_best_move(board)
    print(f"Best first move: {move}")