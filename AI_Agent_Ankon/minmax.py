"""
Minimax AI Agent - Ankon
A Go AI using Minimax algorithm with alpha-beta pruning.
Different evaluation strategy focusing on defensive play and group survival.
"""

import random
from typing import Tuple, List, Optional
from game_models import GoBoard, Stone


class MinimaxAI:
    """
    AI Agent using Minimax with Alpha-Beta pruning.
    Evaluation focuses on defensive play, group survival, and strategic positions.
    """
    
    def __init__(self, player: Stone, depth: int = 3):
        """
        Initialize the AI agent.
        
        Args:
            player: The stone color this AI plays as (Stone.BLACK or Stone.WHITE)
            depth: Maximum depth for minimax search
        """
        self.player = player
        self.depth = depth
        self.opponent = Stone.WHITE if player == Stone.BLACK else Stone.BLACK
    
    def get_best_move(self, board: GoBoard) -> Optional[Tuple[int, int]]:
        """
        Get the best move for the current position.
        
        Args:
            board: Current game state
            
        Returns:
            Best move as (row, col) or None if no valid moves
        """
        # Get all valid moves
        valid_moves = self._get_valid_moves(board)
        
        if not valid_moves:
            return None
        
        # If only one move, take it
        if len(valid_moves) == 1:
            return valid_moves[0]
        
        # Use minimax to find best move
        best_move = None
        best_value = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        
        for move in valid_moves:
            # Create a copy of the board and make the move
            new_board = self._copy_board(board)
            if new_board.place_stone(move[0], move[1]):
                # Evaluate this move
                value = self._minimax(new_board, self.depth - 1, alpha, beta, False)
                
                if value > best_value:
                    best_value = value
                    best_move = move
                
                alpha = max(alpha, best_value)
        
        return best_move if best_move else random.choice(valid_moves)
    
    def _minimax(self, board: GoBoard, depth: int, alpha: float, beta: float, is_maximizing: bool) -> float:
        """
        Minimax algorithm with alpha-beta pruning.
        
        Args:
            board: Current game state
            depth: Remaining search depth
            alpha: Best value maximizer can guarantee
            beta: Best value minimizer can guarantee
            is_maximizing: True if maximizing player's turn
            
        Returns:
            Evaluated board value
        """
        # Check terminal states
        winner = self._check_winner(board)
        if winner == self.player:
            return 10000
        elif winner == self.opponent:
            return -10000
        elif depth == 0:
            return self._evaluate(board)
        
        # Get valid moves
        valid_moves = self._get_valid_moves(board)
        
        if not valid_moves:
            # No moves available - pass
            return self._evaluate(board)
        
        if is_maximizing:
            max_eval = float('-inf')
            for move in valid_moves:
                new_board = self._copy_board(board)
                if new_board.place_stone(move[0], move[1]):
                    eval_score = self._minimax(new_board, depth - 1, alpha, beta, False)
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
                    eval_score = self._minimax(new_board, depth - 1, alpha, beta, True)
                    min_eval = min(min_eval, eval_score)
                    beta = min(beta, eval_score)
                    if beta <= alpha:
                        break
            return min_eval
    
    def _evaluate(self, board: GoBoard) -> float:
        """
        Evaluate the board position.
        This version focuses on defensive play and group survival.
        
        Args:
            board: Current game state
            
        Returns:
            Evaluation score (positive is good for AI)
        """
        score = 0
        
        # Weight for captured stones (most important in Atari Go)
        my_captures = board.captured_stones[self.player]
        opp_captures = board.captured_stones[self.opponent]
        score += my_captures * 100
        score -= opp_captures * 100
        
        # Count stones on board
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
        
        # Evaluate group survival (prioritize groups with more liberties)
        my_groups = self._get_all_groups(board, self.player)
        opp_groups = self._get_all_groups(board, self.opponent)
        
        # Reward groups with many liberties (safe groups)
        for group in my_groups:
            if group:  # Check if group is not empty
                liberties = board._count_liberties(group[0][0], group[0][1])
                score += liberties * 3  # Higher weight for safety
                
                # Bonus for larger groups
                if len(group) > 1:
                    score += len(group) * 2
        
        # Penalty for opponent groups with few liberties (threats)
        for group in opp_groups:
            if group:  # Check if group is not empty
                liberties = board._count_liberties(group[0][0], group[0][1])
                if liberties <= 1:
                    score += 15  # High value for threatening capture
                elif liberties == 2:
                    score += 5
                score -= liberties * 2
        
        # Evaluate connectivity (prefer connected stones)
        score += self._evaluate_connectivity(board, self.player) * 2
        score -= self._evaluate_connectivity(board, self.opponent) * 2
        
        # Corner and edge preference (strategic positions)
        center = board.size // 2
        for row in range(board.size):
            for col in range(board.size):
                if board.board[row][col] == self.player:
                    # Prefer corners and edges for stability
                    if row == 0 or row == board.size - 1 or col == 0 or col == board.size - 1:
                        score += 1
                    # Slight preference for center influence
                    elif abs(row - center) <= 2 and abs(col - center) <= 2:
                        score += 0.5
                elif board.board[row][col] == self.opponent:
                    if row == 0 or row == board.size - 1 or col == 0 or col == board.size - 1:
                        score -= 1
                    elif abs(row - center) <= 2 and abs(col - center) <= 2:
                        score -= 0.5
        
        # Evaluate potential captures (moves that could capture next turn)
        potential_captures = self._count_potential_captures(board, self.player)
        score += potential_captures * 20
        
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
    
    def _evaluate_connectivity(self, board: GoBoard, player: Stone) -> int:
        """Evaluate how connected the stones are."""
        groups = self._get_all_groups(board, player)
        
        # Score based on group sizes
        connectivity_score = 0
        for group in groups:
            connectivity_score += len(group)
        
        return connectivity_score
    
    def _count_potential_captures(self, board: GoBoard, player: Stone) -> int:
        """Count potential capture opportunities (opponent groups with 1 liberty)."""
        opponent = Stone.WHITE if player == Stone.BLACK else Stone.BLACK
        opp_groups = self._get_all_groups(board, opponent)
        
        count = 0
        for group in opp_groups:
            if group and len(group) > 0:
                liberties = board._count_liberties(group[0][0], group[0][1])
                if liberties == 1:
                    count += 1
        
        return count
    
    def _get_valid_moves(self, board: GoBoard) -> List[Tuple[int, int]]:
        """
        Get all valid moves (empty positions).
        
        Args:
            board: Current game state
            
        Returns:
            List of valid moves as (row, col)
        """
        moves = []
        
        for row in range(board.size):
            for col in range(board.size):
                if board.is_empty(row, col):
                    # Check if move is legal (not suicide unless it captures)
                    if self._is_legal_move(board, row, col):
                        moves.append((row, col))
        
        return moves
    
    def _is_legal_move(self, board: GoBoard, row: int, col: int) -> bool:
        """
        Check if a move is legal.
        
        Args:
            board: Current game state
            row: Row position
            col: Column position
            
        Returns:
            True if move is legal
        """
        # Create a copy and try the move
        test_board = self._copy_board(board)
        return test_board.place_stone(row, col)
    
    def _copy_board(self, board: GoBoard) -> GoBoard:
        """
        Create a deep copy of the board.
        
        Args:
            board: Board to copy
            
        Returns:
            New board instance with copied state
        """
        new_board = GoBoard(size=board.size)
        
        # Copy board state
        for row in range(board.size):
            for col in range(board.size):
                new_board.board[row][col] = board.board[row][col]
        
        new_board.current_player = board.current_player
        new_board.move_count = board.move_count
        new_board.captured_stones = board.captured_stones.copy()
        
        return new_board
    
    def _check_winner(self, board: GoBoard) -> Optional[Stone]:
        """
        Check if there's a winner.
        
        Args:
            board: Current game state
            
        Returns:
            Stone color of winner, or None if no winner yet
        """
        if board.captured_stones[self.player] > 0:
            return self.player
        if board.captured_stones[self.opponent] > 0:
            return self.opponent
        return None


def get_ai_move(board_state, player_color, depth=3):
    """
    Function to be called from main.py.
    
    Args:
        board_state: 2D list representing the board (0=empty, 1=black, 2=white)
        player_color: 1 for black, 2 for white
        depth: Search depth (optional, default 3)
        
    Returns:
        Tuple (row, col) of the best move
    """
    # Convert to Stone enum
    player = Stone.BLACK if player_color == 1 else Stone.WHITE
    
    # Create board from state
    board = GoBoard(size=len(board_state))
    for row in range(len(board_state)):
        for col in range(len(board_state[row])):
            if board_state[row][col] == 1:
                board.board[row][col] = Stone.BLACK
            elif board_state[row][col] == 2:
                board.board[row][col] = Stone.WHITE
    
    # Create AI and get move
    ai = MinimaxAI(player=player, depth=depth)
    move = ai.get_best_move(board)
    
    return move if move else (-1, -1)


if __name__ == "__main__":
    # Test the AI
    board = GoBoard(size=19)
    ai = MinimaxAI(player=Stone.BLACK, depth=2)
    
    print("Testing Minimax AI (Ankon)...")
    move = ai.get_best_move(board)
    print(f"Best first move: {move}")