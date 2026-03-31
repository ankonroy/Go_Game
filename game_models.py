"""
Shared game models for Go Game.
Contains Stone enum and GoBoard class without pygame dependencies.
"""

from enum import Enum
from typing import Tuple, List, Set, Optional
from copy import deepcopy


class Stone(Enum):
    EMPTY = 0
    BLACK = 1
    WHITE = 2


class GoBoard:
    """
    Represents a Go game board with standard 19x19 grid.
    Supports stone placement, capture detection, and basic game state.
    """
    
    def __init__(self, size: int = 19):
        """Initialize the Go board."""
        self.size = size
        self.board = [[Stone.EMPTY for _ in range(size)] for _ in range(size)]
        self.current_player = Stone.BLACK
        self.move_count = 0
        self.captured_stones = {Stone.BLACK: 0, Stone.WHITE: 0}
        self.last_move = None  # Track last move for highlighting
        
        # FIXED: Add move history to prevent repetition/Ko
        self.move_history = []
        self.ko_point = None  # Point that cannot be played due to Ko rule
        
    def is_valid_position(self, row: int, col: int) -> bool:
        """Check if a position is within the board."""
        return 0 <= row < self.size and 0 <= col < self.size
    
    def is_empty(self, row: int, col: int) -> bool:
        """Check if a position is empty."""
        return self.board[row][col] == Stone.EMPTY
    
    def place_stone(self, row: int, col: int) -> bool:
        """Place a stone at the specified position."""
        if not self.is_valid_position(row, col):
            return False
        
        if not self.is_empty(row, col):
            return False
        
        # FIXED: Check Ko rule
        if self.ko_point == (row, col):
            return False
        
        # Save current board state for Ko checking
        old_board = self._copy_board_state()
        
        # Place the stone temporarily
        self.board[row][col] = self.current_player
        
        # Check for suicide rule
        if self._count_liberties(row, col) == 0:
            # Try to capture first
            captured = self._capture_opponent_stones(row, col)
            if captured == 0:
                # Suicide move - undo
                self.board[row][col] = Stone.EMPTY
                return False
        
        # Capture any opponent stones
        self._capture_opponent_stones(row, col)
        
        # FIXED: Check if this move recreates a previous board state (Ko)
        current_state = self._board_to_tuple()
        if current_state in self.move_history:
            # Ko violation - undo move
            self.board[row][col] = Stone.EMPTY
            return False
        
        # Record last move
        self.last_move = (row, col)
        
        # FIXED: Add to move history (keep last few moves only)
        self.move_history.append(current_state)
        if len(self.move_history) > 10:  # Keep only last 10 moves
            self.move_history.pop(0)
        
        # FIXED: Update Ko point (point that would recreate previous board)
        self.ko_point = self._calculate_ko_point(row, col, old_board)
        
        # Switch player
        self.current_player = Stone.WHITE if self.current_player == Stone.BLACK else Stone.BLACK
        self.move_count += 1
        
        return True
    
    def _board_to_tuple(self) -> Tuple:
        """Convert board to hashable tuple for history tracking."""
        return tuple(tuple(row) for row in self.board)
    
    def _copy_board_state(self) -> List[List[Stone]]:
        """Create a copy of the current board state."""
        return [row[:] for row in self.board]  # Using list slice for speed
    
    def _calculate_ko_point(self, row: int, col: int, old_board: List[List[Stone]]) -> Optional[Tuple[int, int]]:
        """
        Calculate if this move creates a Ko situation.
        Returns the point that cannot be played next turn, or None.
        """
        # Check if exactly one stone was captured
        captured_count = 0
        captured_pos = None
        
        for r in range(self.size):
            for c in range(self.size):
                if old_board[r][c] != Stone.EMPTY and self.board[r][c] == Stone.EMPTY:
                    captured_count += 1
                    captured_pos = (r, c)
        
        # If exactly one stone was captured and the capturing stone has only one liberty,
        # that liberty is the Ko point
        if captured_count == 1 and captured_pos:
            # The captured position is where the opponent just played
            # The Ko point is where the opponent just captured
            if self._count_liberties(row, col) == 1:
                return captured_pos
        
        return None
    
    def _get_neighbors(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Get all valid neighboring positions."""
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if self.is_valid_position(new_row, new_col):
                neighbors.append((new_row, new_col))
        
        return neighbors
    
    def _get_stone_group(self, row: int, col: int) -> Set[Tuple[int, int]]:
        """Get all connected stones of the same color (group)."""
        if not self.is_valid_position(row, col):
            return set()
        
        stone_color = self.board[row][col]
        if stone_color == Stone.EMPTY:
            return set()
        
        group = set()
        visited = set()
        stack = [(row, col)]
        
        # FIXED: Add iteration limit to prevent infinite loops
        max_iterations = self.size * self.size
        iterations = 0
        
        while stack and iterations < max_iterations:
            r, c = stack.pop()
            if (r, c) in visited:
                continue
            visited.add((r, c))
            
            if self.board[r][c] == stone_color:
                group.add((r, c))
                for nr, nc in self._get_neighbors(r, c):
                    if (nr, nc) not in visited and self.board[nr][nc] == stone_color:
                        stack.append((nr, nc))
            
            iterations += 1
        
        return group
    
    def _count_liberties(self, row: int, col: int) -> int:
        """Count the number of liberties for a stone or group at position."""
        if not self.is_valid_position(row, col):
            return 0
        
        stone_color = self.board[row][col]
        if stone_color == Stone.EMPTY:
            return 0
        
        group = self._get_stone_group(row, col)
        liberties = set()
        
        # FIXED: Add iteration limit
        max_liberties = self.size * 4  # Maximum possible liberties
        for r, c in group:
            for nr, nc in self._get_neighbors(r, c):
                if self.board[nr][nc] == Stone.EMPTY:
                    liberties.add((nr, nc))
                    if len(liberties) > max_liberties:
                        return max_liberties
        
        return len(liberties)
    
    def _capture_opponent_stones(self, row: int, col: int) -> int:
        """Capture opponent stones that have no liberties after a move."""
        captured_count = 0
        opponent = Stone.WHITE if self.current_player == Stone.BLACK else Stone.BLACK
        
        # FIXED: Use a set to avoid processing the same group multiple times
        processed_groups = set()
        
        for nr, nc in self._get_neighbors(row, col):
            if self.board[nr][nc] == opponent:
                # Create a key for this potential group
                if (nr, nc) in processed_groups:
                    continue
                
                liberties = self._count_liberties(nr, nc)
                if liberties == 0:
                    group = self._get_stone_group(nr, nc)
                    # Mark all stones in this group as processed
                    for gr, gc in group:
                        processed_groups.add((gr, gc))
                        self.board[gr][gc] = Stone.EMPTY
                    captured_count += len(group)
                    self.captured_stones[self.current_player] += len(group)
        
        return captured_count
    
    def pass_turn(self):
        """Pass the current turn."""
        self.current_player = Stone.WHITE if self.current_player == Stone.BLACK else Stone.BLACK
        self.move_count += 1
        self.last_move = None
        # FIXED: Reset Ko point on pass
        self.ko_point = None
    
    def reset(self):
        """Reset the board."""
        self.board = [[Stone.EMPTY for _ in range(self.size)] for _ in range(self.size)]
        self.current_player = Stone.BLACK
        self.move_count = 0
        self.captured_stones = {Stone.BLACK: 0, Stone.WHITE: 0}
        self.last_move = None
        # FIXED: Reset history
        self.move_history = []
        self.ko_point = None
    
    def has_captured(self) -> bool:
        """Check if any player has captured at least one stone."""
        return self.captured_stones[Stone.BLACK] > 0 or self.captured_stones[Stone.WHITE] > 0
    
    def get_winner(self) -> Optional[Stone]:
        """Get the winner if game is over (first to capture wins)."""
        if self.captured_stones[Stone.BLACK] > 0:
            return Stone.BLACK
        elif self.captured_stones[Stone.WHITE] > 0:
            return Stone.WHITE
        return None
    
    def is_game_over(self) -> bool:
        """Check if the game is over."""
        return self.get_winner() is not None
    
    # FIXED: Add utility methods for AI
    def get_legal_moves(self) -> List[Tuple[int, int]]:
        """Get all legal moves for current player."""
        moves = []
        for row in range(self.size):
            for col in range(self.size):
                if self.is_empty(row, col) and (row, col) != self.ko_point:
                    # Quick check - actual legality will be verified during placement
                    moves.append((row, col))
        return moves
    
    def copy(self) -> 'GoBoard':
        """Create a deep copy of the board (useful for AI)."""
        new_board = GoBoard(size=self.size)
        for row in range(self.size):
            new_board.board[row] = self.board[row][:]  # List slice for speed
        new_board.current_player = self.current_player
        new_board.move_count = self.move_count
        new_board.captured_stones = self.captured_stones.copy()
        new_board.last_move = self.last_move
        new_board.move_history = self.move_history.copy()
        new_board.ko_point = self.ko_point
        return new_board
    
    def __str__(self) -> str:
        """String representation for debugging."""
        symbols = {Stone.EMPTY: '.', Stone.BLACK: 'B', Stone.WHITE: 'W'}
        result = []
        for row in self.board:
            result.append(' '.join(symbols[stone] for stone in row))
        return '\n'.join(result)