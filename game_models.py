"""
Shared game models for Go Game.
Contains Stone enum and GoBoard class without pygame dependencies.
"""

from enum import Enum
from typing import Tuple, List, Set, Optional


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
        
        # Place the stone temporarily
        self.board[row][col] = self.current_player
        
        # Check for suicide rule
        if self._count_liberties(row, col) == 0:
            captured = self._capture_opponent_stones(row, col)
            if captured == 0:
                self.board[row][col] = Stone.EMPTY
                return False
        
        # Capture any opponent stones
        self._capture_opponent_stones(row, col)
        
        # Record last move
        self.last_move = (row, col)
        
        # Switch player
        self.current_player = Stone.WHITE if self.current_player == Stone.BLACK else Stone.BLACK
        self.move_count += 1
        
        return True
    
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
        
        while stack:
            r, c = stack.pop()
            if (r, c) in visited:
                continue
            visited.add((r, c))
            
            if self.board[r][c] == stone_color:
                group.add((r, c))
                for nr, nc in self._get_neighbors(r, c):
                    if (nr, nc) not in visited and self.board[nr][nc] == stone_color:
                        stack.append((nr, nc))
        
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
        
        for r, c in group:
            for nr, nc in self._get_neighbors(r, c):
                if self.board[nr][nc] == Stone.EMPTY:
                    liberties.add((nr, nc))
        
        return len(liberties)
    
    def _capture_opponent_stones(self, row: int, col: int) -> int:
        """Capture opponent stones that have no liberties after a move."""
        captured_count = 0
        opponent = Stone.WHITE if self.current_player == Stone.BLACK else Stone.BLACK
        
        for nr, nc in self._get_neighbors(row, col):
            if self.board[nr][nc] == opponent:
                liberties = self._count_liberties(nr, nc)
                if liberties == 0:
                    group = self._get_stone_group(nr, nc)
                    for gr, gc in group:
                        self.board[gr][gc] = Stone.EMPTY
                    captured_count += len(group)
                    self.captured_stones[self.current_player] += len(group)
        
        return captured_count
    
    def pass_turn(self):
        """Pass the current turn."""
        self.current_player = Stone.WHITE if self.current_player == Stone.BLACK else Stone.BLACK
        self.move_count += 1
        self.last_move = None
    
    def reset(self):
        """Reset the board."""
        self.board = [[Stone.EMPTY for _ in range(self.size)] for _ in range(self.size)]
        self.current_player = Stone.BLACK
        self.move_count = 0
        self.captured_stones = {Stone.BLACK: 0, Stone.WHITE: 0}
        self.last_move = None
    
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