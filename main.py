"""
Go Game - Graphical Board Implementation using Pygame
A visual Go (Weiqi/Baduk) board for AI agent gameplay.
"""

import pygame
import sys
from enum import Enum
from typing import Tuple, List, Set


# Colors
BACKGROUND_COLOR = (180, 180, 180)      # Light grey
LINE_COLOR = (51, 51, 51)                # Dark grey
BLACK_STONE_COLOR = (26, 26, 26)         # Black
WHITE_STONE_COLOR = (245, 245, 245)      # White
STAR_POINT_COLOR = (51, 51, 51)          # Dark grey


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
    
    def reset(self):
        """Reset the board."""
        self.board = [[Stone.EMPTY for _ in range(self.size)] for _ in range(self.size)]
        self.current_player = Stone.BLACK
        self.move_count = 0
        self.captured_stones = {Stone.BLACK: 0, Stone.WHITE: 0}


class GoGame:
    """Main Go game class with Pygame rendering."""
    
    BOARD_SIZE = 19
    CELL_SIZE = 32
    STONE_RADIUS = 14
    PADDING = 40
    
    # Star points (hoshi) for 19x19 board
    STAR_POINTS = [(3, 3), (3, 9), (3, 15),
                   (9, 3), (9, 9), (9, 15),
                   (15, 3), (15, 9), (15, 15)]
    
    def __init__(self):
        """Initialize Pygame and the game."""
        pygame.init()
        
        # Calculate window size
        self.window_size = (
            self.CELL_SIZE * (self.BOARD_SIZE - 1) + self.PADDING * 2 + 200
        )
        
        # Create window
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("Go Game - AI Playground")
        
        # Game state
        self.game = GoBoard(size=self.BOARD_SIZE)
        
        # Fonts
        self.font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 22)
        
        # Colors
        self.bg_color = BACKGROUND_COLOR
        self.line_color = LINE_COLOR
        
    def draw_board(self):
        """Draw the Go board grid."""
        self.screen.fill(self.bg_color)
        
        # Draw the board background with border
        board_start = self.PADDING
        board_end = self.PADDING + (self.BOARD_SIZE - 1) * self.CELL_SIZE
        
        # Draw board outline
        pygame.draw.rect(
            self.screen, 
            self.line_color,
            (board_start - 3, board_start - 3, 
             board_end - board_start + 6, board_end - board_start + 6),
            2
        )
        
        # Draw vertical lines
        for i in range(self.BOARD_SIZE):
            x = self.PADDING + i * self.CELL_SIZE
            pygame.draw.line(
                self.screen,
                self.line_color,
                (x, self.PADDING),
                (x, board_end),
                2
            )
        
        # Draw horizontal lines
        for i in range(self.BOARD_SIZE):
            y = self.PADDING + i * self.CELL_SIZE
            pygame.draw.line(
                self.screen,
                self.line_color,
                (self.PADDING, y),
                (board_end, y),
                2
            )
        
        # Draw star points (hoshi)
        for row, col in self.STAR_POINTS:
            x = self.PADDING + col * self.CELL_SIZE
            y = self.PADDING + row * self.CELL_SIZE
            pygame.draw.circle(
                self.screen,
                STAR_POINT_COLOR,
                (x, y),
                5
            )
        
        # Draw coordinates
        self.draw_coordinates()
        
        # Draw stones
        self.draw_stones()
        
        # Draw UI
        self.draw_ui()
    
    def draw_coordinates(self):
        """Draw board coordinates (A-T, 1-19)."""
        letters = "ABCDEFGHJKLMNOPQRST"
        
        # Column letters (A-T, skipping I)
        for i in range(self.BOARD_SIZE):
            x = self.PADDING + i * self.CELL_SIZE
            letter = letters[i] if i < len(letters) else ""
            text = self.small_font.render(letter, True, self.line_color)
            text_rect = text.get_rect(center=(x, self.PADDING - 15))
            self.screen.blit(text, text_rect)
            text_rect = text.get_rect(center=(x, self.PADDING + (self.BOARD_SIZE - 1) * self.CELL_SIZE + 15))
            self.screen.blit(text, text_rect)
        
        # Row numbers (1-19)
        for i in range(self.BOARD_SIZE):
            y = self.PADDING + i * self.CELL_SIZE
            number = str(i + 1)
            text = self.small_font.render(number, True, self.line_color)
            text_rect = text.get_rect(center=(self.PADDING - 15, y))
            self.screen.blit(text, text_rect)
            text_rect = text.get_rect(center=(self.PADDING + (self.BOARD_SIZE - 1) * self.CELL_SIZE + 15, y))
            self.screen.blit(text, text_rect)
    
    def draw_stones(self):
        """Draw all stones on the board."""
        for row in range(self.BOARD_SIZE):
            for col in range(self.BOARD_SIZE):
                if self.game.board[row][col] != Stone.EMPTY:
                    x = self.PADDING + col * self.CELL_SIZE
                    y = self.PADDING + row * self.CELL_SIZE
                    
                    stone_color = (
                        BLACK_STONE_COLOR 
                        if self.game.board[row][col] == Stone.BLACK 
                        else WHITE_STONE_COLOR
                    )
                    
                    # Draw stone with shadow
                    shadow_offset = 2
                    pygame.draw.circle(
                        self.screen,
                        (100, 100, 100),
                        (x + shadow_offset, y + shadow_offset),
                        self.STONE_RADIUS
                    )
                    
                    # Draw main stone
                    pygame.draw.circle(
                        self.screen,
                        stone_color,
                        (x, y),
                        self.STONE_RADIUS
                    )
                    
                    # Add highlight for white stones
                    if self.game.board[row][col] == Stone.WHITE:
                        pygame.draw.circle(
                            self.screen,
                            (255, 255, 255),
                            (x - 4, y - 4),
                            4
                        )
    
    def draw_ui(self):
        """Draw the user interface panel."""
        panel_x = self.PADDING + (self.BOARD_SIZE - 1) * self.CELL_SIZE + 30
        
        # Current player
        player_text = "Current: BLACK" if self.game.current_player == Stone.BLACK else "Current: WHITE"
        player_color = BLACK_STONE_COLOR if self.game.current_player == Stone.BLACK else (180, 180, 180)
        
        text = self.font.render(player_text, True, player_color)
        self.screen.blit(text, (panel_x, 30))
        
        # Move count
        move_text = f"Moves: {self.game.move_count}"
        text = self.small_font.render(move_text, True, self.line_color)
        self.screen.blit(text, (panel_x, 70))
        
        # Captured stones
        cap_text = f"Captured:"
        text = self.small_font.render(cap_text, True, self.line_color)
        self.screen.blit(text, (panel_x, 110))
        
        black_cap_text = f"  Black: {self.game.captured_stones[Stone.BLACK]}"
        text = self.small_font.render(black_cap_text, True, (50, 50, 50))
        self.screen.blit(text, (panel_x, 135))
        
        white_cap_text = f"  White: {self.game.captured_stones[Stone.WHITE]}"
        text = self.small_font.render(white_cap_text, True, (100, 100, 100))
        self.screen.blit(text, (panel_x, 160))
        
        # Instructions
        instr_y = 220
        instr_lines = [
            "Controls:",
            "Click - Place stone",
            "P - Pass turn",
            "N - New game",
            "Q - Quit"
        ]
        
        for line in instr_lines:
            text = self.small_font.render(line, True, self.line_color)
            self.screen.blit(text, (panel_x, instr_y))
            instr_y += 25
    
    def get_board_position(self, screen_pos: Tuple[int, int]) -> Tuple[int, int]:
        """Convert screen position to board coordinates."""
        x, y = screen_pos
        
        # Calculate board bounds
        board_start = self.PADDING
        board_end = self.PADDING + (self.BOARD_SIZE - 1) * self.CELL_SIZE
        
        # Check if click is within board area
        if x < board_start or x > board_end or y < board_start or y > board_end:
            return (-1, -1)
        
        # Convert to board coordinates with snapping
        col = round((x - self.PADDING) / self.CELL_SIZE)
        row = round((y - self.PADDING) / self.CELL_SIZE)
        
        # Validate position
        if not self.game.is_valid_position(row, col):
            return (-1, -1)
        
        # Check if click is close enough to intersection
        actual_x = self.PADDING + col * self.CELL_SIZE
        actual_y = self.PADDING + row * self.CELL_SIZE
        
        distance = ((x - actual_x) ** 2 + (y - actual_y) ** 2) ** 0.5
        
        if distance > self.CELL_SIZE / 2:
            return (-1, -1)
        
        return (row, col)
    
    def run(self):
        """Main game loop."""
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        row, col = self.get_board_position(event.pos)
                        if row >= 0 and col >= 0:
                            self.game.place_stone(row, col)
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_p:
                        self.game.pass_turn()
                    elif event.key == pygame.K_n:
                        self.game.reset()
                    elif event.key == pygame.K_q:
                        running = False
            
            # Draw everything
            self.draw_board()
            pygame.display.flip()
        
        pygame.quit()
        sys.exit()


def main():
    """Main function to start the Go game."""
    game = GoGame()
    game.run()


if __name__ == "__main__":
    main()

