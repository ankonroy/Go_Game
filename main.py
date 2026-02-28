"""
Go Game - Graphical Board Implementation using Pygame
A visual Go (Weiqi/Baduk) board for AI agent gameplay.
Supports: Human vs Human, Human vs AI, AI vs AI modes.
Win condition: First to capture wins.
"""

import pygame
import sys
from enum import Enum
from typing import Tuple, List, Optional

from game_models import GoBoard, Stone
from AI_Agent_Albi.minmax import MinimaxAI as AlbiAI
from AI_Agent_Ankon.minmax import MinimaxAI as AnkonAI


# Colors
BACKGROUND_COLOR = (180, 180, 180)      # Light grey
LINE_COLOR = (51, 51, 51)                # Dark grey
BLACK_STONE_COLOR = (26, 26, 26)         # Black
WHITE_STONE_COLOR = (245, 245, 245)      # White
STAR_POINT_COLOR = (51, 51, 51)          # Dark grey
HIGHLIGHT_COLOR = (255, 215, 0)          # Gold for last move


class GameMode(Enum):
    """Game modes available."""
    HUMAN_VS_HUMAN = 1
    HUMAN_VS_AI = 2
    AI_VS_AI = 3


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
    
    def __init__(self, mode: GameMode = GameMode.HUMAN_VS_HUMAN):
        """Initialize Pygame and the game."""
        pygame.init()
        
        # Game mode
        self.mode = mode
        self.ai_depth = 2  # Depth for minimax search
        
        # Calculate window size
        self.window_size = (
            self.CELL_SIZE * (self.BOARD_SIZE - 1) + self.PADDING * 2 + 200
        )
        
        # Create window
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("Go Game - AI Playground")
        
        # Game state
        self.game = GoBoard(size=self.BOARD_SIZE)
        
        # AI agents
        self.ai_white = None
        self.ai_black = None
        self._setup_ai_agents()
        
        # Game state
        self.game_over = False
        self.winner = None
        self.ai_thinking = False
        self.ai_delay = 500  # Delay between AI moves in ms (for AI vs AI)
        self.last_ai_move_time = 0
        
        # Fonts
        self.font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 22)
        self.large_font = pygame.font.Font(None, 48)
        
        # Colors
        self.bg_color = BACKGROUND_COLOR
        self.line_color = LINE_COLOR
    
    def _setup_ai_agents(self):
        """Setup AI agents based on game mode."""
        if self.mode == GameMode.HUMAN_VS_AI:
            # Human plays as Black, AI as White
            self.ai_white = AnkonAI(player=Stone.WHITE, depth=self.ai_depth)
            self.ai_black = None
        elif self.mode == GameMode.AI_VS_AI:
            # Both AI agents
            self.ai_black = AlbiAI(player=Stone.BLACK, depth=self.ai_depth)
            self.ai_white = AnkonAI(player=Stone.WHITE, depth=self.ai_depth)
        else:
            # Human vs Human
            self.ai_black = None
            self.ai_white = None
    
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
        
        # Draw game over overlay
        if self.game_over:
            self.draw_game_over()
    
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
                    
                    # Highlight last move
                    if self.game.last_move == (row, col):
                        pygame.draw.circle(
                            self.screen,
                            HIGHLIGHT_COLOR,
                            (x, y),
                            self.STONE_RADIUS + 4,
                            2
                        )
    
    def draw_ui(self):
        """Draw the user interface panel."""
        panel_x = self.PADDING + (self.BOARD_SIZE - 1) * self.CELL_SIZE + 30
        
        # Mode display
        mode_text = "Mode: "
        if self.mode == GameMode.HUMAN_VS_HUMAN:
            mode_text += "Human vs Human"
        elif self.mode == GameMode.HUMAN_VS_AI:
            mode_text += "Human vs AI"
        else:
            mode_text += "AI vs AI"
        
        text = self.small_font.render(mode_text, True, self.line_color)
        self.screen.blit(text, (panel_x, 10))
        
        # Current player
        player_text = "Current: BLACK" if self.game.current_player == Stone.BLACK else "Current: WHITE"
        player_color = BLACK_STONE_COLOR if self.game.current_player == Stone.BLACK else (180, 180, 180)
        
        text = self.font.render(player_text, True, player_color)
        self.screen.blit(text, (panel_x, 40))
        
        # Move count
        move_text = f"Moves: {self.game.move_count}"
        text = self.small_font.render(move_text, True, self.line_color)
        self.screen.blit(text, (panel_x, 80))
        
        # Captured stones
        cap_text = f"Captured:"
        text = self.small_font.render(cap_text, True, self.line_color)
        self.screen.blit(text, (panel_x, 120))
        
        black_cap_text = f"  Black: {self.game.captured_stones[Stone.BLACK]}"
        text = self.small_font.render(black_cap_text, True, (50, 50, 50))
        self.screen.blit(text, (panel_x, 145))
        
        white_cap_text = f"  White: {self.game.captured_stones[Stone.WHITE]}"
        text = self.small_font.render(white_cap_text, True, (100, 100, 100))
        self.screen.blit(text, (panel_x, 170))
        
        # Win condition info
        win_text = "First to capture wins!"
        text = self.small_font.render(win_text, True, (150, 50, 50))
        self.screen.blit(text, (panel_x, 210))
        
        # Instructions
        instr_y = 250
        instr_lines = [
            "Controls:",
            "Click - Place stone",
            "P - Pass turn",
            "N - New game",
            "Q - Quit",
            "M - Change mode"
        ]
        
        for line in instr_lines:
            text = self.small_font.render(line, True, self.line_color)
            self.screen.blit(text, (panel_x, instr_y))
            instr_y += 25
        
        # AI thinking indicator
        if self.ai_thinking:
            think_text = "AI Thinking..."
            text = self.small_font.render(think_text, True, (200, 100, 0))
            self.screen.blit(text, (panel_x, instr_y + 10))
    
    def draw_game_over(self):
        """Draw game over overlay."""
        # Semi-transparent overlay
        overlay = pygame.Surface((self.window_size, self.window_size), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 128))
        self.screen.blit(overlay, (0, 0))
        
        # Winner text
        if self.winner == Stone.BLACK:
            winner_text = "BLACK WINS!"
            color = BLACK_STONE_COLOR
        else:
            winner_text = "WHITE WINS!"
            color = WHITE_STONE_COLOR
        
        text = self.large_font.render(winner_text, True, color)
        text_rect = text.get_rect(center=(self.window_size // 2, self.window_size // 2 - 30))
        self.screen.blit(text, text_rect)
        
        # Restart instruction
        restart_text = self.font.render("Press N for new game", True, (255, 255, 255))
        restart_rect = restart_text.get_rect(center=(self.window_size // 2, self.window_size // 2 + 30))
        self.screen.blit(text, restart_rect)
    
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
    
    def get_ai_move(self) -> Optional[Tuple[int, int]]:
        """Get move from AI if it's AI's turn."""
        if self.mode == GameMode.HUMAN_VS_AI:
            if self.game.current_player == Stone.WHITE and self.ai_white:
                return self.ai_white.get_best_move(self.game)
        elif self.mode == GameMode.AI_VS_AI:
            if self.game.current_player == Stone.BLACK and self.ai_black:
                return self.ai_black.get_best_move(self.game)
            elif self.game.current_player == Stone.WHITE and self.ai_white:
                return self.ai_white.get_best_move(self.game)
        return None
    
    def is_human_turn(self) -> bool:
        """Check if it's a human's turn."""
        if self.mode == GameMode.HUMAN_VS_HUMAN:
            return True
        elif self.mode == GameMode.HUMAN_VS_AI:
            return self.game.current_player == Stone.BLACK
        else:  # AI vs AI
            return False
    
    def check_game_over(self):
        """Check if game is over."""
        winner = self.game.get_winner()
        if winner:
            self.game_over = True
            self.winner = winner
    
    def run(self):
        """Main game loop."""
        running = True
        clock = pygame.time.Clock()
        
        while running:
            current_time = pygame.time.get_ticks()
            
            # AI move handling for AI vs AI mode
            if self.mode == GameMode.AI_VS_AI and not self.game_over:
                if current_time - self.last_ai_move_time > self.ai_delay:
                    ai_move = self.get_ai_move()
                    if ai_move:
                        self.game.place_stone(ai_move[0], ai_move[1])
                        self.check_game_over()
                    self.last_ai_move_time = current_time
            
            # AI move handling for Human vs AI mode
            elif self.mode == GameMode.HUMAN_VS_AI and not self.game_over:
                if self.game.current_player == Stone.WHITE:
                    # AI plays White
                    ai_move = self.get_ai_move()
                    if ai_move:
                        self.game.place_stone(ai_move[0], ai_move[1])
                        self.check_game_over()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.MOUSEBUTTONDOWN and not self.game_over:
                    if event.button == 1:  # Left click
                        if self.is_human_turn():
                            row, col = self.get_board_position(event.pos)
                            if row >= 0 and col >= 0:
                                if self.game.place_stone(row, col):
                                    self.check_game_over()
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_p:
                        self.game.pass_turn()
                    elif event.key == pygame.K_n:
                        # Reset game
                        self.game.reset()
                        self.game_over = False
                        self.winner = None
                        self._setup_ai_agents()
                    elif event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_m:
                        # Cycle through modes
                        if self.mode == GameMode.HUMAN_VS_HUMAN:
                            self.mode = GameMode.HUMAN_VS_AI
                        elif self.mode == GameMode.HUMAN_VS_AI:
                            self.mode = GameMode.AI_VS_AI
                        else:
                            self.mode = GameMode.HUMAN_VS_HUMAN
                        # Reset with new mode
                        self.game.reset()
                        self.game_over = False
                        self.winner = None
                        self._setup_ai_agents()
            
            # Draw everything
            self.draw_board()
            pygame.display.flip()
            clock.tick(30)  # 30 FPS
        
        pygame.quit()
        sys.exit()


def select_game_mode() -> GameMode:
    """
    Display a mode selection screen and return selected mode.
    """
    pygame.init()
    
    # Window setup
    screen = pygame.display.set_mode((600, 400))
    pygame.display.set_caption("Go Game - Select Mode")
    
    font = pygame.font.Font(None, 36)
    small_font = pygame.font.Font(None, 24)
    
    running = True
    selected_mode = GameMode.HUMAN_VS_HUMAN
    
    while running:
        screen.fill(BACKGROUND_COLOR)
        
        # Title
        title = font.render("Go Game - Select Mode", True, LINE_COLOR)
        title_rect = title.get_rect(center=(300, 50))
        screen.blit(title, title_rect)
        
        # Mode options
        modes = [
            ("1 - Human vs Human", GameMode.HUMAN_VS_HUMAN),
            ("2 - Human vs AI", GameMode.HUMAN_VS_AI),
            ("3 - AI vs AI", GameMode.AI_VS_AI),
        ]
        
        y_pos = 120
        for text, mode in modes:
            if mode == selected_mode:
                # Highlight selected
                color = (200, 50, 50)
                indicator = ">>> "
            else:
                color = LINE_COLOR
                indicator = "    "
            
            mode_text = font.render(indicator + text, True, color)
            mode_rect = mode_text.get_rect(center=(300, y_pos))
            screen.blit(mode_text, mode_rect)
            y_pos += 60
        
        # Instructions
        instr = small_font.render("Use UP/DOWN arrows to select, ENTER to confirm", True, LINE_COLOR)
        instr_rect = instr.get_rect(center=(300, 350))
        screen.blit(instr, instr_rect)
        
        # Controls info
        controls = [
            "In game: Click=place, P=pass, N=new, Q=quit, M=change mode"
        ]
        y = 280
        for c in controls:
            ctrl = small_font.render(c, True, (100, 100, 100))
            ctrl_rect = ctrl.get_rect(center=(300, y))
            screen.blit(ctrl, ctrl_rect)
            y += 25
        
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    selected_mode = GameMode.HUMAN_VS_HUMAN
                elif event.key == pygame.K_2:
                    selected_mode = GameMode.HUMAN_VS_AI
                elif event.key == pygame.K_3:
                    selected_mode = GameMode.AI_VS_AI
                elif event.key == pygame.K_UP:
                    if selected_mode == GameMode.AI_VS_AI:
                        selected_mode = GameMode.HUMAN_VS_AI
                    elif selected_mode == GameMode.HUMAN_VS_AI:
                        selected_mode = GameMode.HUMAN_VS_HUMAN
                elif event.key == pygame.K_DOWN:
                    if selected_mode == GameMode.HUMAN_VS_HUMAN:
                        selected_mode = GameMode.HUMAN_VS_AI
                    elif selected_mode == GameMode.HUMAN_VS_AI:
                        selected_mode = GameMode.AI_VS_AI
                elif event.key == pygame.K_RETURN:
                    running = False
    
    pygame.quit()
    return selected_mode


def main():
    """Main function to start the Go game."""
    # Show mode selection
    mode = select_game_mode()
    
    # Start game with selected mode
    game = GoGame(mode=mode)
    game.run()


if __name__ == "__main__":
    main()