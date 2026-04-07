"""
Go Game - Graphical Board Implementation using Pygame
A visual Go (Weiqi/Baduk) board for AI agent gameplay.
Supports: Human vs Human, Human vs AI, AI vs AI modes.
Win condition: First to capture wins.
"""

import pygame
import sys
import time  # Add time for better control
from enum import Enum
from typing import Tuple, List, Optional

from game_models import GoBoard, Stone

# AI Imports - Albi
from AI_Agent_Albi.minmax import MinimaxAI as AlbiMinimax
from AI_Agent_Albi.montecarlo import MonteCarloAI as AlbiMonteCarlo

# AI Imports - Ankon
from AI_Agent_Ankon.minmax import MinimaxAI as AnkonMinimax
from AI_Agent_Ankon.montecarlo import MonteCarloAI as AnkonMonteCarlo


# Colors
BACKGROUND_COLOR = (180, 180, 180)
LINE_COLOR = (51, 51, 51)
BLACK_STONE_COLOR = (26, 26, 26)
WHITE_STONE_COLOR = (245, 245, 245)
STAR_POINT_COLOR = (51, 51, 51)
HIGHLIGHT_COLOR = (255, 215, 0)
BUTTON_COLOR = (100, 100, 200)
BUTTON_HOVER_COLOR = (120, 120, 220)


class GameMode(Enum):
    HUMAN_VS_HUMAN = 1
    HUMAN_VS_AI = 2
    AI_VS_AI = 3


class AIAlgorithm(Enum):
    MINIMAX = 1
    MONTECARLO = 2


class AIPersonality(Enum):
    ALBI = 1
    ANKON = 2


class Button:
    def __init__(self, x, y, width, height, text, color=BUTTON_COLOR, 
                 hover_color=BUTTON_HOVER_COLOR, text_color=(255, 255, 255)):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.text_color = text_color
        self.is_hovered = False
    
    def draw(self, screen, font):
        color = self.hover_color if self.is_hovered else self.color
        pygame.draw.rect(screen, color, self.rect)
        pygame.draw.rect(screen, (0, 0, 0), self.rect, 2)
        
        text_surf = font.render(self.text, True, self.text_color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)
    
    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.is_hovered:
                return True
        return False


class GoGame:
    BOARD_SIZE = 19
    CELL_SIZE = 32
    STONE_RADIUS = 14
    PADDING = 40
    
    STAR_POINTS = [(3, 3), (3, 9), (3, 15),
                   (9, 3), (9, 9), (9, 15),
                   (15, 3), (15, 9), (15, 15)]
    
    def __init__(self):
        pygame.init()
        
        # Game settings
        self.mode = GameMode.HUMAN_VS_HUMAN
        self.ai_algorithm_black = AIAlgorithm.MINIMAX
        self.ai_algorithm_white = AIAlgorithm.MINIMAX
        self.ai_personality_black = AIPersonality.ALBI
        self.ai_personality_white = AIPersonality.ANKON
        self.ai_depth = 2  # Reduced depth for safety
        self.mcts_simulations = 500  # Reduced simulations
        self.mcts_time_limit = 1.0  # Reduced time limit
        
        # Calculate window size
        self.window_size = (
            self.CELL_SIZE * (self.BOARD_SIZE - 1) + self.PADDING * 2 + 300
        )
        
        # Create window
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("Go Game - AI Playground")
        
        # Game state
        self.game = GoBoard(size=self.BOARD_SIZE)
        
        # AI agents
        self.ai_black = None
        self.ai_white = None
        self._setup_ai_agents()
        
        # Game state
        self.game_over = False
        self.winner = None
        self.ai_thinking = False
        
        # FIXED: Better AI move timing
        self.ai_move_delay = 1.0  # seconds between AI moves
        self.last_ai_move_time = time.time()
        self.ai_move_pending = False
        
        # Fonts
        self.font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 22)
        self.large_font = pygame.font.Font(None, 48)
        self.button_font = pygame.font.Font(None, 24)
        
        # Colors
        self.bg_color = BACKGROUND_COLOR
        self.line_color = LINE_COLOR
        
        # Settings panel buttons
        self._create_settings_buttons()
        
        # FIXED: Add frame rate limiter
        self.clock = pygame.time.Clock()
        self.running = True
    
    def _create_settings_buttons(self):
        """Create buttons for settings panel."""
        panel_x = self.PADDING + (self.BOARD_SIZE - 1) * self.CELL_SIZE + 30
        button_width = 200
        small_button_width = 100
        button_height = 30
        y_start = 250
        
        self.buttons = []
        
        # Mode buttons
        self.mode_buttons = []
        modes = [
            ("Human vs Human", GameMode.HUMAN_VS_HUMAN),
            ("Human vs AI", GameMode.HUMAN_VS_AI),
            ("AI vs AI", GameMode.AI_VS_AI)
        ]
        
        for i, (text, mode) in enumerate(modes):
            btn = Button(panel_x, y_start + i * 35, button_width, button_height, 
                        text, color=(80, 80, 80), hover_color=(100, 100, 100))
            btn.mode = mode
            self.mode_buttons.append(btn)
        
        algos = [
            ("Minimax", AIAlgorithm.MINIMAX),
            ("Monte Carlo", AIAlgorithm.MONTECARLO)
        ]

        personalities = [
            ("Albi", AIPersonality.ALBI),
            ("Ankon", AIPersonality.ANKON)
        ]

        # Black AI controls
        y_start += 120
        self.black_algorithm_buttons = []
        for i, (text, algo) in enumerate(algos):
            btn = Button(panel_x + i * 105, y_start + 30, small_button_width, button_height,
                        text, color=(80, 80, 80), hover_color=(100, 100, 100))
            btn.algorithm = algo
            self.black_algorithm_buttons.append(btn)
        
        y_start += 70
        self.black_personality_buttons = []
        for i, (text, pers) in enumerate(personalities):
            btn = Button(panel_x + i * 105, y_start + 30, small_button_width, button_height,
                        text, color=(80, 80, 80), hover_color=(100, 100, 100))
            btn.personality = pers
            self.black_personality_buttons.append(btn)
        
        # White AI controls
        y_start += 95
        self.white_algorithm_buttons = []
        for i, (text, algo) in enumerate(algos):
            btn = Button(panel_x + i * 105, y_start + 30, small_button_width, button_height,
                        text, color=(80, 80, 80), hover_color=(100, 100, 100))
            btn.algorithm = algo
            self.white_algorithm_buttons.append(btn)

        y_start += 70
        self.white_personality_buttons = []
        for i, (text, pers) in enumerate(personalities):
            btn = Button(panel_x + i * 105, y_start + 30, small_button_width, button_height,
                        text, color=(80, 80, 80), hover_color=(100, 100, 100))
            btn.personality = pers
            self.white_personality_buttons.append(btn)
        
        # Apply settings button
        y_start += 95
        self.apply_button = Button(panel_x, y_start, button_width, button_height,
                                  "Apply Settings", color=(50, 150, 50), 
                                  hover_color=(70, 170, 70))
    
    def _setup_ai_agents(self):
        """Setup AI agents based on current settings."""
        self.ai_black = None
        self.ai_white = None
        
        # Create Black AI if needed
        if self.mode != GameMode.HUMAN_VS_HUMAN:
            if self.mode == GameMode.HUMAN_VS_AI:
                # Human plays Black, AI plays White
                self._create_ai(Stone.WHITE)
            elif self.mode == GameMode.AI_VS_AI:
                # Both AIs
                self._create_ai(Stone.BLACK)
                self._create_ai(Stone.WHITE)
    
    def _create_ai(self, player: Stone):
        """Create AI for specified player."""
        algorithm = (self.ai_algorithm_black if player == Stone.BLACK
                     else self.ai_algorithm_white)
        personality = (self.ai_personality_black if player == Stone.BLACK 
                      else self.ai_personality_white)
        
        try:
            if algorithm == AIAlgorithm.MINIMAX:
                if personality == AIPersonality.ALBI:
                    ai_class = AlbiMinimax
                else:
                    ai_class = AnkonMinimax
                
                ai = ai_class(player=player, depth=self.ai_depth)
                
            else:  # Monte Carlo
                if personality == AIPersonality.ALBI:
                    ai_class = AlbiMonteCarlo
                else:
                    ai_class = AnkonMonteCarlo
                
                ai = ai_class(player=player, simulations=self.mcts_simulations, 
                             time_limit=self.mcts_time_limit)
            
            if player == Stone.BLACK:
                self.ai_black = ai
            else:
                self.ai_white = ai
        except Exception as e:
            print(f"Error creating AI: {e}")
    
    def draw_board(self):
        """Draw the Go board grid."""
        try:
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
            
            # Draw star points
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
            
            # Draw settings panel
            self.draw_settings_panel()
            
            # Draw game over overlay
            if self.game_over:
                self.draw_game_over()
                
        except Exception as e:
            print(f"Error drawing board: {e}")
    
    def draw_coordinates(self):
        """Draw board coordinates."""
        try:
            letters = "ABCDEFGHJKLMNOPQRST"
            
            for i in range(self.BOARD_SIZE):
                x = self.PADDING + i * self.CELL_SIZE
                letter = letters[i] if i < len(letters) else ""
                text = self.small_font.render(letter, True, self.line_color)
                text_rect = text.get_rect(center=(x, self.PADDING - 15))
                self.screen.blit(text, text_rect)
                text_rect = text.get_rect(center=(x, self.PADDING + (self.BOARD_SIZE - 1) * self.CELL_SIZE + 15))
                self.screen.blit(text, text_rect)
            
            for i in range(self.BOARD_SIZE):
                y = self.PADDING + i * self.CELL_SIZE
                number = str(i + 1)
                text = self.small_font.render(number, True, self.line_color)
                text_rect = text.get_rect(center=(self.PADDING - 15, y))
                self.screen.blit(text, text_rect)
                text_rect = text.get_rect(center=(self.PADDING + (self.BOARD_SIZE - 1) * self.CELL_SIZE + 15, y))
                self.screen.blit(text, text_rect)
        except Exception as e:
            print(f"Error drawing coordinates: {e}")
    
    def draw_stones(self):
        """Draw all stones on the board."""
        try:
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
                        
                        pygame.draw.circle(
                            self.screen,
                            stone_color,
                            (x, y),
                            self.STONE_RADIUS
                        )
                        
                        if self.game.board[row][col] == Stone.WHITE:
                            pygame.draw.circle(
                                self.screen,
                                (255, 255, 255),
                                (x - 4, y - 4),
                                4
                            )
                        
                        if self.game.last_move == (row, col):
                            pygame.draw.circle(
                                self.screen,
                                HIGHLIGHT_COLOR,
                                (x, y),
                                self.STONE_RADIUS + 4,
                                2
                            )
        except Exception as e:
            print(f"Error drawing stones: {e}")
    
    def draw_settings_panel(self):
        """Draw the settings panel with buttons."""
        try:
            panel_x = self.PADDING + (self.BOARD_SIZE - 1) * self.CELL_SIZE + 30
            
            pygame.draw.rect(self.screen, (220, 220, 220), 
                            (panel_x - 10, 240, 260, 510))
            pygame.draw.rect(self.screen, self.line_color, 
                            (panel_x - 10, 240, 260, 510), 2)
            
            titles = [
                ("Game Mode", 245),
                ("Black AI", 365),
                ("White AI", 530)
            ]
            
            for title, y_pos in titles:
                text = self.small_font.render(title, True, self.line_color)
                self.screen.blit(text, (panel_x, y_pos))

            labels = [
                ("Algorithm", 395),
                ("Author", 465),
                ("Algorithm", 560),
                ("Author", 630)
            ]

            for label, y_pos in labels:
                text = self.small_font.render(label, True, self.line_color)
                self.screen.blit(text, (panel_x, y_pos))
            
            # Draw mode buttons
            for btn in self.mode_buttons:
                if btn.mode == self.mode:
                    btn.color = (50, 150, 50)
                    btn.hover_color = (70, 170, 70)
                else:
                    btn.color = (80, 80, 80)
                    btn.hover_color = (100, 100, 100)
                btn.draw(self.screen, self.small_font)
            
            # Draw Black AI algorithm buttons
            for btn in self.black_algorithm_buttons:
                if btn.algorithm == self.ai_algorithm_black:
                    btn.color = (50, 150, 50)
                    btn.hover_color = (70, 170, 70)
                else:
                    btn.color = (80, 80, 80)
                    btn.hover_color = (100, 100, 100)
                btn.draw(self.screen, self.small_font)
            
            # Draw Black AI author buttons
            for btn in self.black_personality_buttons:
                if btn.personality == self.ai_personality_black:
                    btn.color = (50, 150, 50)
                    btn.hover_color = (70, 170, 70)
                else:
                    btn.color = (80, 80, 80)
                    btn.hover_color = (100, 100, 100)
                btn.draw(self.screen, self.small_font)
            
            # Draw White AI algorithm buttons
            for btn in self.white_algorithm_buttons:
                if btn.algorithm == self.ai_algorithm_white:
                    btn.color = (50, 150, 50)
                    btn.hover_color = (70, 170, 70)
                else:
                    btn.color = (80, 80, 80)
                    btn.hover_color = (100, 100, 100)
                btn.draw(self.screen, self.small_font)

            # Draw White AI author buttons
            for btn in self.white_personality_buttons:
                if btn.personality == self.ai_personality_white:
                    btn.color = (50, 150, 50)
                    btn.hover_color = (70, 170, 70)
                else:
                    btn.color = (80, 80, 80)
                    btn.hover_color = (100, 100, 100)
                btn.draw(self.screen, self.small_font)
            
            self.apply_button.draw(self.screen, self.small_font)
        except Exception as e:
            print(f"Error drawing settings panel: {e}")
    
    def draw_ui(self):
        """Draw the user interface panel."""
        try:
            panel_x = self.PADDING + (self.BOARD_SIZE - 1) * self.CELL_SIZE + 30
            
            player_text = "Current: BLACK" if self.game.current_player == Stone.BLACK else "Current: WHITE"
            player_color = BLACK_STONE_COLOR if self.game.current_player == Stone.BLACK else (100, 100, 100)
            
            text = self.font.render(player_text, True, player_color)
            self.screen.blit(text, (panel_x, 10))
            
            move_text = f"Moves: {self.game.move_count}"
            text = self.small_font.render(move_text, True, self.line_color)
            self.screen.blit(text, (panel_x, 50))
            
            cap_text = f"Captured:"
            text = self.small_font.render(cap_text, True, self.line_color)
            self.screen.blit(text, (panel_x, 80))
            
            black_cap_text = f"  Black: {self.game.captured_stones[Stone.BLACK]}"
            text = self.small_font.render(black_cap_text, True, (50, 50, 50))
            self.screen.blit(text, (panel_x, 105))
            
            white_cap_text = f"  White: {self.game.captured_stones[Stone.WHITE]}"
            text = self.small_font.render(white_cap_text, True, (100, 100, 100))
            self.screen.blit(text, (panel_x, 130))
            
            win_text = "First to capture wins!"
            text = self.small_font.render(win_text, True, (150, 50, 50))
            self.screen.blit(text, (panel_x, 165))
            
            instr_y = 770
            instr_lines = [
                "Controls:",
                "Click - Place stone",
                "P - Pass turn",
                "N - New game",
                "R - Reset settings",
                "Q - Quit"
            ]
            
            for line in instr_lines:
                text = self.small_font.render(line, True, self.line_color)
                self.screen.blit(text, (panel_x, instr_y))
                instr_y += 25
            
            if self.ai_thinking:
                think_text = "AI Thinking..."
                text = self.small_font.render(think_text, True, (200, 100, 0))
                self.screen.blit(text, (panel_x, instr_y + 10))
        except Exception as e:
            print(f"Error drawing UI: {e}")
    
    def draw_game_over(self):
        """Draw game over overlay."""
        try:
            overlay = pygame.Surface((self.window_size, self.window_size), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))
            
            if self.winner == Stone.BLACK:
                winner_text = "BLACK WINS!"
                color = BLACK_STONE_COLOR
            else:
                winner_text = "WHITE WINS!"
                color = WHITE_STONE_COLOR
            
            text = self.large_font.render(winner_text, True, color)
            text_rect = text.get_rect(center=(self.window_size // 2, self.window_size // 2 - 30))
            self.screen.blit(text, text_rect)
            
            restart_text = self.font.render("Press N for new game", True, (255, 255, 255))
            restart_rect = restart_text.get_rect(center=(self.window_size // 2, self.window_size // 2 + 30))
            self.screen.blit(restart_text, restart_rect)
        except Exception as e:
            print(f"Error drawing game over: {e}")
    
    def get_board_position(self, screen_pos: Tuple[int, int]) -> Tuple[int, int]:
        """Convert screen position to board coordinates."""
        try:
            x, y = screen_pos
            
            board_start = self.PADDING
            board_end = self.PADDING + (self.BOARD_SIZE - 1) * self.CELL_SIZE
            
            if x < board_start or x > board_end or y < board_start or y > board_end:
                return (-1, -1)
            
            col = round((x - self.PADDING) / self.CELL_SIZE)
            row = round((y - self.PADDING) / self.CELL_SIZE)
            
            if not self.game.is_valid_position(row, col):
                return (-1, -1)
            
            actual_x = self.PADDING + col * self.CELL_SIZE
            actual_y = self.PADDING + row * self.CELL_SIZE
            
            distance = ((x - actual_x) ** 2 + (y - actual_y) ** 2) ** 0.5
            
            if distance > self.CELL_SIZE / 2:
                return (-1, -1)
            
            return (row, col)
        except Exception:
            return (-1, -1)
    
    def get_ai_move(self) -> Optional[Tuple[int, int]]:
        """Get move from AI if it's AI's turn."""
        try:
            if self.mode == GameMode.HUMAN_VS_AI:
                if self.game.current_player == Stone.WHITE and self.ai_white:
                    return self.ai_white.get_best_move(self.game)
            elif self.mode == GameMode.AI_VS_AI:
                if self.game.current_player == Stone.BLACK and self.ai_black:
                    return self.ai_black.get_best_move(self.game)
                elif self.game.current_player == Stone.WHITE and self.ai_white:
                    return self.ai_white.get_best_move(self.game)
        except Exception as e:
            print(f"Error getting AI move: {e}")
        return None
    
    def is_human_turn(self) -> bool:
        """Check if it's a human's turn."""
        if self.mode == GameMode.HUMAN_VS_HUMAN:
            return True
        elif self.mode == GameMode.HUMAN_VS_AI:
            return self.game.current_player == Stone.BLACK
        else:
            return False
    
    def check_game_over(self):
        """Check if game is over."""
        try:
            winner = self.game.get_winner()
            if winner:
                self.game_over = True
                self.winner = winner
        except Exception as e:
            print(f"Error checking game over: {e}")
    
    def handle_settings_click(self, pos):
        """Handle clicks on settings panel."""
        try:
            for btn in self.mode_buttons:
                if btn.rect.collidepoint(pos):
                    self.mode = btn.mode
                    return True
            
            for btn in self.black_algorithm_buttons:
                if btn.rect.collidepoint(pos):
                    self.ai_algorithm_black = btn.algorithm
                    return True

            for btn in self.white_algorithm_buttons:
                if btn.rect.collidepoint(pos):
                    self.ai_algorithm_white = btn.algorithm
                    return True
            
            for btn in self.black_personality_buttons:
                if btn.rect.collidepoint(pos):
                    self.ai_personality_black = btn.personality
                    return True
            
            for btn in self.white_personality_buttons:
                if btn.rect.collidepoint(pos):
                    self.ai_personality_white = btn.personality
                    return True
            
            if self.apply_button.rect.collidepoint(pos):
                self.apply_settings()
                return True
        except Exception as e:
            print(f"Error handling settings click: {e}")
        return False
    
    def apply_settings(self):
        """Apply current settings and restart game."""
        try:
            self.game.reset()
            self.game_over = False
            self.winner = None
            self._setup_ai_agents()
            self.last_ai_move_time = time.time()
        except Exception as e:
            print(f"Error applying settings: {e}")
    
    def run(self):
        """Main game loop - FIXED version with proper timing and event handling."""
        try:
            while self.running:
                # FIXED: Control frame rate to prevent CPU overload
                self.clock.tick(30)  # Limit to 30 FPS
                
                current_time = time.time()
                
                # Handle events first
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        if event.button == 1:  # Left click
                            if not self.handle_settings_click(event.pos):
                                if not self.game_over and self.is_human_turn():
                                    row, col = self.get_board_position(event.pos)
                                    if row >= 0 and col >= 0:
                                        if self.game.place_stone(row, col):
                                            self.check_game_over()
                                            self.last_ai_move_time = current_time
                    
                    elif event.type == pygame.MOUSEMOTION:
                        # Update button hover states
                        for btn in (self.mode_buttons + self.black_algorithm_buttons +
                                   self.white_algorithm_buttons +
                                   self.black_personality_buttons + 
                                   self.white_personality_buttons + [self.apply_button]):
                            btn.handle_event(event)
                    
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_p:
                            self.game.pass_turn()
                        elif event.key == pygame.K_n:
                            self.game.reset()
                            self.game_over = False
                            self.winner = None
                            self.last_ai_move_time = current_time
                        elif event.key == pygame.K_r:
                            self.mode = GameMode.HUMAN_VS_HUMAN
                            self.ai_algorithm_black = AIAlgorithm.MINIMAX
                            self.ai_algorithm_white = AIAlgorithm.MINIMAX
                            self.ai_personality_black = AIPersonality.ALBI
                            self.ai_personality_white = AIPersonality.ANKON
                            self.apply_settings()
                        elif event.key == pygame.K_q:
                            self.running = False
                
                # FIXED: AI move handling with proper timing
                if not self.game_over and self.mode != GameMode.HUMAN_VS_HUMAN:
                    # Check if it's time for an AI move
                    if current_time - self.last_ai_move_time > self.ai_move_delay:
                        self.ai_thinking = True
                        ai_move = self.get_ai_move()
                        self.ai_thinking = False
                        
                        if ai_move:
                            self.game.place_stone(ai_move[0], ai_move[1])
                            self.check_game_over()
                        
                        self.last_ai_move_time = current_time
                
                # Draw everything
                self.draw_board()
                pygame.display.flip()
            
        except Exception as e:
            print(f"Error in game loop: {e}")
        finally:
            pygame.quit()
            sys.exit()


def main():
    """Main function to start the Go game."""
    try:
        game = GoGame()
        game.run()
    except KeyboardInterrupt:
        print("\nGame interrupted by user")
        pygame.quit()
        sys.exit()
    except Exception as e:
        print(f"Error starting game: {e}")
        pygame.quit()
        sys.exit(1)


if __name__ == "__main__":
    main()
