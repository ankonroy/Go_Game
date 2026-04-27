"""
Go Game - Graphical Board Implementation using Pygame
A visual Go (Weiqi/Baduk) board for AI agent gameplay.
Supports: Human vs Human, Human vs AI, AI vs AI modes.
Win condition: First to capture wins.
"""

import pygame
import sys
import time
from enum import Enum
from typing import Tuple, Optional

from game_models import GoBoard, Stone

# AI Imports - Albi
from AI_Agent_Albi.minmax import MinimaxAI as AlbiMinimax
from AI_Agent_Albi.montecarlo import MonteCarloAI as AlbiMonteCarlo

# AI Imports - Ankon
from AI_Agent_Ankon.minmax import MinimaxAI as AnkonMinimax
from AI_Agent_Ankon.montecarlo import MonteCarloAI as AnkonMonteCarlo

# AI Import - Neural
from AI_Agent_NN.neural_agent import NeuralNetAI


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
    NEURAL = 3


class AIPersonality(Enum):
    ALBI = 1
    ANKON = 2


class Button:
    def __init__(
        self,
        x,
        y,
        width,
        height,
        text,
        color=BUTTON_COLOR,
        hover_color=BUTTON_HOVER_COLOR,
        text_color=(255, 255, 255),
    ):
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
    TOP_PADDING = 70
    BOTTOM_PADDING = 40
    SIDE_PANEL_WIDTH = 320
    SIDE_PANEL_GAP = 45
    MIN_WINDOW_HEIGHT = 730

    STAR_POINTS = [
        (3, 3), (3, 9), (3, 15),
        (9, 3), (9, 9), (9, 15),
        (15, 3), (15, 9), (15, 15)
    ]

    def __init__(self):
        pygame.init()

        # Game settings
        self.mode = GameMode.HUMAN_VS_HUMAN
        self.ai_algorithm_black = AIAlgorithm.MINIMAX
        self.ai_algorithm_white = AIAlgorithm.MINIMAX
        self.ai_personality_black = AIPersonality.ALBI
        self.ai_personality_white = AIPersonality.ANKON

        # Classical AI settings
        self.ai_depth = 2
        self.mcts_simulations = 500
        self.mcts_time_limit = 1.0

        # Neural AI settings (light UI profile)
        self.neural_model_path = "AI_Agent_NN/weights/best_model.keras"
        self.neural_top_k = 3
        self.neural_simulations = 6
        self.neural_child_top_k = 2
        self.neural_c_puct = 1.5
        self.neural_max_depth = 5

        # Layout
        board_span = self.CELL_SIZE * (self.BOARD_SIZE - 1)
        self.board_left = self.PADDING
        self.board_top = self.TOP_PADDING
        self.board_right = self.board_left + board_span
        self.board_bottom = self.board_top + board_span
        self.panel_x = self.board_right + self.SIDE_PANEL_GAP
        self.window_width = self.panel_x + self.SIDE_PANEL_WIDTH + self.PADDING
        self.window_height = max(
            self.board_bottom + self.BOTTOM_PADDING,
            self.MIN_WINDOW_HEIGHT,
        )

        # Create window
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
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

        # Keep delay unchanged
        self.ai_move_delay = 1.0
        self.last_ai_move_time = time.time()

        # Fonts
        self.font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 22)
        self.large_font = pygame.font.Font(None, 48)

        # Colors
        self.bg_color = BACKGROUND_COLOR
        self.line_color = LINE_COLOR

        # Settings panel buttons
        self._create_settings_buttons()

        # Frame rate limiter
        self.clock = pygame.time.Clock()
        self.running = True

    def _create_settings_buttons(self):
        """Create buttons for settings panel."""
        panel_x = self.panel_x
        button_width = 220
        algo_button_width = 85
        author_button_width = 100
        button_height = 30

        # Mode buttons
        self.mode_buttons = []
        modes = [
            ("Human vs Human", GameMode.HUMAN_VS_HUMAN),
            ("Human vs AI", GameMode.HUMAN_VS_AI),
            ("AI vs AI", GameMode.AI_VS_AI),
        ]

        mode_buttons_y = 240
        for i, (text, mode) in enumerate(modes):
            btn = Button(
                panel_x,
                mode_buttons_y + i * 35,
                button_width,
                button_height,
                text,
                color=(80, 80, 80),
                hover_color=(100, 100, 100),
            )
            btn.mode = mode
            self.mode_buttons.append(btn)

        algos = [
            ("Minimax", AIAlgorithm.MINIMAX),
            ("MonteCarlo", AIAlgorithm.MONTECARLO),
            ("Neural", AIAlgorithm.NEURAL),
        ]

        personalities = [
            ("Albi", AIPersonality.ALBI),
            ("Ankon", AIPersonality.ANKON),
        ]

        # Black AI controls
        self.black_algorithm_buttons = []
        for i, (text, algo) in enumerate(algos):
            btn = Button(
                panel_x + i * 90,
                395,
                algo_button_width,
                button_height,
                text,
                color=(80, 80, 80),
                hover_color=(100, 100, 100),
            )
            btn.algorithm = algo
            self.black_algorithm_buttons.append(btn)

        self.black_personality_buttons = []
        for i, (text, pers) in enumerate(personalities):
            btn = Button(
                panel_x + i * 105,
                455,
                author_button_width,
                button_height,
                text,
                color=(80, 80, 80),
                hover_color=(100, 100, 100),
            )
            btn.personality = pers
            self.black_personality_buttons.append(btn)

        # White AI controls
        self.white_algorithm_buttons = []
        for i, (text, algo) in enumerate(algos):
            btn = Button(
                panel_x + i * 90,
                550,
                algo_button_width,
                button_height,
                text,
                color=(80, 80, 80),
                hover_color=(100, 100, 100),
            )
            btn.algorithm = algo
            self.white_algorithm_buttons.append(btn)

        self.white_personality_buttons = []
        for i, (text, pers) in enumerate(personalities):
            btn = Button(
                panel_x + i * 105,
                610,
                author_button_width,
                button_height,
                text,
                color=(80, 80, 80),
                hover_color=(100, 100, 100),
            )
            btn.personality = pers
            self.white_personality_buttons.append(btn)

        # Apply settings button
        self.apply_button = Button(
            panel_x,
            655,
            button_width,
            button_height,
            "Apply Settings",
            color=(50, 150, 50),
            hover_color=(70, 170, 70),
        )

    def _setup_ai_agents(self):
        """Setup AI agents based on current settings."""
        self.ai_black = None
        self.ai_white = None

        if self.mode != GameMode.HUMAN_VS_HUMAN:
            if self.mode == GameMode.HUMAN_VS_AI:
                self._create_ai(Stone.WHITE)
            elif self.mode == GameMode.AI_VS_AI:
                self._create_ai(Stone.BLACK)
                self._create_ai(Stone.WHITE)

    def _create_ai(self, player: Stone):
        """Create AI for specified player."""
        algorithm = self.ai_algorithm_black if player == Stone.BLACK else self.ai_algorithm_white
        personality = self.ai_personality_black if player == Stone.BLACK else self.ai_personality_white

        try:
            if algorithm == AIAlgorithm.MINIMAX:
                if personality == AIPersonality.ALBI:
                    ai_class = AlbiMinimax
                else:
                    ai_class = AnkonMinimax

                ai = ai_class(player=player, depth=self.ai_depth)

            elif algorithm == AIAlgorithm.MONTECARLO:
                if personality == AIPersonality.ALBI:
                    ai_class = AlbiMonteCarlo
                else:
                    ai_class = AnkonMonteCarlo

                ai = ai_class(
                    player=player,
                    simulations=self.mcts_simulations,
                    time_limit=self.mcts_time_limit,
                )

            else:  # Neural
                ai = NeuralNetAI(
                    player=player,
                    board_size=self.BOARD_SIZE,
                    model_path=self.neural_model_path,
                    top_k=self.neural_top_k,
                    simulations=self.neural_simulations,
                    child_top_k=self.neural_child_top_k,
                    c_puct=self.neural_c_puct,
                    max_depth=self.neural_max_depth,
                )

            if player == Stone.BLACK:
                self.ai_black = ai
            else:
                self.ai_white = ai

        except Exception as e:
            print(f"Error creating AI: {e}")

    def _uses_personality(self, algorithm: AIAlgorithm) -> bool:
        """Return whether the selected AI exposes Albi/Ankon variants."""
        return algorithm != AIAlgorithm.NEURAL

    def _draw_single_neural_agent(self, buttons):
        """Draw a single non-clickable row for the shared neural agent."""
        if not buttons:
            return

        row_rect = pygame.Rect(
            buttons[0].rect.left,
            buttons[0].rect.top,
            buttons[-1].rect.right - buttons[0].rect.left,
            buttons[0].rect.height,
        )
        pygame.draw.rect(self.screen, (90, 90, 90), row_rect)
        pygame.draw.rect(self.screen, (0, 0, 0), row_rect, 2)

        text = self.small_font.render("Single Neural Agent", True, (255, 255, 255))
        text_rect = text.get_rect(center=row_rect.center)
        self.screen.blit(text, text_rect)

    def _reset_game(self, rebuild_ai: bool = False):
        """Reset the current game state and optionally recreate AI agents."""
        self.game.reset()
        self.game_over = False
        self.winner = None
        self.ai_thinking = False
        self.last_ai_move_time = time.time()

        if rebuild_ai:
            self._setup_ai_agents()

    def draw_board(self):
        """Draw the Go board grid."""
        try:
            self.screen.fill(self.bg_color)

            pygame.draw.rect(
                self.screen,
                self.line_color,
                (
                    self.board_left - 3,
                    self.board_top - 3,
                    self.board_right - self.board_left + 6,
                    self.board_bottom - self.board_top + 6,
                ),
                2,
            )

            # Vertical lines
            for i in range(self.BOARD_SIZE):
                x = self.board_left + i * self.CELL_SIZE
                pygame.draw.line(
                    self.screen,
                    self.line_color,
                    (x, self.board_top),
                    (x, self.board_bottom),
                    2,
                )

            # Horizontal lines
            for i in range(self.BOARD_SIZE):
                y = self.board_top + i * self.CELL_SIZE
                pygame.draw.line(
                    self.screen,
                    self.line_color,
                    (self.board_left, y),
                    (self.board_right, y),
                    2,
                )

            # Star points
            for row, col in self.STAR_POINTS:
                x = self.board_left + col * self.CELL_SIZE
                y = self.board_top + row * self.CELL_SIZE
                pygame.draw.circle(self.screen, STAR_POINT_COLOR, (x, y), 5)

            self.draw_coordinates()
            self.draw_stones()
            self.draw_ui()
            self.draw_settings_panel()

            if self.game_over:
                self.draw_game_over()

        except Exception as e:
            print(f"Error drawing board: {e}")

    def draw_coordinates(self):
        """Draw board coordinates."""
        try:
            letters = "ABCDEFGHJKLMNOPQRST"

            for i in range(self.BOARD_SIZE):
                x = self.board_left + i * self.CELL_SIZE
                letter = letters[i] if i < len(letters) else ""
                text = self.small_font.render(letter, True, self.line_color)
                text_rect = text.get_rect(center=(x, self.board_top - 20))
                self.screen.blit(text, text_rect)
                text_rect = text.get_rect(
                    center=(x, self.board_bottom + 20)
                )
                self.screen.blit(text, text_rect)

            for i in range(self.BOARD_SIZE):
                y = self.board_top + i * self.CELL_SIZE
                number = str(i + 1)
                text = self.small_font.render(number, True, self.line_color)
                text_rect = text.get_rect(center=(self.board_left - 20, y))
                self.screen.blit(text, text_rect)
                text_rect = text.get_rect(
                    center=(self.board_right + 20, y)
                )
                self.screen.blit(text, text_rect)
        except Exception as e:
            print(f"Error drawing coordinates: {e}")

    def draw_stones(self):
        """Draw all stones on the board."""
        try:
            for row in range(self.BOARD_SIZE):
                for col in range(self.BOARD_SIZE):
                    if self.game.board[row][col] != Stone.EMPTY:
                        x = self.board_left + col * self.CELL_SIZE
                        y = self.board_top + row * self.CELL_SIZE

                        stone_color = (
                            BLACK_STONE_COLOR
                            if self.game.board[row][col] == Stone.BLACK
                            else WHITE_STONE_COLOR
                        )

                        shadow_offset = 2
                        pygame.draw.circle(
                            self.screen,
                            (100, 100, 100),
                            (x + shadow_offset, y + shadow_offset),
                            self.STONE_RADIUS,
                        )

                        pygame.draw.circle(
                            self.screen,
                            stone_color,
                            (x, y),
                            self.STONE_RADIUS,
                        )

                        if self.game.board[row][col] == Stone.WHITE:
                            pygame.draw.circle(
                                self.screen,
                                (255, 255, 255),
                                (x - 4, y - 4),
                                4,
                            )

                        if self.game.last_move == (row, col):
                            pygame.draw.circle(
                                self.screen,
                                HIGHLIGHT_COLOR,
                                (x, y),
                                self.STONE_RADIUS + 4,
                                2,
                            )
        except Exception as e:
            print(f"Error drawing stones: {e}")

    def draw_settings_panel(self):
        """Draw the settings panel with buttons."""
        try:
            panel_x = self.panel_x
            black_has_variants = self._uses_personality(self.ai_algorithm_black)
            white_has_variants = self._uses_personality(self.ai_algorithm_white)

            pygame.draw.rect(
                self.screen,
                (220, 220, 220),
                (panel_x - 10, 205, self.SIDE_PANEL_WIDTH, 490),
            )
            pygame.draw.rect(
                self.screen,
                self.line_color,
                (panel_x - 10, 205, self.SIDE_PANEL_WIDTH, 490),
                2,
            )

            titles = [
                ("Game Mode", 210),
                ("Black AI", 345),
                ("White AI", 500),
            ]

            for title, y_pos in titles:
                text = self.small_font.render(title, True, self.line_color)
                self.screen.blit(text, (panel_x, y_pos))

            labels = [
                ("Algorithm", 370),
                ("Agent", 430),
                ("Algorithm", 525),
                ("Agent", 585),
            ]

            for label, y_pos in labels:
                text = self.small_font.render(label, True, self.line_color)
                self.screen.blit(text, (panel_x, y_pos))

            for btn in self.mode_buttons:
                if btn.mode == self.mode:
                    btn.color = (50, 150, 50)
                    btn.hover_color = (70, 170, 70)
                else:
                    btn.color = (80, 80, 80)
                    btn.hover_color = (100, 100, 100)
                btn.draw(self.screen, self.small_font)

            for btn in self.black_algorithm_buttons:
                if btn.algorithm == self.ai_algorithm_black:
                    btn.color = (50, 150, 50)
                    btn.hover_color = (70, 170, 70)
                else:
                    btn.color = (80, 80, 80)
                    btn.hover_color = (100, 100, 100)
                btn.draw(self.screen, self.small_font)

            for btn in self.black_personality_buttons:
                if black_has_variants:
                    if btn.personality == self.ai_personality_black:
                        btn.color = (50, 150, 50)
                        btn.hover_color = (70, 170, 70)
                    else:
                        btn.color = (80, 80, 80)
                        btn.hover_color = (100, 100, 100)
                    btn.draw(self.screen, self.small_font)

            if not black_has_variants:
                self._draw_single_neural_agent(self.black_personality_buttons)

            for btn in self.white_algorithm_buttons:
                if btn.algorithm == self.ai_algorithm_white:
                    btn.color = (50, 150, 50)
                    btn.hover_color = (70, 170, 70)
                else:
                    btn.color = (80, 80, 80)
                    btn.hover_color = (100, 100, 100)
                btn.draw(self.screen, self.small_font)

            for btn in self.white_personality_buttons:
                if white_has_variants:
                    if btn.personality == self.ai_personality_white:
                        btn.color = (50, 150, 50)
                        btn.hover_color = (70, 170, 70)
                    else:
                        btn.color = (80, 80, 80)
                        btn.hover_color = (100, 100, 100)
                    btn.draw(self.screen, self.small_font)

            if not white_has_variants:
                self._draw_single_neural_agent(self.white_personality_buttons)

            self.apply_button.draw(self.screen, self.small_font)

        except Exception as e:
            print(f"Error drawing settings panel: {e}")

    def draw_ui(self):
        """Draw the user interface panel."""
        try:
            panel_x = self.panel_x

            player_text = "Current: BLACK" if self.game.current_player == Stone.BLACK else "Current: WHITE"
            player_color = BLACK_STONE_COLOR if self.game.current_player == Stone.BLACK else (100, 100, 100)

            text = self.font.render(player_text, True, player_color)
            self.screen.blit(text, (panel_x, 10))

            move_text = f"Moves: {self.game.move_count}"
            text = self.small_font.render(move_text, True, self.line_color)
            self.screen.blit(text, (panel_x, 50))

            cap_text = "Captured:"
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

            if self.ai_thinking:
                think_text = "AI Thinking..."
                text = self.small_font.render(think_text, True, (200, 100, 0))
                self.screen.blit(text, (panel_x, 190))

            instr_y = 685
            instr_lines = [
                "Controls: Click - Place stone",
                "P - Pass | N - New | R - Reset | Q - Quit",
            ]

            for line in instr_lines:
                text = self.small_font.render(line, True, self.line_color)
                self.screen.blit(text, (panel_x, instr_y))
                instr_y += 25

        except Exception as e:
            print(f"Error drawing UI: {e}")

    def draw_game_over(self):
        """Draw game over overlay."""
        try:
            overlay = pygame.Surface((self.window_width, self.window_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))

            if self.winner == Stone.BLACK:
                winner_text = "BLACK WINS!"
                color = BLACK_STONE_COLOR
            else:
                winner_text = "WHITE WINS!"
                color = WHITE_STONE_COLOR

            board_center_x = (self.board_left + self.board_right) // 2
            board_center_y = (self.board_top + self.board_bottom) // 2

            text = self.large_font.render(winner_text, True, color)
            text_rect = text.get_rect(center=(board_center_x, board_center_y - 30))
            self.screen.blit(text, text_rect)

            restart_text = self.font.render("Press N for new game", True, (255, 255, 255))
            restart_rect = restart_text.get_rect(center=(board_center_x, board_center_y + 30))
            self.screen.blit(restart_text, restart_rect)

        except Exception as e:
            print(f"Error drawing game over: {e}")

    def get_board_position(self, screen_pos: Tuple[int, int]) -> Tuple[int, int]:
        """Convert screen position to board coordinates."""
        try:
            x, y = screen_pos

            if (
                x < self.board_left
                or x > self.board_right
                or y < self.board_top
                or y > self.board_bottom
            ):
                return (-1, -1)

            col = round((x - self.board_left) / self.CELL_SIZE)
            row = round((y - self.board_top) / self.CELL_SIZE)

            if not self.game.is_valid_position(row, col):
                return (-1, -1)

            actual_x = self.board_left + col * self.CELL_SIZE
            actual_y = self.board_top + row * self.CELL_SIZE
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

            if self._uses_personality(self.ai_algorithm_black):
                for btn in self.black_personality_buttons:
                    if btn.rect.collidepoint(pos):
                        self.ai_personality_black = btn.personality
                        return True

            if self._uses_personality(self.ai_algorithm_white):
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
            self._reset_game(rebuild_ai=True)
        except Exception as e:
            print(f"Error applying settings: {e}")

    def run(self):
        """Main game loop."""
        try:
            while self.running:
                self.clock.tick(30)
                current_time = time.time()

                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False

                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        if event.button == 1:
                            if not self.handle_settings_click(event.pos):
                                if not self.game_over and self.is_human_turn():
                                    row, col = self.get_board_position(event.pos)
                                    if row >= 0 and col >= 0:
                                        if self.game.place_stone(row, col):
                                            self.check_game_over()
                                            self.last_ai_move_time = current_time

                    elif event.type == pygame.MOUSEMOTION:
                        hover_buttons = (
                            self.mode_buttons
                            + self.black_algorithm_buttons
                            + self.white_algorithm_buttons
                            + [self.apply_button]
                        )
                        if self._uses_personality(self.ai_algorithm_black):
                            hover_buttons += self.black_personality_buttons
                        if self._uses_personality(self.ai_algorithm_white):
                            hover_buttons += self.white_personality_buttons

                        for btn in hover_buttons:
                            btn.handle_event(event)

                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_p:
                            if not self.game_over and self.is_human_turn():
                                self.game.pass_turn()
                                self.check_game_over()
                                self.last_ai_move_time = current_time
                        elif event.key == pygame.K_n:
                            self._reset_game(rebuild_ai=True)
                        elif event.key == pygame.K_r:
                            self.mode = GameMode.HUMAN_VS_HUMAN
                            self.ai_algorithm_black = AIAlgorithm.MINIMAX
                            self.ai_algorithm_white = AIAlgorithm.MINIMAX
                            self.ai_personality_black = AIPersonality.ALBI
                            self.ai_personality_white = AIPersonality.ANKON
                            self.apply_settings()
                        elif event.key == pygame.K_q:
                            self.running = False

                # AI move handling
                if (
                    not self.game_over
                    and self.mode != GameMode.HUMAN_VS_HUMAN
                    and not self.is_human_turn()
                ):
                    if current_time - self.last_ai_move_time > self.ai_move_delay:
                        self.ai_thinking = True
                        ai_move = self.get_ai_move()
                        self.ai_thinking = False

                        if ai_move:
                            self.game.place_stone(ai_move[0], ai_move[1])
                            self.check_game_over()
                        else:
                            self.game.pass_turn()
                            self.check_game_over()

                        self.last_ai_move_time = current_time

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
