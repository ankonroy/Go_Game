"""
Monte Carlo Tree Search AI Agent - Ankon
A Go AI using Monte Carlo Tree Search algorithm with safety limits.
Focuses on defensive play and territory control.
"""

import math
import random
import time
from typing import Tuple, List, Optional, Set, Dict
from copy import deepcopy
from game_models import GoBoard, Stone


class MCTSNode:
    """Node for Monte Carlo Tree Search with safety features."""
    
    def __init__(self, board: GoBoard, move: Optional[Tuple[int, int]] = None, 
                 parent: Optional['MCTSNode'] = None, player: Optional[Stone] = None):
        self.board = board
        self.move = move
        self.parent = parent
        self.children: List[MCTSNode] = []
        self.visits = 0
        self.wins = 0
        self.player = player
        # Lazy initialization to prevent recursion
        self._untried_moves = None
        self._move_scores = None
    
    @property
    def untried_moves(self) -> List[Tuple[int, int]]:
        """Lazy computation of untried moves."""
        if self._untried_moves is None:
            self._untried_moves = self._get_promising_moves(self.board)
        return self._untried_moves
    
    def _get_promising_moves(self, board: GoBoard) -> List[Tuple[int, int]]:
        """Get promising moves (not all moves for performance)."""
        moves = []
        
        # FIXED: Limit search area - focus on areas near existing stones
        # First, collect all positions adjacent to stones
        considered = set()
        
        for row in range(board.size):
            for col in range(board.size):
                if board.board[row][col] != Stone.EMPTY:
                    # Add neighbors
                    for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                        nr, nc = row + dr, col + dc
                        if board.is_valid_position(nr, nc) and board.is_empty(nr, nc):
                            considered.add((nr, nc))
        
        # If no nearby positions, consider center
        if not considered:
            center = board.size // 2
            for i in range(-3, 4):
                for j in range(-3, 4):
                    r, c = center + i, center + j
                    if board.is_valid_position(r, c) and board.is_empty(r, c):
                        considered.add((r, c))
        
        # Convert to list and limit
        moves = list(considered)
        
        # Always include some random moves for exploration
        if len(moves) < 10:
            # Add some random positions
            for _ in range(20):
                r = random.randint(0, board.size - 1)
                c = random.randint(0, board.size - 1)
                if board.is_empty(r, c) and (r, c) not in moves:
                    moves.append((r, c))
                    if len(moves) >= 30:
                        break
        
        return moves[:30]  # Limit to 30 moves max
    
    def ucb1(self, exploration_constant: float = 1.41) -> float:
        """Calculate UCB1 value with safety checks."""
        if self.visits == 0:
            return float('inf')
        
        if self.parent is None or self.parent.visits == 0:
            return float('inf')
        
        try:
            exploitation = self.wins / self.visits
            exploration = exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)
            return exploitation + exploration + random.uniform(0, 0.01)  # Add small noise
        except (ValueError, ZeroDivisionError):
            return float('inf')
    
    def is_fully_expanded(self) -> bool:
        """Check if all moves have been tried."""
        return len(self.untried_moves) == 0
    
    def best_child(self, exploration_constant: float = 1.41) -> Optional['MCTSNode']:
        """Select best child."""
        if not self.children:
            return None
        try:
            return max(self.children, key=lambda c: c.ucb1(exploration_constant))
        except Exception:
            return random.choice(self.children) if self.children else None


class MonteCarloAI:
    """
    AI Agent using Monte Carlo Tree Search with safety limits.
    Focuses on defensive play and territory control.
    """
    
    def __init__(self, player: Stone, simulations: int = 500, time_limit: float = 1.0):
        """
        Initialize the Monte Carlo AI agent.
        
        Args:
            player: The stone color this AI plays as
            simulations: Number of simulations per move (reduced for safety)
            time_limit: Time limit in seconds for move calculation
        """
        self.player = player
        self.opponent = Stone.WHITE if player == Stone.BLACK else Stone.BLACK
        
        # FIXED: Add safety caps
        self.max_simulations = min(simulations, 800)  # Cap at 800
        self.time_limit = min(time_limit, 2.0)  # Cap at 2 seconds
        
        # Territory weights
        self.corner_weight = 2.0
        self.edge_weight = 1.5
        self.center_weight = 1.0
        
        # Safety counters
        self.simulations_done = 0
        self.max_depth = 30  # Max simulation depth
    
    def get_best_move(self, board: GoBoard) -> Optional[Tuple[int, int]]:
        """
        Get the best move using Monte Carlo Tree Search with safety limits.
        """
        try:
            # Reset counter
            self.simulations_done = 0
            
            # Get valid moves (quick version)
            valid_moves = self._get_valid_moves_quick(board)
            if not valid_moves:
                return None
            
            # If only one move, take it
            if len(valid_moves) == 1:
                return valid_moves[0]
            
            # Create root node (light copy)
            root_board = self._copy_board_light(board)
            root = MCTSNode(root_board, player=self.opponent)
            
            # Run MCTS with safety limits
            start_time = time.time()
            
            while (self.simulations_done < self.max_simulations and 
                   time.time() - start_time < self.time_limit):
                
                try:
                    # Selection
                    node = self._select(root)
                    if node is None:
                        break
                    
                    # Expansion
                    if node.untried_moves:
                        node = self._expand(node)
                        if node is None:
                            continue
                    
                    # Simulation (with depth limit)
                    result = self._simulate(node)
                    
                    # Backpropagation
                    self._backpropagate(node, result)
                    
                    self.simulations_done += 1
                    
                except Exception as e:
                    print(f"Error in MCTS iteration: {e}")
                    continue
            
            # Choose best move (consider both wins and visits)
            if root.children:
                # FIXED: Use robust selection
                best_child = self._select_best_move(root)
                if best_child and best_child.move:
                    return best_child.move
            
            # Fallback to random move
            return random.choice(valid_moves) if valid_moves else None
            
        except Exception as e:
            print(f"Error in get_best_move: {e}")
            # Ultimate fallback
            return self._get_random_move(board)
    
    def _select_best_move(self, root: MCTSNode) -> Optional[MCTSNode]:
        """Select best move using robust criteria."""
        if not root.children:
            return None
        
        # Filter children with visits
        valid_children = [c for c in root.children if c.visits > 0]
        if not valid_children:
            return None
        
        # Try win rate first
        best_by_winrate = max(valid_children, key=lambda c: c.wins / c.visits)
        
        # Try visit count as backup
        best_by_visits = max(valid_children, key=lambda c: c.visits)
        
        # Combine both
        if best_by_winrate.visits > 10:  # If enough samples, trust win rate
            return best_by_winrate
        else:
            return best_by_visits
    
    def _select(self, node: MCTSNode) -> Optional[MCTSNode]:
        """Select node using UCB1 with safety."""
        try:
            depth = 0
            while node.is_fully_expanded() and node.children and depth < 20:
                next_node = node.best_child()
                if next_node is None:
                    break
                node = next_node
                depth += 1
            return node
        except Exception:
            return node
    
    def _expand(self, node: MCTSNode) -> Optional[MCTSNode]:
        """Expand node by adding child with safety."""
        try:
            if not node.untried_moves:
                return node
            
            # Take best heuristic move instead of random
            move = self._select_best_untried_move(node)
            if move in node.untried_moves:
                node.untried_moves.remove(move)
            
            # Create new board state
            new_board = self._copy_board_light(node.board)
            new_board.current_player = self._get_next_player(node.board.current_player)
            
            # Check if move is legal
            if not new_board.place_stone(move[0], move[1]):
                return node
            
            # Create child node
            child = MCTSNode(
                board=new_board,
                move=move,
                parent=node,
                player=node.board.current_player
            )
            
            node.children.append(child)
            return child
            
        except Exception as e:
            print(f"Error in expand: {e}")
            return node
    
    def _select_best_untried_move(self, node: MCTSNode) -> Tuple[int, int]:
        """Select best untried move based on heuristic."""
        if not node.untried_moves:
            return None
        
        # Score each untried move
        best_move = node.untried_moves[0]
        best_score = -float('inf')
        
        for move in node.untried_moves[:10]:  # Only check first 10 for speed
            score = self._heuristic_move_score(node.board, move)
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move
    
    def _heuristic_move_score(self, board: GoBoard, move: Tuple[int, int]) -> float:
        """Heuristic to score moves for expansion."""
        row, col = move
        score = 0.0
        
        # Corner preference
        if (row == 0 or row == board.size - 1) and (col == 0 or col == board.size - 1):
            score += self.corner_weight * 2
        # Edge preference
        elif row == 0 or row == board.size - 1 or col == 0 or col == board.size - 1:
            score += self.edge_weight
        
        # Check for nearby friendly stones (connection)
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = row + dr, col + dc
            if board.is_valid_position(nr, nc):
                if board.board[nr][nc] == self.player:
                    score += 1.5  # Strong preference for extension
                elif board.board[nr][nc] == self.opponent:
                    score += 0.8  # Slight preference for proximity to opponent
        
        # Avoid filling own eyes (simplified)
        if self._is_eye_shape(board, row, col, self.player):
            score -= 5.0
        
        return score
    
    def _is_eye_shape(self, board: GoBoard, row: int, col: int, player: Stone) -> bool:
        """Quick check if position might be an eye."""
        if not board.is_empty(row, col):
            return False
        
        # Check all four neighbors
        friendly_count = 0
        neighbor_count = 0
        
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = row + dr, col + dc
            if board.is_valid_position(nr, nc):
                neighbor_count += 1
                if board.board[nr][nc] == player:
                    friendly_count += 1
        
        # If surrounded by friendly stones, might be an eye
        return neighbor_count > 0 and friendly_count == neighbor_count
    
    def _simulate(self, node: MCTSNode) -> float:
        """
        Run simulation with depth limit and territory focus.
        """
        try:
            board = self._copy_board_light(node.board)
            
            # FIXED: Add simulation depth limit
            moves_played = 0
            max_sim_moves = 40  # Limit simulation length
            
            while moves_played < max_sim_moves:
                # Check if game is over
                winner = self._check_winner(board)
                if winner is not None:
                    break
                
                # Get valid moves
                valid_moves = self._get_valid_moves_quick(board)
                
                if not valid_moves:
                    board.pass_turn()
                    moves_played += 1
                    continue
                
                # Choose move with heuristic bias
                if random.random() < 0.4:  # 40% heuristic moves
                    move = self._simulation_heuristic_move(board, valid_moves)
                else:
                    move = random.choice(valid_moves)
                
                board.place_stone(move[0], move[1])
                moves_played += 1
            
            # Evaluate result
            winner = self._check_winner(board)
            if winner == self.player:
                return 1.0
            elif winner == self.opponent:
                return 0.0
            else:
                # Territory-based evaluation
                return self._quick_evaluate(board)
                
        except Exception as e:
            print(f"Error in simulate: {e}")
            return 0.5
    
    def _simulation_heuristic_move(self, board: GoBoard, 
                                   moves: List[Tuple[int, int]]) -> Tuple[int, int]:
        """Heuristic move selection for simulations."""
        if not moves:
            return None
        
        # Limit moves to check for speed
        check_moves = moves[:15]
        
        best_move = random.choice(check_moves)
        best_score = -float('inf')
        
        for move in check_moves:
            row, col = move
            score = 0
            
            # Prefer territory building
            center = board.size // 2
            distance = abs(row - center) + abs(col - center)
            score += (board.size - distance) * 0.5
            
            # Check for nearby stones
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = row + dr, col + dc
                if board.is_valid_position(nr, nc):
                    if board.board[nr][nc] == self.player:
                        score += 2  # Connect to own stones
                    elif board.board[nr][nc] == self.opponent:
                        score += 1  # Approach opponent
            
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move
    
    def _quick_evaluate(self, board: GoBoard) -> float:
        """Quick territory evaluation for simulations."""
        my_captures = board.captured_stones[self.player]
        opp_captures = board.captured_stones[self.opponent]
        
        # Count stones roughly
        my_stones = 0
        opp_stones = 0
        
        # Sample every other row/col for speed
        for row in range(0, board.size, 2):
            for col in range(0, board.size, 2):
                if board.board[row][col] == self.player:
                    my_stones += 4  # Approximate
                elif board.board[row][col] == self.opponent:
                    opp_stones += 4
        
        my_score = my_captures * 100 + my_stones * 2
        opp_score = opp_captures * 100 + opp_stones * 2
        
        total = my_score + opp_score
        if total > 0:
            return my_score / total
        return 0.5
    
    def _backpropagate(self, node: MCTSNode, result: float):
        """Backpropagate results with safety."""
        try:
            depth = 0
            while node and depth < 50:  # Limit backprop depth
                node.visits += 1
                node.wins += result
                result = 1 - result  # Switch perspective
                node = node.parent
                depth += 1
        except Exception:
            pass
    
    def _get_valid_moves_quick(self, board: GoBoard) -> List[Tuple[int, int]]:
        """Quick version of valid moves."""
        moves = []
        for row in range(board.size):
            for col in range(board.size):
                if board.is_empty(row, col):
                    moves.append((row, col))
        return moves
    
    def _get_random_move(self, board: GoBoard) -> Optional[Tuple[int, int]]:
        """Get a completely random move as fallback."""
        empty_positions = []
        for row in range(board.size):
            for col in range(board.size):
                if board.is_empty(row, col):
                    empty_positions.append((row, col))
        
        return random.choice(empty_positions) if empty_positions else None
    
    def _copy_board_light(self, board: GoBoard) -> GoBoard:
        """Lightweight board copy for MCTS."""
        new_board = GoBoard(size=board.size)
        # FIXED: Efficient copying
        for row in range(board.size):
            new_board.board[row] = board.board[row][:]  # List slice for speed
        new_board.current_player = board.current_player
        new_board.move_count = board.move_count
        new_board.captured_stones = board.captured_stones.copy()
        return new_board
    
    def _check_winner(self, board: GoBoard) -> Optional[Stone]:
        """Check for winner."""
        if board.captured_stones[self.player] > 0:
            return self.player
        if board.captured_stones[self.opponent] > 0:
            return self.opponent
        return None
    
    def _get_next_player(self, current: Stone) -> Stone:
        """Get next player."""
        return Stone.WHITE if current == Stone.BLACK else Stone.BLACK


def get_ai_move(board_state, player_color, simulations=500, time_limit=1.0):
    """
    Function to be called from main.py with error handling.
    """
    try:
        player = Stone.BLACK if player_color == 1 else Stone.WHITE
        
        board = GoBoard(size=len(board_state))
        for row in range(len(board_state)):
            for col in range(len(board_state[row])):
                if board_state[row][col] == 1:
                    board.board[row][col] = Stone.BLACK
                elif board_state[row][col] == 2:
                    board.board[row][col] = Stone.WHITE
        
        ai = MonteCarloAI(player=player, simulations=simulations, time_limit=time_limit)
        move = ai.get_best_move(board)
        
        return move if move else (-1, -1)
        
    except Exception as e:
        print(f"Error in get_ai_move: {e}")
        return (-1, -1)


if __name__ == "__main__":
    # Test the AI
    board = GoBoard(size=19)
    ai = MonteCarloAI(player=Stone.BLACK, simulations=100)
    
    print("Testing Monte Carlo AI (Ankon)...")
    move = ai.get_best_move(board)
    print(f"Best first move: {move}")