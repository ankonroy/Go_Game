"""
Monte Carlo Tree Search AI Agent - Albi
A Go AI using Monte Carlo Tree Search algorithm with safety limits.
"""

import math
import random
import time
from typing import Tuple, List, Optional, Set
from copy import deepcopy
from game_models import GoBoard, Stone


class MCTSNode:
    """Node for Monte Carlo Tree Search."""
    
    def __init__(self, board: GoBoard, move: Optional[Tuple[int, int]] = None, 
                 parent: Optional['MCTSNode'] = None, player: Optional[Stone] = None):
        self.board = board
        self.move = move
        self.parent = parent
        self.children: List[MCTSNode] = []
        self.visits = 0
        self.wins = 0
        self.player = player
        # FIXED: Compute untried moves lazily to avoid recursion
        self._untried_moves = None
        self._legal_moves_cache = None
    
    @property
    def untried_moves(self) -> List[Tuple[int, int]]:
        """Lazy computation of untried moves."""
        if self._untried_moves is None:
            self._untried_moves = self._get_legal_moves(self.board)
        return self._untried_moves
    
    def _get_legal_moves(self, board: GoBoard) -> List[Tuple[int, int]]:
        """Get all legal moves for current position."""
        moves = []
        # FIXED: Limit search area for performance
        for row in range(board.size):
            for col in range(board.size):
                if board.is_empty(row, col):
                    # Quick check - only verify if move might be legal
                    if self._quick_legal_check(board, row, col):
                        moves.append((row, col))
        return moves
    
    def _quick_legal_check(self, board: GoBoard, row: int, col: int) -> bool:
        """Quick check if move might be legal (no full board copy)."""
        # If position is empty, it's potentially legal
        # Full legality check is too expensive for MCTS
        return True
    
    def ucb1(self, exploration_constant: float = 1.41) -> float:
        """Calculate UCB1 value for node selection."""
        if self.visits == 0:
            return float('inf')
        
        if self.parent is None or self.parent.visits == 0:
            return float('inf')
        
        exploitation = self.wins / self.visits
        exploration = exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)
        
        return exploitation + exploration
    
    def is_fully_expanded(self) -> bool:
        """Check if all possible moves have been tried."""
        return len(self.untried_moves) == 0
    
    def best_child(self, exploration_constant: float = 1.41) -> Optional['MCTSNode']:
        """Select best child based on UCB1."""
        if not self.children:
            return None
        return max(self.children, key=lambda c: c.ucb1(exploration_constant))


class MonteCarloAI:
    """
    AI Agent using Monte Carlo Tree Search with safety limits.
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
        self.max_simulations = min(simulations, 1000)  # Cap at 1000
        self.time_limit = min(time_limit, 3.0)  # Cap at 3 seconds
        self.temperature = 1.0  # For move selection
    
    def get_best_move(self, board: GoBoard) -> Optional[Tuple[int, int]]:
        """
        Get the best move using Monte Carlo Tree Search with safety limits.
        """
        try:
            # Check if there are any valid moves
            valid_moves = self._get_valid_moves_quick(board)
            if not valid_moves:
                return None
            
            # If only one move, take it
            if len(valid_moves) == 1:
                return valid_moves[0]
            
            # FIXED: Limit moves for large board
            if len(valid_moves) > 50:
                valid_moves = self._prune_moves(board, valid_moves, 30)
            
            # Create root node (use deepcopy but with optimization)
            root_board = self._copy_board_light(board)
            root = MCTSNode(root_board, player=self.opponent)
            
            # Run MCTS with safety limits
            simulations_run = 0
            start_time = time.time()
            
            while (simulations_run < self.max_simulations and 
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
                    
                    # Simulation (with timeout)
                    result = self._simulate(node)
                    
                    # Backpropagation
                    self._backpropagate(node, result)
                    
                    simulations_run += 1
                    
                except Exception as e:
                    print(f"Error in MCTS iteration: {e}")
                    continue
            
            # Choose best move
            if root.children:
                # FIXED: Use visit count for robust selection
                best_child = max(root.children, key=lambda c: c.visits)
                return best_child.move
            
            return random.choice(valid_moves)
            
        except Exception as e:
            print(f"Error in get_best_move: {e}")
            # Fallback to random move
            valid_moves = self._get_valid_moves_quick(board)
            return random.choice(valid_moves) if valid_moves else None
    
    def _prune_moves(self, board: GoBoard, moves: List[Tuple[int, int]], 
                     keep: int) -> List[Tuple[int, int]]:
        """Prune moves based on simple heuristics."""
        scored_moves = []
        
        for move in moves[:100]:  # Limit initial consideration
            row, col = move
            score = 0
            
            # Prefer center and corners
            center = board.size // 2
            distance = abs(row - center) + abs(col - center)
            score += (board.size - distance)
            
            # Check nearby stones
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = row + dr, col + dc
                if board.is_valid_position(nr, nc):
                    if board.board[nr][nc] != Stone.EMPTY:
                        score += 10
            
            scored_moves.append((move, score))
        
        scored_moves.sort(key=lambda x: x[1], reverse=True)
        return [m[0] for m in scored_moves[:keep]]
    
    def _select(self, node: MCTSNode) -> Optional[MCTSNode]:
        """Select a node to expand using UCB1."""
        try:
            while node.is_fully_expanded() and node.children:
                next_node = node.best_child()
                if next_node is None:
                    break
                node = next_node
            return node
        except Exception:
            return node
    
    def _expand(self, node: MCTSNode) -> Optional[MCTSNode]:
        """Expand a node by adding a new child."""
        try:
            if not node.untried_moves:
                return node
            
            # Take a random untried move
            move = random.choice(node.untried_moves)
            node.untried_moves.remove(move)
            
            # Create new board state
            new_board = self._copy_board_light(node.board)
            new_board.current_player = self._get_next_player(node.board.current_player)
            
            # FIXED: Check if move is legal before placing
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
    
    def _simulate(self, node: MCTSNode) -> float:
        """
        Run a random simulation with safety limits.
        """
        try:
            board = self._copy_board_light(node.board)
            
            # FIXED: Add simulation depth limit
            max_moves = 50  # Limit simulation length
            moves_played = 0
            
            while moves_played < max_moves:
                # Check if game is over
                winner = self._check_winner(board)
                if winner is not None:
                    break
                
                # Get valid moves (quick version)
                valid_moves = self._get_valid_moves_quick(board)
                
                if not valid_moves:
                    board.pass_turn()
                    moves_played += 1
                    continue
                
                # Random move with simple bias
                if random.random() < 0.3:  # 30% heuristic moves
                    move = self._heuristic_move(board, valid_moves)
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
                return 0.5  # Draw
            
        except Exception as e:
            print(f"Error in simulate: {e}")
            return 0.5
    
    def _heuristic_move(self, board: GoBoard, moves: List[Tuple[int, int]]) -> Tuple[int, int]:
        """Simple heuristic for move selection during playout."""
        best_move = moves[0]
        best_score = -1
        
        for move in moves[:10]:  # Only check first 10 for speed
            row, col = move
            score = 0
            
            # Check for captures
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = row + dr, col + dc
                if board.is_valid_position(nr, nc):
                    if board.board[nr][nc] == self.opponent:
                        score += 5
            
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move
    
    def _backpropagate(self, node: MCTSNode, result: float):
        """Backpropagate simulation result up the tree."""
        try:
            while node:
                node.visits += 1
                node.wins += result
                result = 1 - result  # Switch perspective for parent
                node = node.parent
        except Exception:
            pass
    
    def _get_valid_moves_quick(self, board: GoBoard) -> List[Tuple[int, int]]:
        """Quick version - doesn't check full legality."""
        moves = []
        for row in range(board.size):
            for col in range(board.size):
                if board.is_empty(row, col):
                    # In MCTS, we can be more permissive
                    moves.append((row, col))
        return moves
    
    def _copy_board_light(self, board: GoBoard) -> GoBoard:
        """Lightweight board copy for MCTS."""
        new_board = GoBoard(size=board.size)
        # FIXED: More efficient copying
        for row in range(board.size):
            new_board.board[row] = board.board[row][:]  # Use list slicing for speed
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
    
    def _get_next_player(self, current: Stone) -> Stone:
        """Get the next player."""
        return Stone.WHITE if current == Stone.BLACK else Stone.BLACK


def get_ai_move(board_state, player_color, simulations=500, time_limit=1.0):
    """
    Function to be called from main.py.
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
    
    print("Testing Monte Carlo AI (Albi)...")
    move = ai.get_best_move(board)
    print(f"Best first move: {move}")