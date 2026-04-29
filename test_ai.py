
import time
from game_models import GoBoard, Stone
from AI_Agent_Albi.minmax import MinimaxAI as AlbiMinimax
from AI_Agent_Ankon.minmax import MinimaxAI as AnkonMinimax

def test_ai_vs_ai():
    print("Testing AI vs AI (Albi vs Ankon)...")
    board = GoBoard(size=9) # Small board for speed
    
    # Albi (Black) vs Ankon (White)
    ai_black = AlbiMinimax(player=Stone.BLACK, depth=2)
    ai_white = AnkonMinimax(player=Stone.WHITE, depth=2)
    
    max_moves = 20
    for i in range(max_moves):
        if board.is_game_over():
            break
            
        current_ai = ai_black if board.current_player == Stone.BLACK else ai_white
        start_time = time.time()
        move = current_ai.get_best_move(board)
        elapsed = time.time() - start_time
        
        if move:
            row, col = move
            board.place_stone(row, col)
            print(f"Move {i+1}: {'Black' if board.current_player == Stone.WHITE else 'White'} plays at {move} (took {elapsed:.2f}s)")
        else:
            print(f"Move {i+1}: {'Black' if board.current_player == Stone.BLACK else 'White'} passes.")
            board.pass_turn()
            
        if board.is_game_over():
            break
            
    print("-" * 20)
    print(f"Final board state:\n{board}")
    print(f"Captured stones: {board.captured_stones}")
    if board.is_game_over():
        print(f"Winner: {board.get_winner()}")
    else:
        print("Game ended after max moves.")
    print("AI vs AI test finished!")

if __name__ == "__main__":
    test_ai_vs_ai()
