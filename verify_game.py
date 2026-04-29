
from game_models import GoBoard, Stone

def test_capture_win():
    print("Testing 'First Capture Wins' condition...")
    board = GoBoard(size=5) # Smaller board for quick test
    
    # Setup a situation where Black can capture White
    # W B . . .
    # B . . . .
    # . . . . .
    
    board.place_stone(0, 1) # Black
    board.place_stone(0, 0) # White
    board.place_stone(1, 0) # Black
    
    print(f"Board state:\n{board}")
    print(f"Captured stones: {board.captured_stones}")
    print(f"Is game over? {board.is_game_over()}")
    
    # Black captures White at (0,0) by playing at (0,1) and (1,0)
    # Wait, White at (0,0) has liberties at (0,1) and (1,0).
    # If Black plays (0,1) and (1,0), White is captured.
    
    if board.is_game_over():
        print(f"Winner: {board.get_winner()}")
    else:
        print("Game still in progress.")

    print("-" * 20)
    
    # Try another scenario
    board = GoBoard(size=5)
    board.place_stone(1, 1) # Black (0,1), (2,1), (1,0), (1,2) are liberties
    board.place_stone(0, 1) # White
    board.place_stone(2, 1) # Black
    board.place_stone(1, 0) # White
    board.place_stone(1, 2) # Black
    
    # Now White stone at (0,1) has liberties at (0,0), (0,2), (1,1) is occupied by Black
    # This is getting complicated to manual setup. Let's just do a simple surround.
    
    board = GoBoard(size=3)
    # B W B
    # . B .
    # . . .
    board.place_stone(0, 0) # B
    board.pass_turn()       # Skip W
    board.place_stone(0, 2) # B
    board.pass_turn()       # Skip W
    board.place_stone(1, 1) # B
    
    # Now W tries to play at (0,1)
    # It will be surrounded by B at (0,0), (0,2), (1,1)
    # Wait, if W plays at (0,1), it has 0 liberties.
    board.place_stone(0, 1) # W plays at (0,1)
    
    # Now B plays at (1,0) to capture? No, W is already surrounded.
    # Actually, if W plays at (0,1) and it has 0 liberties and captures no one, it's suicide.
    
    # Let's do it properly:
    board = GoBoard(size=3)
    board.place_stone(0, 0) # B (0,0)
    board.place_stone(0, 1) # W (0,1)
    board.place_stone(0, 2) # B (0,2)
    board.place_stone(1, 2) # W (1,2) - random move
    board.place_stone(1, 1) # B (1,1)
    # Now W at (0,1) has liberties: (0,0)B, (0,2)B, (1,1)B. 
    # Wait, (0,1) neighbor is also (1,1).
    # So W at (0,1) has neighbors: (0,0), (0,2), (1,1).
    # If all these are B, then W is captured.
    
    print(f"Board state before final move:\n{board}")
    board.place_stone(1, 0) # W random
    board.place_stone(1, 1) # B already played (1,1) above. 
    
    # Let's restart and be precise.
    board = GoBoard(size=3)
    # B W .
    # . . .
    board.place_stone(0, 0) # B
    board.place_stone(0, 1) # W
    # B W B
    # . . .
    board.place_stone(0, 2) # B
    board.place_stone(2, 2) # W (away)
    # B W B
    # . B .
    board.place_stone(1, 1) # B -> This should capture W at (0,1)
    
    print(f"Board state:\n{board}")
    print(f"Captured stones: {board.captured_stones}")
    print(f"Is game over? {board.is_game_over()}")
    print(f"Winner: {board.get_winner()}")
    
    assert board.is_game_over() == True
    assert board.get_winner() == Stone.BLACK
    print("Test passed!")

if __name__ == "__main__":
    test_capture_win()
