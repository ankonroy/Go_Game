
import os
import tensorflow as tf
from game_models import GoBoard, Stone
from AI_Agent_NN.neural_agent import NeuralNetAI

def test_neural_ai():
    print("Testing Neural AI...")
    model_path = "AI_Agent_NN/weights/best_model.keras"
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Skipping neural test.")
        return

    try:
        board = GoBoard(size=19)
        ai = NeuralNetAI(player=Stone.BLACK, model_path=model_path)
        move = ai.get_best_move(board)
        print(f"Neural AI suggested move: {move}")
        print("Neural AI test passed!")
    except Exception as e:
        print(f"Neural AI test failed: {e}")

if __name__ == "__main__":
    test_neural_ai()
