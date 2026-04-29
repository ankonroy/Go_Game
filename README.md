# Go Game AI Playground

A comprehensive Python implementation of the classic board game Go (Weiqi/Baduk), featuring a custom graphical interface, multiple AI architectures (Minimax, MCTS, and Neural Networks), and a complete training pipeline for machine learning.

The project is specifically tuned for a **"First Capture Wins"** variant, making games fast-paced and tactically intense.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- `pip` (Python package installer)

### Installation & Execution
The easiest way to run the project is using the provided `run.sh` script, which handles environment creation and dependency installation automatically:

```bash
# Clone the repository
git clone https://github.com/ankonroy/go-game.git
cd go-game

# Make script executable and run
chmod +x run.sh
./run.sh
```

---

## 🎮 Game Features

### Graphical User Interface (GUI)
Built with **Pygame**, the interface provides:
- **Traditional 19x19 Board**: Featuring star points (hoshi) and coordinate labels.
- **Move Highlighting**: Visually tracks the last move played.
- **Real-time Capture Tracking**: Displays the number of stones captured by each player.
- **Dynamic Settings Panel**: Change game modes, AI algorithms, and personalities on the fly.
- **AI Thinking Indicators**: Visual feedback when the computer is calculating its next move.

### Game Modes
- **Human vs Human**: Local multiplayer for two people.
- **Human vs AI**: Challenge different AI personalities and algorithms.
- **AI vs AI**: Spectate matches between different AI agents (e.g., Neural Net vs MCTS).

### Game Rules
- **Board Size**: Standard 19x19.
- **Win Condition**: First player to capture at least one stone wins the game.
- **Suicide Rule**: Players cannot place a stone in a position where it has no liberties and captures no opponent stones.
- **Ko Rule**: Prevents board state repetition to avoid infinite loops.

---

## 🤖 AI Agents

The project features three distinct AI architectures:

### 1. Minimax with Alpha-Beta Pruning
A classical tree search algorithm enhanced with:
- **Alpha-Beta Pruning**: Drastically reduces the number of nodes evaluated.
- **Move Ordering**: Prioritizes promising moves (like captures or blocks) to speed up pruning.
- **Tactical Heuristics**: Immediate checks for winning moves or blocking opponent threats.
- **Memoization**: Uses a Transposition Table to cache board state evaluations.

### 2. Monte Carlo Tree Search (MCTS)
A simulation-based algorithm that:
- **UCB1 Selection**: Balances exploration and exploitation.
- **Random Rollouts**: Simulates games to the end to estimate move quality.
- **Performance Optimized**: Features safety caps on simulations and time limits to ensure a smooth UI experience.

### 3. Neural Network AI
A modern deep learning approach:
- **Residual CNN Architecture**: A policy-value network using Residual Blocks (Conv2D -> BatchNormalization -> ReLU).
- **6-Plane Input**: Encodes board state, stone positions, current player, and the last move.
- **Neural-Guided Search**: Uses a **Restricted MCTS** that leverages the policy head for move selection and the value head for leaf evaluation.

---

## 🧠 Training & Dataset Pipeline

The project includes a full suite of tools for training the Neural AI:

### Dataset Generation (`training/dataset_gen.py`)
Generates training data by having high-simulation MCTS "Teachers" play against each other.
- **Shard-based Storage**: Data is saved in `.npz` files (shards) for efficient loading.
- **Symmetry Augmentation**: Automatically applies rotations and flips to increase data variety.

### Training (`training/train.py`)
- **Custom Keras Model**: Implements the dual-head policy-value network.
- **Data Augmentation**: Real-time symmetry application during training.
- **Tournament Mode (`training/tournament.py`)**: Automatically evaluates model performance by playing matches against classical AI baselines.

### Inspection & Evaluation
- **Dataset Inspector (`training/dataset_inspect.py`)**: Visualizes saved board states and labels.
- **Evaluator (`training/evaluate.py`)**: Measures model accuracy on validation and test splits.

---

## 📂 Project Structure

- `AI_Agent_Albi/`: Aggressive, capture-focused Minimax and MCTS agents.
- `AI_Agent_Ankon/`: Defensive, territory-focused Minimax and MCTS agents.
- `AI_Agent_NN/`: Neural network model definition, weights, and restricted MCTS.
- `dataset/`: Training, validation, and test data shards.
- `training/`: Scripts for data generation, training, and tournaments.
- `game_models.py`: Core Go logic and board state management.
- `main.py`: Pygame-based GUI and game entry point.
- `requirements.txt`: Python dependencies.
- `run.sh`: Automated setup and launch script.

---

## 🛠 Technical Details

- **Language**: Python 3.x
- **UI Framework**: Pygame
- **Deep Learning**: TensorFlow / Keras
- **Mathematics**: NumPy, SciPy

---

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.
