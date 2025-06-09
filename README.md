# Chess Self-Play Reinforcement Learning

A research project focused on studying self-play reinforcement learning models in chess. This project provides a foundation for experimenting with different AI agents and observing how they learn and improve through self-play.

## 🎯 Project Goals

The primary objective is to explore and study **self-play reinforcement learning** in the context of chess:

- **Agent Development**: Create and compare different AI architectures
- **Self-Play Training**: Implement algorithms where agents learn by playing against themselves
- **Performance Analysis**: Study how agents improve over time through self-play
- **Algorithm Comparison**: Compare traditional search methods vs. modern RL approaches
- **Research Platform**: Provide a flexible framework for chess AI experimentation

## 🏗️ Project Structure

```
chess_self_play/
├── agents/
│   ├── minimax_agent.py    # Classical minimax AI with opening book
│   └── mcts_agent.py       # Optimized Monte Carlo Tree Search agent
├── gui.py                  # Interactive chess GUI with agent selection
├── README.md              # This file
└── (future additions)
    ├── neural_agent.py     # Neural network-based agents
    └── training/           # Self-play training scripts
```

## 🤖 Current Agents

### MinimaxAgent
A classical chess AI featuring:
- **Opening Book**: 7-move fixed opening sequences for consistent play
- **Minimax Search**: Alpha-beta pruning with move ordering
- **Enhanced Evaluation**: Material, positional, and tactical considerations
- **Optimizations**: MVV-LVA capture ordering, check/promotion bonuses

### MCTSAgent (Enhanced)
A state-of-the-art Monte Carlo Tree Search implementation with modern optimizations:
- **RAVE Enhancement**: Rapid Action Value Estimation for faster learning
- **Progressive Bias**: Chess-specific move prioritization (captures, checks, promotions)
- **Sophisticated Evaluation**: Piece-square tables and positional analysis
- **Advanced Selection**: UCB1 + RAVE + progressive bias for optimal exploration
- **Virtual Loss Support**: Ready for parallel search implementations
- **Smart Simulation**: Weighted random policy favoring tactical moves

**Key MCTS Optimizations:**
- **RAVE (Rapid Action Value Estimation)**: Shares move statistics across tree nodes
- **Progressive Bias**: Prioritizes tactical moves early in search
- **Enhanced Position Evaluation**: Uses piece-square tables for better position assessment
- **Improved Move Ordering**: MVV-LVA for captures, priority for checks and promotions
- **Robust Move Selection**: Combines visit count and win rate for final decisions

## 🎮 Features

### Interactive GUI
- **Agent Selection**: Choose between Minimax and MCTS agents
- **Multiple Game Modes**: Human vs Human, Human vs AI, AI vs AI
- **Agent Configuration**: Select different agents for white and black in AI vs AI mode
- **Visual Feedback**: Piece highlighting, legal move visualization
- **Game Control**: Pause/resume functionality, game reset
- **Export Options**: Save board positions as SVG

### Research Tools
- **Agent vs Agent**: Watch different AI approaches compete
- **Performance Comparison**: Observe MCTS vs Minimax gameplay
- **Real-time Analysis**: See agent thinking process and move evaluation

## 📚 Dependencies & Credits

This project builds upon several excellent open-source libraries:

### Core Libraries
- **[python-chess](https://github.com/niklasf/python-chess)** by Niklas Fiekas
  - Chess game logic, move generation, and board representation
  - FEN/PGN parsing and game state management
  - Essential foundation for all chess-related operations

- **[PyQt6](https://www.riverbankcomputing.com/software/pyqt/)** by Riverbank Computing
  - Cross-platform GUI framework
  - Interactive chess board visualization
  - Real-time game interaction and control

### Future Dependencies (Self-Play RL)
- **PyTorch**: For neural network agents and self-play training
- **NumPy**: Numerical computations for RL algorithms
- **Matplotlib**: Training progress visualization and learning curves
- **Tensorboard**: Experiment tracking and monitoring

## 🚀 Getting Started

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd chess_self_play

# Install dependencies
pip install python-chess PyQt6
```

### Running the GUI

```bash
python gui.py
```

### Playing with Different Agents

1. **Human vs Human**: Traditional chess play
2. **Human vs Minimax**: Test against classical search AI
3. **Human vs MCTS**: Challenge the enhanced Monte Carlo agent
4. **MCTS vs Minimax**: Watch modern vs classical AI approaches compete
5. **Agent vs Agent**: Customize both sides with different agents

### Agent Configuration

The GUI allows you to select different agents:
- **Minimax Agent**: Classical approach with depth-5 search
- **MCTS Agent**: Modern approach with 5000 iterations and 3-second time limit

## 🔬 Research Applications

This platform is designed for studying:

### Self-Play Reinforcement Learning
- **AlphaZero-style Training**: Neural networks learning through self-play
- **MCTS Integration**: Using MCTS as the foundation for neural-guided search
- **Policy Gradient Methods**: Direct policy optimization through self-play

### Comparative Analysis
- **Traditional vs Modern**: Minimax vs MCTS performance analysis
- **Search vs Learning**: How different approaches handle complex positions
- **Tactical vs Positional**: Comparing agent strengths in different game phases

### Algorithm Development
- **MCTS Enhancements**: Testing and improving tree search algorithms
- **Evaluation Functions**: Developing better position assessment methods
- **Opening Theory**: Studying how fixed openings affect agent performance

## 🎯 Next Steps: Self-Play Training

### Planned Self-Play Infrastructure
- **Neural Network Integration**: Combine MCTS with deep learning
- **Training Pipeline**: Automated self-play game generation
- **Policy and Value Networks**: Learn from MCTS-guided play
- **Iterative Improvement**: Continuously update networks through self-play

### Training Architecture
- **Game Generation**: Parallel self-play using enhanced MCTS
- **Data Collection**: Store positions, moves, and outcomes
- **Network Training**: Update neural networks on self-play data
- **Evaluation**: Test improved networks against previous versions

### Research Questions
- **Sample Efficiency**: How quickly can agents learn strong play?
- **Emergent Strategy**: What playing styles develop through self-play?
- **Generalization**: How well do trained agents transfer to new positions?

## 🤝 Contributing

This project welcomes contributions in several areas:

- **MCTS Improvements**: Further algorithm enhancements
- **Neural Integration**: Implementing neural network agents
- **Training Methods**: Developing self-play algorithms
- **Analysis Tools**: Create evaluation and visualization tools
- **Documentation**: Improve explanations and tutorials

## 📖 Research Background

Self-play has proven highly effective in game AI, notably in:

- **AlphaGo/AlphaZero**: Achieved superhuman performance through self-play
- **MuZero**: General-purpose self-play learning algorithm
- **Leela Chess Zero**: Open-source AlphaZero implementation for chess

This project aims to build upon these concepts, starting with a strong MCTS foundation and progressing toward neural self-play training.

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **Chess Programming Community**: For decades of chess AI research
- **DeepMind**: For pioneering self-play reinforcement learning
- **Leela Chess Zero**: For demonstrating open-source AlphaZero implementation
- **Open Source Contributors**: For the excellent libraries this project depends on
- **MCTS Research Community**: For advancing Monte Carlo Tree Search algorithms

---

*This project is actively developed for chess self-play research. The next major milestone is integrating neural networks with the enhanced MCTS foundation for true self-play learning!* 