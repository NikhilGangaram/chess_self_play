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
│   └── minimax_agent.py    # Classical minimax AI with opening book
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

## 🎮 Features

### Interactive GUI
- **Agent Selection**: Currently features Minimax agent
- **Multiple Game Modes**: Human vs Human, Human vs AI, AI vs AI
- **Visual Feedback**: Piece highlighting, legal move visualization
- **Game Control**: Pause/resume functionality, game reset
- **Export Options**: Save board positions as SVG

### Research Tools
- **Agent vs Agent**: Framework ready for future agent comparisons
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
3. **Agent vs Agent**: Ready for future agent implementations

### Agent Configuration

The GUI currently features:
- **Minimax Agent**: Classical approach with depth-5 search

## 🔬 Research Applications

This platform is designed for studying:

### Self-Play Reinforcement Learning
- **AlphaZero-style Training**: Neural networks learning through self-play
- **Policy Gradient Methods**: Direct policy optimization through self-play

### Comparative Analysis
- **Search vs Learning**: How different approaches handle complex positions
- **Tactical vs Positional**: Comparing agent strengths in different game phases

### Algorithm Development
- **Evaluation Functions**: Developing better position assessment methods
- **Opening Theory**: Studying how fixed openings affect agent performance

## 🎯 Next Steps: Self-Play Training

### Planned Self-Play Infrastructure
- **Neural Network Integration**: Combine traditional search with deep learning
- **Training Pipeline**: Automated self-play game generation
- **Policy and Value Networks**: Learn from self-play data
- **Iterative Improvement**: Continuously update networks through self-play

### Training Architecture
- **Game Generation**: Parallel self-play using enhanced search algorithms
- **Data Collection**: Store positions, moves, and outcomes
- **Network Training**: Update neural networks on self-play data
- **Evaluation**: Test improved networks against previous versions

### Research Questions
- **Sample Efficiency**: How quickly can agents learn strong play?
- **Emergent Strategy**: What playing styles develop through self-play?
- **Generalization**: How well do trained agents transfer to new positions?

## 🤝 Contributing

This project welcomes contributions in several areas:

- **Algorithm Improvements**: Enhance existing agents
- **Neural Integration**: Implementing neural network agents
- **Training Methods**: Developing self-play algorithms
- **Analysis Tools**: Create evaluation and visualization tools
- **Documentation**: Improve explanations and tutorials

## 📖 Research Background

Self-play has proven highly effective in game AI, notably in:

- **AlphaGo/AlphaZero**: Achieved superhuman performance through self-play
- **MuZero**: General-purpose self-play learning algorithm
- **Leela Chess Zero**: Open-source AlphaZero implementation for chess

This project aims to build upon these concepts, starting with a solid foundation and progressing toward neural self-play training.

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **Chess Programming Community**: For decades of chess AI research
- **DeepMind**: For pioneering self-play reinforcement learning
- **Leela Chess Zero**: For demonstrating open-source AlphaZero implementation
- **Open Source Contributors**: For the excellent libraries this project depends on
- **Chess AI Research Community**: For advancing chess AI algorithms

---

*This project is actively developed for chess self-play research. The next major milestone is integrating neural networks for true self-play learning!* 