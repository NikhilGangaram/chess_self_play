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
├── gui.py                  # Interactive chess GUI for visualization
├── README.md              # This file
└── (future additions)
    ├── neural_agent.py     # Neural network-based agents
    ├── mcts_agent.py       # Monte Carlo Tree Search agents
    └── training/           # Self-play training scripts
```

## 🤖 Current Agents

### MinimaxAgent
A classical chess AI featuring:
- **Opening Book**: Classical openings (Italian Game, Ruy Lopez, Queen's Gambit)
- **Minimax Search**: Alpha-beta pruning with move ordering
- **Enhanced Evaluation**: Material, positional, and tactical considerations
- **Optimizations**: MVV-LVA capture ordering, check/promotion bonuses

*Future agents will include neural networks, MCTS, and hybrid approaches.*

## 🎮 Features

### Interactive GUI
- **Multiple Game Modes**: Human vs Human, Human vs AI, AI vs AI
- **Visual Feedback**: Piece highlighting, legal move visualization
- **Game Control**: Pause/resume functionality, game reset
- **Export Options**: Save board positions as SVG

### Research Tools
- **Agent vs Agent**: Watch different AI approaches compete
- **Performance Metrics**: (Planned) Win rates, move quality analysis
- **Training Visualization**: (Planned) Learning curve tracking

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

### Planned Dependencies
- **PyTorch** / **TensorFlow**: For neural network agents
- **NumPy**: Numerical computations for RL algorithms
- **Matplotlib**: Training progress visualization
- **Ray/RLlib**: Distributed training frameworks

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

### Playing with Different Modes

1. **Human vs Human**: Traditional chess play
2. **Human vs AI**: Test your skills against the minimax agent
3. **AI vs AI**: Observe agent behavior and decision-making

## 🔬 Research Applications

This platform is designed for studying:

### Self-Play Reinforcement Learning
- **AlphaZero-style Training**: Neural networks learning through self-play
- **Policy Gradient Methods**: Direct policy optimization
- **Actor-Critic Approaches**: Combined value and policy learning

### Comparative Analysis
- **Traditional vs Modern**: Minimax vs Neural Networks
- **Sample Efficiency**: How quickly different agents learn
- **Generalization**: Performance across different positions

### Emergent Behavior
- **Strategy Evolution**: How playing styles develop over time
- **Opening Preferences**: What openings emerge from self-play
- **Tactical Discovery**: Novel tactical patterns learned by agents

## 🎯 Future Development

### Planned Agents
- **Neural Network Agent**: Deep learning-based chess AI
- **MCTS Agent**: Monte Carlo Tree Search implementation
- **Hybrid Agents**: Combining search and neural evaluation

### Training Infrastructure
- **Self-Play Pipeline**: Automated training loops
- **Distributed Training**: Multi-process game generation
- **Experiment Tracking**: Performance monitoring and comparison

### Analysis Tools
- **Game Database**: Store and analyze self-play games
- **Position Analysis**: Evaluate specific chess positions
- **Learning Curves**: Visualize agent improvement over time

## 🤝 Contributing

This project welcomes contributions in several areas:

- **New Agents**: Implement different AI approaches
- **Training Methods**: Develop self-play algorithms
- **Analysis Tools**: Create evaluation and visualization tools
- **Documentation**: Improve explanations and tutorials

## 📖 Research Background

Self-play has proven highly effective in game AI, notably in:

- **AlphaGo/AlphaZero**: Achieved superhuman performance through self-play
- **OpenAI Five/Dota**: Demonstrated complex strategy learning
- **MuZero**: General-purpose self-play learning algorithm

This project aims to make these concepts accessible for chess research and provide a platform for experimenting with novel approaches.

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **Chess Programming Community**: For decades of chess AI research
- **DeepMind**: For pioneering self-play reinforcement learning
- **Open Source Contributors**: For the excellent libraries this project depends on

---

*This project is actively developed for research purposes. Star the repository to follow development progress!* 