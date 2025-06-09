import chess
import chess.polyglot
import random
import math
import time
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class MCTSNode:
    """A node in the Monte Carlo Tree Search tree with RAVE support."""
    board_hash: int
    parent: Optional['MCTSNode'] = None
    children: Dict[chess.Move, 'MCTSNode'] = None
    visits: int = 0
    wins: float = 0.0
    move_from_parent: Optional[chess.Move] = None
    untried_moves: List[chess.Move] = None
    is_terminal: bool = False
    terminal_value: Optional[float] = None
    # RAVE statistics
    rave_visits: Dict[chess.Move, int] = None
    rave_wins: Dict[chess.Move, float] = None
    # Virtual loss for parallel search
    virtual_losses: int = 0
    
    def __post_init__(self):
        if self.children is None:
            self.children = {}
        if self.untried_moves is None:
            self.untried_moves = []
        if self.rave_visits is None:
            self.rave_visits = {}
        if self.rave_wins is None:
            self.rave_wins = {}

    def is_fully_expanded(self) -> bool:
        """Check if all legal moves have been tried."""
        return len(self.untried_moves) == 0

    def is_leaf(self) -> bool:
        """Check if this is a leaf node (no children)."""
        return len(self.children) == 0

    def ucb1_value(self, exploration_constant: float = math.sqrt(2), use_virtual_loss: bool = True) -> float:
        """Calculate UCB1 value for this node with virtual loss."""
        if self.visits == 0:
            return float('inf')
        
        effective_visits = self.visits + self.virtual_losses if use_virtual_loss else self.visits
        effective_wins = self.wins - self.virtual_losses if use_virtual_loss else self.wins
        
        exploitation = effective_wins / effective_visits if effective_visits > 0 else 0
        exploration = exploration_constant * math.sqrt(math.log(self.parent.visits) / effective_visits)
        return exploitation + exploration

    def best_child(self, exploration_constant: float = math.sqrt(2)) -> 'MCTSNode':
        """Select the best child using UCB1."""
        return max(self.children.values(), key=lambda child: child.ucb1_value(exploration_constant))

    def select_child(self) -> 'MCTSNode':
        """Select a child for exploration (UCB1 + RAVE + progressive bias)."""
        if not self.children:
            return self
        
        best_child = None
        best_value = float('-inf')
        
        for move, child in self.children.items():
            if child.visits == 0:
                return child  # Prioritize unvisited children
            
            # Standard UCB1 value
            ucb_value = child.ucb1_value()
            
            # RAVE enhancement (if enabled and available)
            if hasattr(self, 'rave_visits') and move in self.rave_visits and self.rave_visits[move] > 0:
                rave_value = self.rave_wins[move] / self.rave_visits[move]
                # Progressive RAVE blending
                beta = self.rave_visits[move] / (self.rave_visits[move] + child.visits + 
                                               4 * 0.5 * child.visits * self.rave_visits[move])
                ucb_value = (1 - beta) * ucb_value + beta * rave_value
            
            # Progressive bias for chess-specific heuristics
            if child.move_from_parent:
                bias = self._get_move_bias(child.move_from_parent)
                progressive_bias = bias / (child.visits + 1)
                ucb_value += progressive_bias
            
            if ucb_value > best_value:
                best_value = ucb_value
                best_child = child
        
        return best_child

    def _get_move_bias(self, move: chess.Move) -> float:
        """Get a bias value for chess moves to guide exploration."""
        bias = 0.0
        
        # Promotion bias
        if move.promotion == chess.QUEEN:
            bias += 2.0
        elif move.promotion:
            bias += 1.0
            
        # Center square bias (e4, e5, d4, d5)
        center_squares = {chess.E4, chess.E5, chess.D4, chess.D5}
        if move.to_square in center_squares:
            bias += 0.5
            
        return bias

    def add_child(self, move: chess.Move, board_hash: int) -> 'MCTSNode':
        """Add a child node for the given move."""
        child = MCTSNode(
            board_hash=board_hash,
            parent=self,
            move_from_parent=move
        )
        self.children[move] = child
        if move in self.untried_moves:
            self.untried_moves.remove(move)
        return child

    def update(self, result: float):
        """Update this node with a simulation result."""
        self.visits += 1
        self.wins += result

    def __repr__(self):
        return f"MCTSNode(move={self.move_from_parent}, visits={self.visits}, wins={self.wins:.2f})"


class MCTSAgent:
    """
    Monte Carlo Tree Search agent for chess with various optimizations.
    
    This implementation includes:
    - UCB1 selection with progressive bias
    - Optimized simulation policy 
    - Transposition table for node reuse
    - Time-based and iteration-based search limits
    - Opening book integration
    """
    
    def __init__(self, 
                 time_limit: float = 5.0,
                 max_iterations: int = 10000,
                 exploration_constant: float = math.sqrt(2),
                 use_opening_book: bool = True,
                 use_rave: bool = True,
                 rave_constant: float = 0.5,
                 progressive_bias: bool = True,
                 virtual_loss: float = 1.0):
        """
        Initialize the MCTS agent with advanced optimizations.
        
        Args:
            time_limit: Maximum time in seconds for each move
            max_iterations: Maximum MCTS iterations per move
            exploration_constant: UCB1 exploration parameter
            use_opening_book: Whether to use opening book for early game
            use_rave: Whether to use Rapid Action Value Estimation
            rave_constant: RAVE weighting parameter
            progressive_bias: Whether to use progressive bias
            virtual_loss: Virtual loss for parallel search
        """
        self.time_limit = time_limit
        self.max_iterations = max_iterations
        self.exploration_constant = exploration_constant
        self.use_opening_book = use_opening_book
        self.use_rave = use_rave
        self.rave_constant = rave_constant
        self.progressive_bias = progressive_bias
        self.virtual_loss = virtual_loss
        
        # Tree storage and statistics
        self.nodes: Dict[int, MCTSNode] = {}
        self.root: Optional[MCTSNode] = None
        self.total_simulations = 0
        
        # Opening book (simple but effective)
        self.opening_moves_white = [
            "e2e4", "d2d4", "b1c3", "g1f3", "f1c4", "e1g1", "a2a3"
        ]
        self.opening_moves_black = [
            "e7e5", "d7d5", "b8c6", "g8f6", "f8c5", "e8g8", "a7a6"
        ]
        
        # Piece values for quick evaluation
        self.piece_values = {
            chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
            chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 20000
        }
        
        # Position-square tables for more sophisticated evaluation
        self.pst = self._init_piece_square_tables()

    def _init_piece_square_tables(self) -> Dict:
        """Initialize piece-square tables for position evaluation."""
        return {
            chess.PAWN: [
                0,  0,  0,  0,  0,  0,  0,  0,
                50, 50, 50, 50, 50, 50, 50, 50,
                10, 10, 20, 30, 30, 20, 10, 10,
                5,  5, 10, 25, 25, 10,  5,  5,
                0,  0,  0, 20, 20,  0,  0,  0,
                5, -5,-10,  0,  0,-10, -5,  5,
                5, 10, 10,-20,-20, 10, 10,  5,
                0,  0,  0,  0,  0,  0,  0,  0
            ],
            chess.KNIGHT: [
                -50,-40,-30,-30,-30,-30,-40,-50,
                -40,-20,  0,  0,  0,  0,-20,-40,
                -30,  0, 10, 15, 15, 10,  0,-30,
                -30,  5, 15, 20, 20, 15,  5,-30,
                -30,  0, 15, 20, 20, 15,  0,-30,
                -30,  5, 10, 15, 15, 10,  5,-30,
                -40,-20,  0,  5,  5,  0,-20,-40,
                -50,-40,-30,-30,-30,-30,-40,-50
            ],
            chess.BISHOP: [
                -20,-10,-10,-10,-10,-10,-10,-20,
                -10,  0,  0,  0,  0,  0,  0,-10,
                -10,  0,  5, 10, 10,  5,  0,-10,
                -10,  5,  5, 10, 10,  5,  5,-10,
                -10,  0, 10, 10, 10, 10,  0,-10,
                -10, 10, 10, 10, 10, 10, 10,-10,
                -10,  5,  0,  0,  0,  0,  5,-10,
                -20,-10,-10,-10,-10,-10,-10,-20
            ],
            chess.ROOK: [
                0,  0,  0,  0,  0,  0,  0,  0,
                5, 10, 10, 10, 10, 10, 10,  5,
                -5,  0,  0,  0,  0,  0,  0, -5,
                -5,  0,  0,  0,  0,  0,  0, -5,
                -5,  0,  0,  0,  0,  0,  0, -5,
                -5,  0,  0,  0,  0,  0,  0, -5,
                -5,  0,  0,  0,  0,  0,  0, -5,
                0,  0,  0,  5,  5,  0,  0,  0
            ],
            chess.QUEEN: [
                -20,-10,-10, -5, -5,-10,-10,-20,
                -10,  0,  0,  0,  0,  0,  0,-10,
                -10,  0,  5,  5,  5,  5,  0,-10,
                -5,  0,  5,  5,  5,  5,  0, -5,
                0,  0,  5,  5,  5,  5,  0, -5,
                -10,  5,  5,  5,  5,  5,  0,-10,
                -10,  0,  5,  0,  0,  0,  0,-10,
                -20,-10,-10, -5, -5,-10,-10,-20
            ],
            chess.KING: [
                -30,-40,-40,-50,-50,-40,-40,-30,
                -30,-40,-40,-50,-50,-40,-40,-30,
                -30,-40,-40,-50,-50,-40,-40,-30,
                -30,-40,-40,-50,-50,-40,-40,-30,
                -20,-30,-30,-40,-40,-30,-30,-20,
                -10,-20,-20,-20,-20,-20,-20,-10,
                20, 20,  0,  0,  0,  0, 20, 20,
                20, 30, 10,  0,  0, 10, 30, 20
            ]
        }

    def _evaluate_position(self, board: chess.Board) -> float:
        """Enhanced position evaluation using piece-square tables."""
        if board.is_checkmate():
            return -9999.0 if board.turn else 9999.0
        if board.is_stalemate() or board.is_insufficient_material():
            return 0.0
        
        score = 0.0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None:
                continue
            
            value = self.piece_values[piece.piece_type]
            
            # Add positional value
            if piece.piece_type in self.pst:
                if piece.color == chess.WHITE:
                    pos_value = self.pst[piece.piece_type][square]
                else:
                    pos_value = self.pst[piece.piece_type][chess.square_mirror(square)]
                value += pos_value
            
            if piece.color == chess.WHITE:
                score += value
            else:
                score -= value
        
        # Normalize to [-1, 1] range
        return max(-1.0, min(1.0, score / 2000.0))

    def get_opening_move(self, board: chess.Board) -> Optional[chess.Move]:
        """Get opening book move if available."""
        if not self.use_opening_book or board.fullmove_number > 7:
            return None
            
        try:
            if board.turn == chess.WHITE:
                move_uci = self.opening_moves_white[board.fullmove_number - 1]
            else:
                move_uci = self.opening_moves_black[board.fullmove_number - 1]
            
            move = chess.Move.from_uci(move_uci)
            if move in board.legal_moves:
                return move
        except (IndexError, ValueError):
            pass
        
        return None

    def get_or_create_node(self, board: chess.Board) -> MCTSNode:
        """Get existing node or create new one for the board position."""
        board_hash = chess.polyglot.zobrist_hash(board)
        
        if board_hash not in self.nodes:
            node = MCTSNode(board_hash=board_hash)
            
            # Check if position is terminal
            if board.is_game_over():
                node.is_terminal = True
                result = board.result()
                if result == "1-0":
                    node.terminal_value = 1.0 if board.turn == chess.WHITE else -1.0
                elif result == "0-1":
                    node.terminal_value = -1.0 if board.turn == chess.WHITE else 1.0
                else:
                    node.terminal_value = 0.0
            else:
                # Initialize untried moves
                node.untried_moves = list(board.legal_moves)
                # Prioritize interesting moves (captures, checks)
                node.untried_moves = self._order_moves(board, node.untried_moves)
            
            self.nodes[board_hash] = node
        
        return self.nodes[board_hash]

    def _order_moves(self, board: chess.Board, moves: List[chess.Move]) -> List[chess.Move]:
        """Order moves for better MCTS performance (captures first, etc.)."""
        def move_priority(move):
            priority = 0
            
            # Captures
            if board.is_capture(move):
                victim = board.piece_at(move.to_square)
                attacker = board.piece_at(move.from_square)
                if victim and attacker:
                    # MVV-LVA: Most Valuable Victim - Least Valuable Attacker
                    priority += 10000 + 10 * self.piece_values[victim.piece_type] - self.piece_values[attacker.piece_type]
            
            # Checks
            board.push(move)
            if board.is_check():
                priority += 1000
            board.pop()
            
            # Promotions
            if move.promotion == chess.QUEEN:
                priority += 5000
            elif move.promotion:
                priority += 2000
            
            # Castle
            if board.is_castling(move):
                priority += 500
                
            return priority
        
        return sorted(moves, key=move_priority, reverse=True)

    def selection(self, node: MCTSNode, board: chess.Board) -> MCTSNode:
        """Selection phase: traverse tree using UCB1."""
        current = node
        
        while not current.is_terminal and current.is_fully_expanded() and not current.is_leaf():
            current = current.select_child()
            # Apply the move to get to the child's position
            if current.move_from_parent:
                board.push(current.move_from_parent)
        
        return current

    def expansion(self, node: MCTSNode, board: chess.Board) -> MCTSNode:
        """Expansion phase: add a new child node."""
        if node.is_terminal:
            return node
            
        if node.untried_moves:
            # Select a random untried move (could be improved with heuristics)
            move = node.untried_moves[0]  # Already ordered by priority
            board.push(move)
            
            child = node.add_child(move, chess.polyglot.zobrist_hash(board))
            child = self.get_or_create_node(board)
            node.children[move] = child
            child.parent = node
            child.move_from_parent = move
            
            return child
        
        return node

    def simulation(self, board: chess.Board) -> Tuple[float, List[chess.Move]]:
        """
        Simulation phase: play out the game with a fast policy.
        Returns result and move sequence for RAVE updates.
        """
        original_turn = board.turn
        moves_played = 0
        max_simulation_moves = 60  # Shorter simulations for better performance
        simulation_moves = []
        
        # Use a smarter simulation policy
        while not board.is_game_over() and moves_played < max_simulation_moves:
            moves = list(board.legal_moves)
            
            # Weighted random selection favoring good moves
            move = self._select_simulation_move(board, moves)
            board.push(move)
            simulation_moves.append(move)
            moves_played += 1
        
        # Evaluate final position - use evaluation function for early termination
        if board.is_game_over():
            result = board.result()
            if result == "1-0":
                final_result = 1.0 if original_turn == chess.WHITE else -1.0
            elif result == "0-1":
                final_result = -1.0 if original_turn == chess.WHITE else 1.0
            else:
                final_result = 0.0
        else:
            # Use position evaluation for non-terminal positions
            eval_score = self._evaluate_position(board)
            final_result = eval_score if board.turn == original_turn else -eval_score
        
        return final_result, simulation_moves

    def _select_simulation_move(self, board: chess.Board, moves: List[chess.Move]) -> chess.Move:
        """Select a move during simulation using a simple heuristic policy."""
        # Weighted selection based on move type
        capture_moves = []
        check_moves = []
        normal_moves = []
        
        for move in moves:
            if board.is_capture(move):
                capture_moves.append(move)
            else:
                board.push(move)
                if board.is_check():
                    check_moves.append(move)
                else:
                    normal_moves.append(move)
                board.pop()
        
        # Prefer captures > checks > normal moves
        if capture_moves and random.random() < 0.6:
            return random.choice(capture_moves)
        elif check_moves and random.random() < 0.3:
            return random.choice(check_moves)
        else:
            return random.choice(moves)

    def backpropagation(self, node: MCTSNode, result: float, simulation_moves: List[chess.Move] = None):
        """Backpropagation phase: update statistics up the tree with RAVE support."""
        current = node
        current_result = result
        
        while current is not None:
            current.update(current_result)
            
            # Update RAVE statistics if enabled
            if self.use_rave and simulation_moves:
                for move in simulation_moves:
                    if move in current.rave_visits:
                        current.rave_visits[move] += 1
                        current.rave_wins[move] += current_result
                    else:
                        current.rave_visits[move] = 1
                        current.rave_wins[move] = current_result
            
            # Flip result for parent (opponent's perspective)
            current_result = -current_result
            current = current.parent

    def mcts_search(self, board: chess.Board) -> Optional[chess.Move]:
        """
        Perform MCTS search and return the best move.
        """
        # Check for opening book move first
        opening_move = self.get_opening_move(board)
        if opening_move:
            print(f"MCTS Agent plays '{opening_move.uci()}' from opening book.")
            return opening_move
        
        print(f"MCTS Agent thinking... (time limit: {self.time_limit}s)")
        
        # Initialize or update root
        self.root = self.get_or_create_node(board)
        
        start_time = time.time()
        iterations = 0
        
        while (time.time() - start_time < self.time_limit and 
               iterations < self.max_iterations):
            
            # Make a copy of the board for this iteration
            board_copy = board.copy()
            
            # 1. Selection
            selected_node = self.selection(self.root, board_copy)
            
            # 2. Expansion
            expanded_node = self.expansion(selected_node, board_copy)
            
            # 3. Simulation
            if not expanded_node.is_terminal:
                result, simulation_moves = self.simulation(board_copy)
            else:
                result = expanded_node.terminal_value
                simulation_moves = []
            
            # 4. Backpropagation
            self.backpropagation(expanded_node, result, simulation_moves)
            
            iterations += 1
        
        self.total_simulations += iterations
        
        # Select best move using a combination of visits and win rate
        if not self.root.children:
            return random.choice(list(board.legal_moves)) if board.legal_moves else None
        
        # For robust play, prefer moves with high visit count
        # But also consider win rate for close decisions
        best_move = None
        best_score = float('-inf')
        
        for move, child in self.root.children.items():
            if child.visits == 0:
                continue
            
            win_rate = child.wins / child.visits
            # Combine visits (robustness) with win rate (quality)
            # Higher weight on visits for more robust play
            score = 0.8 * child.visits + 0.2 * win_rate * child.visits
            
            if score > best_score:
                best_score = score
                best_move = move
        
        if best_move is None:
            return random.choice(list(board.legal_moves)) if board.legal_moves else None
        
        best_child = self.root.children[best_move]
        win_rate = best_child.wins / best_child.visits if best_child.visits > 0 else 0
        
        print(f"MCTS completed {iterations} iterations. "
              f"Best move: {best_move.uci()} "
              f"(visits: {best_child.visits}, win rate: {win_rate:.2%})")
        
        return best_move

    def get_best_move(self, board: chess.Board) -> Optional[chess.Move]:
        """Main interface method to get the best move."""
        if board.is_game_over():
            return None
        
        return self.mcts_search(board)

    def get_stats(self) -> Dict:
        """Get statistics about the MCTS search."""
        return {
            'total_simulations': self.total_simulations,
            'nodes_in_tree': len(self.nodes),
            'root_visits': self.root.visits if self.root else 0
        } 