import chess
import chess.polyglot
import random
import math
import time
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class MCTSNode:
    """
    A node in the Monte Carlo Tree Search tree with RAVE support.
    
    Each node represents a chess position and stores statistics about:
    - How many times we've visited this position (visits)
    - How many games we've won from positions reached through this node (wins)
    - What moves we haven't tried yet from this position (untried_moves)
    - Child nodes representing positions after each possible move (children)
    """
    board_hash: int
    parent: Optional['MCTSNode'] = None
    children: Dict[chess.Move, 'MCTSNode'] = None
    visits: int = 0  # How many times we've explored this node
    wins: float = 0.0  # How many wins we got from simulations through this node
    move_from_parent: Optional[chess.Move] = None  # The move that led to this position
    untried_moves: List[chess.Move] = None  # Moves we haven't explored yet
    is_terminal: bool = False  # True if this is a game-ending position
    terminal_value: Optional[float] = None  # The result if this is terminal (1, 0, -1)
    
    # RAVE (Rapid Action Value Estimation) statistics
    # RAVE Concept: If a move is good in one part of the tree, it might be good elsewhere too
    rave_visits: Dict[chess.Move, int] = None  # How often each move appeared in simulations
    rave_wins: Dict[chess.Move, float] = None  # How often each move led to wins
    
    # Virtual loss for parallel search (advanced feature)
    virtual_losses: int = 0
    
    def __post_init__(self):
        """Initialize empty collections if not provided."""
        if self.children is None:
            self.children = {}
        if self.untried_moves is None:
            self.untried_moves = []
        if self.rave_visits is None:
            self.rave_visits = {}
        if self.rave_wins is None:
            self.rave_wins = {}

    def is_fully_expanded(self) -> bool:
        """Check if all legal moves have been tried from this position."""
        return len(self.untried_moves) == 0

    def is_leaf(self) -> bool:
        """Check if this is a leaf node (no children explored yet)."""
        return len(self.children) == 0

    def ucb1_value(self, exploration_constant: float = math.sqrt(2), use_virtual_loss: bool = True) -> float:
        """
        Calculate UCB1 (Upper Confidence Bound) value for this node.
        
        UCB1 FORMULA: win_rate + exploration_constant * sqrt(ln(parent_visits) / node_visits)
        
        WHY UCB1 WORKS:
        - First term (win_rate): Exploit moves that have worked well (high win rate)
        - Second term (exploration): Explore moves we haven't tried much (uncertainty)
        - The balance between exploitation and exploration is crucial for good performance
        
        Higher UCB1 = more promising to explore next
        """
        if self.visits == 0:
            return float('inf')  # Prioritize completely unexplored nodes
        
        # Virtual loss helps with parallel search (can ignore for basic understanding)
        effective_visits = self.visits + self.virtual_losses if use_virtual_loss else self.visits
        effective_wins = self.wins - self.virtual_losses if use_virtual_loss else self.wins
        
        # EXPLOITATION: How well has this move performed so far?
        exploitation = effective_wins / effective_visits if effective_visits > 0 else 0
        
        # EXPLORATION: How uncertain are we about this move? (less visited = more uncertain = more exploration bonus)
        exploration = exploration_constant * math.sqrt(math.log(self.parent.visits) / effective_visits)
        
        return exploitation + exploration

    def best_child(self, exploration_constant: float = math.sqrt(2)) -> 'MCTSNode':
        """Select the best child using UCB1 (for final move selection)."""
        return max(self.children.values(), key=lambda child: child.ucb1_value(exploration_constant))

    def select_child(self) -> 'MCTSNode':
        """
        Select a child for exploration during MCTS search.
        
        This combines multiple heuristics:
        1. UCB1 (exploration vs exploitation balance)
        2. RAVE (rapid action value estimation)
        3. Progressive bias (chess-specific knowledge)
        """
        if not self.children:
            return self
        
        best_child = None
        best_value = float('-inf')
        
        for move, child in self.children.items():
            if child.visits == 0:
                return child  # Always try unvisited children first
            
            # Start with standard UCB1 value
            ucb_value = child.ucb1_value()
            
            # RAVE ENHANCEMENT: Use statistics from simulations to improve estimates
            if hasattr(self, 'rave_visits') and move in self.rave_visits and self.rave_visits[move] > 0:
                rave_value = self.rave_wins[move] / self.rave_visits[move]
                # Progressive RAVE blending: Use RAVE more when we have few direct visits
                beta = self.rave_visits[move] / (self.rave_visits[move] + child.visits + 
                                               4 * 0.5 * child.visits * self.rave_visits[move])
                ucb_value = (1 - beta) * ucb_value + beta * rave_value
            
            # PROGRESSIVE BIAS: Add chess knowledge to guide exploration
            if child.move_from_parent:
                bias = self._get_move_bias(child.move_from_parent)
                progressive_bias = bias / (child.visits + 1)  # Bias decreases as we learn more
                ucb_value += progressive_bias
            
            if ucb_value > best_value:
                best_value = ucb_value
                best_child = child
        
        return best_child

    def _get_move_bias(self, move: chess.Move) -> float:
        """
        Get a bias value for chess moves to guide exploration.
        
        PROGRESSIVE BIAS CONCEPT:
        Start with chess knowledge (captures are usually good, center control is good)
        but reduce this bias as we gather actual statistics from simulations.
        """
        bias = 0.0
        
        # PROMOTION BIAS: Promoting pawns is usually very good
        if move.promotion == chess.QUEEN:
            bias += 2.0
        elif move.promotion:
            bias += 1.0
            
        # CENTER CONTROL BIAS: Central squares are usually important
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
        """
        Update this node with a simulation result.
        
        Args:
            result: 1.0 = win, 0.0 = draw, -1.0 = loss (from this node's perspective)
        """
        self.visits += 1
        self.wins += result

    def __repr__(self):
        return f"MCTSNode(move={self.move_from_parent}, visits={self.visits}, wins={self.wins:.2f})"


class MCTSAgent:
    """
    Monte Carlo Tree Search agent for chess with various optimizations.
    
    CORE MCTS CONCEPT:
    Instead of exhaustively searching like minimax, MCTS uses random sampling to focus
    computational effort on the most promising parts of the game tree.
    
    THE FOUR PHASES OF MCTS:
    1. SELECTION: Walk down the tree using UCB1 to balance exploration vs exploitation
    2. EXPANSION: Add a new child node to expand our knowledge
    3. SIMULATION: Play out a random game from the new position  
    4. BACKPROPAGATION: Update statistics for all nodes in the path
    
    WHY MCTS WORKS:
    - Focuses search on promising moves (good moves get explored more)
    - Can handle large branching factors (doesn't need to search everything)
    - Improves with more time (anytime algorithm)
    - Uses random sampling to discover unexpected good moves
    
    KEY OPTIMIZATIONS:
    - UCB1 selection with progressive bias
    - RAVE (Rapid Action Value Estimation) 
    - Optimized simulation policy 
    - Transposition table for node reuse
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
            exploration_constant: UCB1 exploration parameter (higher = more exploration)
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
        self.nodes: Dict[int, MCTSNode] = {}  # Transposition table: position hash -> node
        self.root: Optional[MCTSNode] = None  # Current root of our search tree
        self.total_simulations = 0  # Track how much work we've done
        
        # --- Fixed opening sequence (7 moves) ---
        # Opening books help avoid weak early moves and speed up the opening phase
        self.opening_moves_white = [
            "e2e4",  # Move 1: King's pawn - controls center, opens lines for bishop/queen
            "d2d4",  # Move 2: Queen's pawn - establishes strong pawn center
            "b1c3",  # Move 3: Knight development - develops piece toward center
            "g1f3",  # Move 4: Knight development - attacks center, prepares castle
            "f1c4",  # Move 5: Bishop development - aims at weak f7 square
            "e1g1",  # Move 6: Castle kingside - king safety is critical
            "a2a3",  # Move 7: Prepare b4 - prevents opponent pieces on b4
        ]
        self.opening_moves_black = [
            "e7e5",  # Move 1: King's pawn - mirrors white's central control
            "d7d5",  # Move 2: Queen's pawn - challenges white's center
            "b8c6",  # Move 3: Knight development - develops with tempo
            "g8f6",  # Move 4: Knight development - attacks white's center
            "f8c5",  # Move 5: Bishop development - active bishop placement
            "e8g8",  # Move 6: Castle kingside - ensures king safety
            "a7a6",  # Move 7: Prepare b5 - creates space for queenside expansion
        ]
        
        # Piece values for quick evaluation (used in simulations and move ordering)
        self.piece_values = {
            chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
            chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 20000
        }
        
        # Position-square tables for more sophisticated evaluation
        self.pst = self._init_piece_square_tables()

    def _init_piece_square_tables(self) -> Dict:
        """
        Initialize piece-square tables for position evaluation.
        
        These tables help the simulation policy and position evaluation
        understand that piece placement matters (knight on e4 > knight on a1).
        """
        return {
            chess.PAWN: [
                0,  0,  0,  0,  0,  0,  0,  0,   # 8th rank: pawns can't be here
                50, 50, 50, 50, 50, 50, 50, 50,  # 7th rank: about to promote!
                10, 10, 20, 30, 30, 20, 10, 10,  # 6th rank: advanced pawns are strong
                5,  5, 10, 25, 25, 10,  5,  5,   # 5th rank: good pawn advancement
                0,  0,  0, 20, 20,  0,  0,  0,   # 4th rank: center pawns get bonus
                5, -5,-10,  0,  0,-10, -5,  5,   # 3rd rank: slight penalty for early moves
                5, 10, 10,-20,-20, 10, 10,  5,   # 2nd rank: penalty for blocking development
                0,  0,  0,  0,  0,  0,  0,  0    # 1st rank: starting position
            ],
            chess.KNIGHT: [
                -50,-40,-30,-30,-30,-30,-40,-50,  # Knights hate the edges
                -40,-20,  0,  0,  0,  0,-20,-40,
                -30,  0, 10, 15, 15, 10,  0,-30,
                -30,  5, 15, 20, 20, 15,  5,-30,  # Knights love the center
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
                5, 10, 10, 10, 10, 10, 10,  5,   # 7th rank is great for rooks
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
                -30,-40,-40,-50,-50,-40,-40,-30,  # King should stay safe early
                -30,-40,-40,-50,-50,-40,-40,-30,
                -30,-40,-40,-50,-50,-40,-40,-30,
                -30,-40,-40,-50,-50,-40,-40,-30,
                -20,-30,-30,-40,-40,-30,-30,-20,
                -10,-20,-20,-20,-20,-20,-20,-10,
                20, 20,  0,  0,  0,  0, 20, 20,   # Castled position
                20, 30, 10,  0,  0, 10, 30, 20
            ]
        }

    def _evaluate_position(self, board: chess.Board) -> float:
        """
        Enhanced position evaluation using piece-square tables.
        
        This is used during simulations to evaluate non-terminal positions.
        Unlike minimax, MCTS doesn't rely heavily on evaluation (it uses random sampling),
        but having a reasonable evaluation helps guide simulations and early termination.
        """
        # Handle terminal positions first
        if board.is_checkmate():
            return -9999.0 if board.turn else 9999.0
        if board.is_stalemate() or board.is_insufficient_material():
            return 0.0
        
        score = 0.0
        
        # Evaluate material and position for each piece
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None:
                continue
            
            # Basic piece value
            value = self.piece_values[piece.piece_type]
            
            # Add positional value from piece-square tables
            if piece.piece_type in self.pst:
                if piece.color == chess.WHITE:
                    pos_value = self.pst[piece.piece_type][square]
                else:
                    # Flip the square for black's perspective
                    pos_value = self.pst[piece.piece_type][chess.square_mirror(square)]
                value += pos_value
            
            # Add to total score
            if piece.color == chess.WHITE:
                score += value
            else:
                score -= value
        
        # Normalize to [-1, 1] range for MCTS (wins/losses)
        return max(-1.0, min(1.0, score / 2000.0))

    def get_opening_move(self, board: chess.Board) -> Optional[chess.Move]:
        """
        Get the next move from our fixed opening sequence based on whose turn it is.
        
        WHY USE OPENING BOOKS: 
        - Avoids spending time computing well-known opening theory
        - Ensures we don't make obvious mistakes in the opening
        - Gets us to interesting middlegame positions faster
        """
        if not self.use_opening_book or board.fullmove_number > 7:
            return None
            
        try:
            # Choose the appropriate move list based on which color is to move
            if board.turn == chess.WHITE:
                move_uci = self.opening_moves_white[board.fullmove_number - 1]
            else:
                move_uci = self.opening_moves_black[board.fullmove_number - 1]
            
            move = chess.Move.from_uci(move_uci)
            
            # Safety check: make sure the opening book move is actually legal
            if move in board.legal_moves:
                return move
        except (IndexError, ValueError):
            pass  # Fall back to MCTS if opening book fails
        
        return None

    def get_or_create_node(self, board: chess.Board) -> MCTSNode:
        """
        Get existing node or create new one for the board position.
        
        TRANSPOSITION TABLE CONCEPT:
        Different move sequences can lead to the same position (transpositions).
        We store each unique position only once and reuse the statistics,
        making our search more efficient.
        """
        board_hash = chess.polyglot.zobrist_hash(board)
        
        if board_hash not in self.nodes:
            node = MCTSNode(board_hash=board_hash)
            
            # Check if position is terminal (game over)
            if board.is_game_over():
                node.is_terminal = True
                result = board.result()
                if result == "1-0":  # White wins
                    node.terminal_value = 1.0 if board.turn == chess.WHITE else -1.0
                elif result == "0-1":  # Black wins
                    node.terminal_value = -1.0 if board.turn == chess.WHITE else 1.0
                else:  # Draw
                    node.terminal_value = 0.0
            else:
                # Initialize untried moves for non-terminal positions
                node.untried_moves = list(board.legal_moves)
                # Order moves to try promising ones first
                node.untried_moves = self._order_moves(board, node.untried_moves)
            
            self.nodes[board_hash] = node
        
        return self.nodes[board_hash]

    def _order_moves(self, board: chess.Board, moves: List[chess.Move]) -> List[chess.Move]:
        """
        Order moves for better MCTS performance.
        
        MOVE ORDERING FOR MCTS:
        Unlike minimax, MCTS doesn't depend critically on move ordering for correctness,
        but trying good moves first helps build a better tree faster.
        """
        def move_priority(move):
            priority = 0
            
            # CAPTURES: Usually tactical and important
            if board.is_capture(move):
                victim = board.piece_at(move.to_square)
                attacker = board.piece_at(move.from_square)
                if victim and attacker:
                    # MVV-LVA: Most Valuable Victim - Least Valuable Attacker
                    priority += 10000 + 10 * self.piece_values[victim.piece_type] - self.piece_values[attacker.piece_type]
            
            # CHECKS: Often strong tactical moves
            board.push(move)
            if board.is_check():
                priority += 1000
            board.pop()
            
            # PROMOTIONS: Almost always good
            if move.promotion == chess.QUEEN:
                priority += 5000
            elif move.promotion:
                priority += 2000
            
            # CASTLING: Usually good for king safety
            if board.is_castling(move):
                priority += 500
                
            return priority
        
        return sorted(moves, key=move_priority, reverse=True)

    def selection(self, node: MCTSNode, board: chess.Board) -> MCTSNode:
        """
        PHASE 1: SELECTION
        
        Walk down the tree from root using UCB1 to select the most promising path.
        
        SELECTION STRATEGY:
        - Use UCB1 to balance exploration (trying new things) vs exploitation (doing what works)
        - Continue until we reach a node that isn't fully expanded or is terminal
        - Apply moves to the board as we go down the tree
        """
        current = node
        
        # Keep going down the tree while we can
        while not current.is_terminal and current.is_fully_expanded() and not current.is_leaf():
            current = current.select_child()  # Use UCB1 + enhancements to pick child
            # Apply the move to get to the child's position
            if current.move_from_parent:
                board.push(current.move_from_parent)
        
        return current

    def expansion(self, node: MCTSNode, board: chess.Board) -> MCTSNode:
        """
        PHASE 2: EXPANSION
        
        Add a new child node to expand our search tree.
        
        EXPANSION STRATEGY:
        - If we have untried moves, pick one and create a new child node
        - This grows our tree one node at a time
        - We'll simulate from this new node to get information about it
        """
        if node.is_terminal:
            return node  # Can't expand terminal positions
            
        if node.untried_moves:
            # Pick the first untried move (they're already ordered by priority)
            move = node.untried_moves[0]
            board.push(move)
            
            # Create and add the new child node
            child = node.add_child(move, chess.polyglot.zobrist_hash(board))
            child = self.get_or_create_node(board)  # Use transposition table
            node.children[move] = child
            child.parent = node
            child.move_from_parent = move
            
            return child
        
        return node

    def simulation(self, board: chess.Board) -> Tuple[float, List[chess.Move]]:
        """
        PHASE 3: SIMULATION (also called "playout" or "rollout")
        
        Play out a random game from the current position to estimate its value.
        
        SIMULATION STRATEGY:
        - Play moves randomly (with some chess knowledge) until game ends
        - Don't play too long (can terminate early and use evaluation)
        - Return the result: 1.0 = win, 0.0 = draw, -1.0 = loss
        
        WHY SIMULATION WORKS:
        - Random games give us an unbiased sample of what might happen
        - Many random games will converge to the true value of the position
        - It's much faster than deep search
        """
        original_turn = board.turn
        moves_played = 0
        max_simulation_moves = 60  # Don't simulate forever
        simulation_moves = []  # Track moves for RAVE updates
        
        # Play random moves until game ends or we hit move limit
        while not board.is_game_over() and moves_played < max_simulation_moves:
            moves = list(board.legal_moves)
            
            # Use a smarter policy than pure random
            move = self._select_simulation_move(board, moves)
            board.push(move)
            simulation_moves.append(move)
            moves_played += 1
        
        # Determine the result
        if board.is_game_over():
            # Game actually ended - use the real result
            result = board.result()
            if result == "1-0":  # White wins
                final_result = 1.0 if original_turn == chess.WHITE else -1.0
            elif result == "0-1":  # Black wins
                final_result = -1.0 if original_turn == chess.WHITE else 1.0
            else:  # Draw
                final_result = 0.0
        else:
            # Game didn't end - use position evaluation
            eval_score = self._evaluate_position(board)
            final_result = eval_score if board.turn == original_turn else -eval_score
        
        return final_result, simulation_moves

    def _select_simulation_move(self, board: chess.Board, moves: List[chess.Move]) -> chess.Move:
        """
        Select a move during simulation using a simple heuristic policy.
        
        SIMULATION POLICY:
        Pure random is too weak - we bias toward reasonable moves:
        - Captures (especially good trades)
        - Checks (often strong)
        - Random otherwise
        
        This isn't as sophisticated as minimax evaluation, but it's much faster
        and gives us reasonable random games.
        """
        # Categorize moves by type
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
        
        # Weighted selection: prefer captures > checks > normal moves
        if capture_moves and random.random() < 0.6:  # 60% chance to play capture
            return random.choice(capture_moves)
        elif check_moves and random.random() < 0.3:  # 30% chance to play check
            return random.choice(check_moves)
        else:
            return random.choice(moves)  # Otherwise random

    def backpropagation(self, node: MCTSNode, result: float, simulation_moves: List[chess.Move] = None):
        """
        PHASE 4: BACKPROPAGATION
        
        Update statistics for all nodes in the path from root to the simulated node.
        
        BACKPROPAGATION PROCESS:
        - Start at the leaf node where we ran the simulation
        - Work backwards to the root, updating each node's statistics
        - Flip the result at each level (your win is my loss)
        - Update RAVE statistics if enabled
        
        This is how the tree learns: good moves get higher win rates,
        bad moves get lower win rates.
        """
        current = node
        current_result = result
        
        while current is not None:
            # Update this node's statistics
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
            
            # Move up the tree and flip the result (opponent's perspective)
            current_result = -current_result
            current = current.parent

    def mcts_search(self, board: chess.Board) -> Optional[chess.Move]:
        """
        Perform MCTS search and return the best move.
        
        MAIN MCTS LOOP:
        1. Check opening book first (if enabled)
        2. Run MCTS iterations until time limit or max iterations
        3. Each iteration: Selection -> Expansion -> Simulation -> Backpropagation
        4. Pick the move with the most visits (most robust choice)
        
        WHY THIS WORKS:
        - Each iteration gives us more information about the position
        - Good moves get explored more (higher visit counts)
        - Bad moves get abandoned (lower visit counts)
        - The tree grows in the most promising directions
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
        
        # MAIN MCTS LOOP
        while (time.time() - start_time < self.time_limit and 
               iterations < self.max_iterations):
            
            # Make a copy of the board for this iteration
            board_copy = board.copy()
            
            # PHASE 1: SELECTION - walk down tree using UCB1
            selected_node = self.selection(self.root, board_copy)
            
            # PHASE 2: EXPANSION - add a new child node
            expanded_node = self.expansion(selected_node, board_copy)
            
            # PHASE 3: SIMULATION - play random game from new position
            if not expanded_node.is_terminal:
                result, simulation_moves = self.simulation(board_copy)
            else:
                result = expanded_node.terminal_value
                simulation_moves = []
            
            # PHASE 4: BACKPROPAGATION - update statistics up the tree
            self.backpropagation(expanded_node, result, simulation_moves)
            
            iterations += 1
        
        self.total_simulations += iterations
        
        # Select best move: prefer high visit count (robustness) over high win rate
        if not self.root.children:
            return random.choice(list(board.legal_moves)) if board.legal_moves else None
        
        best_move = None
        best_score = float('-inf')
        
        for move, child in self.root.children.items():
            if child.visits == 0:
                continue
            
            win_rate = child.wins / child.visits
            # Combine visits (robustness) with win rate (quality)
            # Higher weight on visits for more robust play in real games
            score = 0.8 * child.visits + 0.2 * win_rate * child.visits
            
            if score > best_score:
                best_score = score
                best_move = move
        
        if best_move is None:
            return random.choice(list(board.legal_moves)) if board.legal_moves else None
        
        # Print some statistics about our search
        best_child = self.root.children[best_move]
        win_rate = best_child.wins / best_child.visits if best_child.visits > 0 else 0
        
        print(f"MCTS completed {iterations} iterations. "
              f"Best move: {best_move.uci()} "
              f"(visits: {best_child.visits}, win rate: {win_rate:.2%})")
        
        return best_move

    def get_best_move(self, board: chess.Board) -> Optional[chess.Move]:
        """
        Main interface method to get the best move.
        
        This is the method that gets called by the game engine.
        It's a simple wrapper around our MCTS search.
        """
        if board.is_game_over():
            return None
        
        return self.mcts_search(board)

    def get_stats(self) -> Dict:
        """Get statistics about the MCTS search for debugging/analysis."""
        return {
            'total_simulations': self.total_simulations,
            'nodes_in_tree': len(self.nodes),
            'root_visits': self.root.visits if self.root else 0
        } 