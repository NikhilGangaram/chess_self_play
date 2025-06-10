import chess
import chess.polyglot
import random
from typing import Optional, Dict, List, Tuple

class MinimaxAgent:
    """
    A chess agent that uses the Minimax algorithm with Alpha-Beta pruning.
    
    CORE CONCEPT: Minimax assumes both players play optimally. It builds a search tree
    where each level alternates between maximizing (our moves) and minimizing (opponent moves).
    We try to find the move that gives us the best outcome assuming our opponent 
    will also choose their best moves.
    
    KEY OPTIMIZATIONS:
    - Alpha-Beta Pruning: Cuts off branches that won't affect the final decision
    - Transposition Tables: Caches previously computed positions to avoid re-computation
    - Quiescence Search: Looks deeper at tactical positions (captures, checks)
    - Move Ordering: Searches promising moves first to improve pruning efficiency
    """
    
    def __init__(self, depth: int = 5):
        """
        Initialize the agent with a depth and a pre-built opening book
        that plays a fixed sequence of 10 moves regardless of opponent moves.
        
        Args:
            depth: How many moves ahead to search (depth 5 = 2.5 moves for each side)
        """
        self.depth = depth
        
        # TRANSPOSITION TABLE: A cache that stores previously evaluated positions
        # Key: zobrist hash of position, Value: (depth, score, best_move)
        # This prevents re-computing the same position multiple times
        self.transposition_table: Dict[int, Tuple[int, int, chess.Move]] = {}

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

        # --- Piece Values (in centipawns, where 100 = 1 pawn) ---
        # These represent the relative strength of each piece type
        self.piece_values = {
            chess.PAWN: 100,    # Base unit of measurement
            chess.KNIGHT: 320,  # Roughly equal to 3 pawns
            chess.BISHOP: 330,  # Slightly stronger than knight (long-range)
            chess.ROOK: 500,    # About 5 pawns worth
            chess.QUEEN: 900,   # Most powerful piece, about 9 pawns
            chess.KING: 20000   # Invaluable - losing it ends the game
        }

        # --- Piece-Square Tables ---
        # These tables give bonuses/penalties for pieces on different squares
        # The idea: a knight on e4 is much stronger than a knight on a1
        self.pawn_table = [
              0,  0,  0,  0,  0,  0,  0,  0,  # 8th rank: pawns can't be here
             50, 50, 50, 50, 50, 50, 50, 50,  # 7th rank: about to promote!
             10, 10, 20, 30, 30, 20, 10, 10,  # 6th rank: advanced pawns are strong
              5,  5, 10, 25, 25, 10,  5,  5,  # 5th rank: good pawn advancement
              0,  0,  0, 20, 20,  0,  0,  0,  # 4th rank: center pawns get bonus
              5, -5,-10,  0,  0,-10, -5,  5,  # 3rd rank: slight penalty for early moves
              5, 10, 10,-20,-20, 10, 10,  5,  # 2nd rank: penalty for blocking development
              0,  0,  0,  0,  0,  0,  0,  0   # 1st rank: starting position
        ]
        
        # Knight table: Knights are strongest in the center, weak on edges
        self.knight_table = [-50,-40,-30,-30,-30,-30,-40,-50,-40,-20,  0,  5,  5,  0,-20,-40,-30,  5, 10, 15, 15, 10,  5,-30,-30,  0, 15, 20, 20, 15,  0,-30,-30,  5, 15, 20, 20, 15,  5,-30,-30,  0, 10, 15, 15, 10,  0,-30,-40,-20,  0,  0,  0,  0,-20,-40,-50,-40,-30,-30,-30,-30,-40,-50]
        
        # Bishop table: Bishops prefer long diagonals and central squares
        self.bishop_table = [-20,-10,-10,-10,-10,-10,-10,-20,-10,  0,  0,  0,  0,  0,  0,-10,-10,  0,  5, 10, 10,  5,  0,-10,-10,  5,  5, 10, 10,  5,  5,-10,-10,  0, 10, 10, 10, 10,  0,-10,-10, 10, 10, 10, 10, 10, 10,-10,-10,  5,  0,  0,  0,  0,  5,-10,-20,-10,-10,-10,-10,-10,-10,-20]
        
        # Rook table: Rooks want open files and 7th rank
        self.rook_table = [ 0,  0,  0,  5,  5,  0,  0,  0,-5,  0,  0,  0,  0,  0,  0, -5,-5,  0,  0,  0,  0,  0,  0, -5,-5,  0,  0,  0,  0,  0,  0, -5,-5,  0,  0,  0,  0,  0,  0, -5,-5,  0,  0,  0,  0,  0,  0, -5, 5, 10, 10, 10, 10, 10, 10,  5, 0,  0,  0,  0,  0,  0,  0,  0]
        
        # Queen table: Similar to rook/bishop combined, prefers center
        self.queen_table = [-20,-10,-10, -5, -5,-10,-10,-20,-10,  0,  0,  0,  0,  0,  0,-10,-10,  0,  5,  5,  5,  5,  0,-10, -5,  0,  5,  5,  5,  5,  0, -5,  0,  0,  5,  5,  5,  5,  0, -5,-10,  5,  5,  5,  5,  5,  0,-10,-10,  0,  5,  0,  0,  0,  0,-10,-20,-10,-10, -5, -5,-10,-10,-20]
        
        # King tables: Different for middlegame (mg) and endgame (eg)
        # Middlegame: King wants to stay safe (corner/back rank)
        self.king_mg_table = [-30,-40,-40,-50,-50,-40,-40,-30,-30,-40,-40,-50,-50,-40,-40,-30,-30,-40,-40,-50,-50,-40,-40,-30,-30,-40,-40,-50,-50,-40,-40,-30,-20,-30,-30,-40,-40,-30,-30,-20,-10,-20,-20,-20,-20,-20,-20,-10, 20, 20,  0,  0,  0,  0, 20, 20, 20, 30, 10,  0,  0, 10, 30, 20]
        
        # Endgame: King becomes active, wants to centralize
        self.king_eg_table = [-50,-40,-30,-20,-20,-30,-40,-50,-30,-20,-10,  0,  0,-10,-20,-30,-30,-10, 20, 30, 30, 20,-10,-30,-30,-10, 30, 40, 40, 30,-10,-30,-30,-10, 30, 40, 40, 30,-10,-30,-30,-10, 20, 30, 30, 20,-10,-30,-30,-30,  0,  0,  0,  0,-30,-30,-50,-30,-30,-30,-30,-30,-30,-50]

    def get_opening_move(self, board: chess.Board) -> Optional[chess.Move]:
        """
        Get the next move from our fixed opening sequence based on whose turn it is.
        
        WHY USE OPENING BOOKS: 
        - Avoids spending time computing well-known opening theory
        - Ensures we don't make obvious mistakes in the opening
        - Gets us to interesting middlegame positions faster
        """
        if board.fullmove_number <= 7:  # First 7 moves from opening book
            # Choose the appropriate move list based on which color is to move
            if board.turn == chess.WHITE:
                move_uci = self.opening_moves_white[board.fullmove_number - 1]
            else:
                move_uci = self.opening_moves_black[board.fullmove_number - 1]
            
            move = chess.Move.from_uci(move_uci)
            
            # Safety check: make sure the opening book move is actually legal
            if move in board.legal_moves:
                return move
        
        return None

    def get_game_phase(self, board: chess.Board) -> float:
        """
        Determine what phase of the game we're in (0.0 = opening/middlegame, 1.0 = endgame).
        
        GAME PHASES MATTER: Different pieces have different values in different phases.
        For example, kings should hide in the opening but become active in the endgame.
        """
        # Count material on the board (excluding pawns and kings)
        piece_count = 0
        for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            piece_count += len(board.pieces(piece_type, chess.WHITE))
            piece_count += len(board.pieces(piece_type, chess.BLACK))
        
        # 24 = maximum material (4 knights + 4 bishops + 4 rooks + 2 queens)
        # As pieces get traded off, we move toward endgame
        phase = max(0, 24 - piece_count) / 24.0
        return phase

    def evaluate_board(self, board: chess.Board) -> int:
        """
        Evaluate how good a chess position is from the current player's perspective.
        
        EVALUATION COMPONENTS:
        1. Material balance (who has more/better pieces)
        2. Piece positioning (piece-square tables)
        3. Game phase considerations (opening vs endgame)
        4. Special situations (checkmate, stalemate)
        
        Returns: Positive number = good for current player, Negative = bad
        """
        # TERMINAL POSITIONS: Game-ending situations get extreme scores
        if board.is_checkmate(): 
            return -99999 if board.turn else 99999  # Losing is very bad!
        if board.is_stalemate() or board.is_insufficient_material(): 
            return 0  # Draws are neutral
        
        # Determine what phase of the game we're in
        game_phase = self.get_game_phase(board)
        
        # Calculate material and positional evaluation for both sides
        white_eval = 0
        black_eval = 0
        
        # Go through every square on the board
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None: 
                continue  # Empty square
            
            # Start with basic piece value
            value = self.piece_values[piece.piece_type]
            
            # Add positional bonus based on piece-square tables
            if piece.piece_type == chess.PAWN: 
                pos_value = self.pawn_table[square]
            elif piece.piece_type == chess.KNIGHT: 
                pos_value = self.knight_table[square]
            elif piece.piece_type == chess.BISHOP: 
                pos_value = self.bishop_table[square]
            elif piece.piece_type == chess.ROOK: 
                pos_value = self.rook_table[square]
            elif piece.piece_type == chess.QUEEN: 
                pos_value = self.queen_table[square]
            elif piece.piece_type == chess.KING:
                # King evaluation depends on game phase
                mg_value = self.king_mg_table[square]  # Middlegame table
                eg_value = self.king_eg_table[square]  # Endgame table
                # Interpolate between middlegame and endgame values
                pos_value = int((1 - game_phase) * mg_value + game_phase * eg_value)
            
            # Add to the appropriate side's evaluation
            if piece.color == chess.WHITE: 
                white_eval += value + pos_value
            else:
                # For black pieces, flip the square (black's perspective)
                flipped_square = chess.square_mirror(square)
                if piece.piece_type == chess.PAWN: 
                    pos_value = self.pawn_table[flipped_square]
                elif piece.piece_type == chess.KNIGHT: 
                    pos_value = self.knight_table[flipped_square]
                elif piece.piece_type == chess.BISHOP: 
                    pos_value = self.bishop_table[flipped_square]
                elif piece.piece_type == chess.ROOK: 
                    pos_value = self.rook_table[flipped_square]
                elif piece.piece_type == chess.QUEEN: 
                    pos_value = self.queen_table[flipped_square]
                elif piece.piece_type == chess.KING:
                    mg_value = self.king_mg_table[flipped_square]
                    eg_value = self.king_eg_table[flipped_square]
                    pos_value = int((1 - game_phase) * mg_value + game_phase * eg_value)
                black_eval += value + pos_value
        
        # Final score: positive = white is better, negative = black is better
        score = white_eval - black_eval
        
        # Return from current player's perspective
        return score if board.turn == chess.WHITE else -score

    def order_moves(self, board: chess.Board) -> List[chess.Move]:
        """
        Order moves to search the most promising ones first.
        
        WHY MOVE ORDERING MATTERS:
        Alpha-beta pruning works much better when we search good moves first.
        If we find a great move early, we can skip searching many bad moves.
        
        MOVE ORDERING HEURISTICS:
        1. Captures (especially good trades: take queen with pawn)
        2. Promotions (pawns becoming queens)
        3. Other moves in random order
        """
        moves = list(board.legal_moves)
        
        def move_score(move):
            """Calculate a rough score for move ordering (higher = search first)."""
            score = 0
            
            # CAPTURES: Taking opponent pieces is often good
            if board.is_capture(move):
                victim = board.piece_at(move.to_square)      # What we're capturing
                attacker = board.piece_at(move.from_square)  # What we're using to capture
                
                if victim and attacker: 
                    # MVV-LVA: Most Valuable Victim - Least Valuable Attacker
                    # Taking a queen with a pawn is better than taking a pawn with a queen
                    score += 10 * self.piece_values[victim.piece_type] - self.piece_values[attacker.piece_type]
            
            # PROMOTIONS: Pawns becoming queens are very valuable
            if move.promotion: 
                score += self.piece_values[move.promotion]
            
            return score
        
        # Sort moves by score (highest first)
        return sorted(moves, key=move_score, reverse=True)

    def quiescence_search(self, board: chess.Board, alpha: float, beta: float) -> int:
        """
        Search only "noisy" moves (captures, promotions) until the position is quiet.
        
        THE HORIZON EFFECT PROBLEM:
        If we stop searching in the middle of a sequence like:
        1. Take queen with bishop
        2. Retake bishop with pawn
        
        We might think taking the queen is great, but miss that we lose our bishop!
        
        QUIESCENCE SEARCH SOLUTION:
        Keep searching captures and promotions until the position is "quiet"
        (no immediate tactical shots available).
        """
        # "Stand pat" evaluation: assume we don't make any more moves
        stand_pat_score = self.evaluate_board(board)
        
        # Beta cutoff: position is already too good for the opponent
        if stand_pat_score >= beta: 
            return beta
        
        # Update alpha if our current position is our best so far
        if alpha < stand_pat_score: 
            alpha = stand_pat_score
        
        # Only search captures and promotions (the "noisy" moves)
        capture_moves = [move for move in self.order_moves(board) 
                        if board.is_capture(move) or move.promotion]
        
        for move in capture_moves:
            board.push(move)
            score = -self.quiescence_search(board, -beta, -alpha)  # Recursively search
            board.pop()
            
            # Alpha-beta pruning in quiescence search too
            if score >= beta: 
                return beta
            if score > alpha: 
                alpha = score
        
        return alpha

    def minimax(self, board: chess.Board, depth: int, alpha: float, beta: float) -> Tuple[int, Optional[chess.Move]]:
        """
        The core Minimax algorithm with Alpha-Beta pruning.
        
        MINIMAX CONCEPT:
        - At our turn (maximizing): Choose the move that gives us the highest score
        - At opponent's turn (minimizing): Assume they choose the move that gives us the lowest score
        - Recursively build a tree of all possible move sequences up to 'depth' moves
        
        ALPHA-BETA PRUNING:
        - Alpha: The best score WE can guarantee so far
        - Beta: The best score OUR OPPONENT can guarantee so far
        - If alpha >= beta, we can stop searching this branch (opponent won't allow it)
        
        TRANSPOSITION TABLE:
        - Cache results to avoid recomputing the same position multiple times
        - Chess positions can be reached by different move orders (transpositions)
        """
        # Store original alpha for transposition table entry type
        alpha_orig = alpha
        
        # Check if we've seen this position before
        zobrist_hash = chess.polyglot.zobrist_hash(board)
        if zobrist_hash in self.transposition_table:
            stored_depth, stored_score, stored_move = self.transposition_table[zobrist_hash]
            
            # Only use cached result if it was searched to at least the same depth
            if stored_depth >= depth: 
                return stored_score, stored_move
        
        # BASE CASE: We've reached our search depth or the game is over
        if depth == 0 or board.is_game_over():
            # Use quiescence search to avoid horizon effect
            score = self.quiescence_search(board, alpha, beta)
            return score, None
        
        # RECURSIVE CASE: Try all possible moves
        best_move = None
        max_eval = float('-inf')  # Start with worst possible score
        
        # Try each legal move (in order of most promising first)
        for move in self.order_moves(board):
            # Make the move and recursively evaluate the resulting position
            board.push(move)
            eval_score, _ = self.minimax(board, depth - 1, -beta, -alpha)
            eval_score = -eval_score  # Flip sign (opponent's best is our worst)
            board.pop()  # Undo the move
            
            # Check if this is our new best move
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
            
            # Update alpha (our best guaranteed score)
            alpha = max(alpha, eval_score)
            
            # ALPHA-BETA PRUNING: If alpha >= beta, opponent won't allow this line
            if alpha >= beta: 
                break  # Cut off the search (beta cutoff)
        
        # Store result in transposition table for future use
        self.transposition_table[zobrist_hash] = (depth, max_eval, best_move)
        
        return max_eval, best_move

    def get_best_move(self, board: chess.Board) -> Optional[chess.Move]:
        """
        Main method to get the best move for the current position.
        
        DECISION PROCESS:
        1. Check if game is over (no moves available)
        2. Try opening book first (for speed and good opening play)
        3. Fall back to minimax search for original thinking
        """
        if board.is_game_over():
            return None

        # --- ENFORCE OPENING BOOK ---
        # Use pre-computed opening moves for the first 7 moves
        if board.fullmove_number <= 7:
            opening_move = self.get_opening_move(board)
            if opening_move:
                print(f"Agent plays '{opening_move.uci()}' from its opening book.")
                return opening_move

        # --- FALLBACK TO MINIMAX SEARCH ---
        print("Agent is thinking... (using minimax search)")
        
        # Clear transposition table for fresh search
        # (Could keep it for move-to-move persistence, but memory management is simpler this way)
        self.transposition_table.clear()
        
        # Run the minimax algorithm
        _, best_move = self.minimax(board, self.depth, float('-inf'), float('inf'))

        return best_move