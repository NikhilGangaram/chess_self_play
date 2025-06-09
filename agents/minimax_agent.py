import chess
import chess.polyglot
import random
from typing import Optional, Dict, List, Tuple

class MinimaxAgent:
    def __init__(self, depth: int = 3):
        """
        Initialize the agent with a depth and a pre-built opening book
        that plays a fixed sequence of 10 moves regardless of opponent moves.
        """
        self.depth = depth
        self.transposition_table: Dict[int, Tuple[int, int, chess.Move]] = {}

        # --- Fixed opening sequence (7 moves) ---
        self.opening_moves_white = [
            "e2e4",  # Move 1: King's pawn
            "d2d4",  # Move 2: Queen's pawn  
            "b1c3",  # Move 3: Knight development
            "g1f3",  # Move 4: Knight development
            "f1c4",  # Move 5: Bishop development
            "e1g1",  # Move 6: Castle kingside
            "a2a3",  # Move 7: Prepare b4
        ]
        self.opening_moves_black = [
            "e7e5",  # Move 1: King's pawn
            "d7d5",  # Move 2: Queen's pawn  
            "b8c6",  # Move 3: Knight development
            "g8f6",  # Move 4: Knight development
            "f8c5",  # Move 5: Bishop development
            "e8g8",  # Move 6: Castle kingside
            "a7a6",  # Move 7: Prepare b5
        ]
        self.move_count = 0  # Track how many moves this agent has made

        # --- Piece Values ---
        self.piece_values = {
            chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
            chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 20000
        }

        # --- Piece-Square Tables ---
        self.pawn_table = [
              0,  0,  0,  0,  0,  0,  0,  0,
             50, 50, 50, 50, 50, 50, 50, 50,
             10, 10, 20, 30, 30, 20, 10, 10,
              5,  5, 10, 25, 25, 10,  5,  5,
              0,  0,  0, 20, 20,  0,  0,  0,
              5, -5,-10,  0,  0,-10, -5,  5,
              5, 10, 10,-20,-20, 10, 10,  5,
              0,  0,  0,  0,  0,  0,  0,  0
        ]
        self.knight_table = [-50,-40,-30,-30,-30,-30,-40,-50,-40,-20,  0,  5,  5,  0,-20,-40,-30,  5, 10, 15, 15, 10,  5,-30,-30,  0, 15, 20, 20, 15,  0,-30,-30,  5, 15, 20, 20, 15,  5,-30,-30,  0, 10, 15, 15, 10,  0,-30,-40,-20,  0,  0,  0,  0,-20,-40,-50,-40,-30,-30,-30,-30,-40,-50]
        self.bishop_table = [-20,-10,-10,-10,-10,-10,-10,-20,-10,  0,  0,  0,  0,  0,  0,-10,-10,  0,  5, 10, 10,  5,  0,-10,-10,  5,  5, 10, 10,  5,  5,-10,-10,  0, 10, 10, 10, 10,  0,-10,-10, 10, 10, 10, 10, 10, 10,-10,-10,  5,  0,  0,  0,  0,  5,-10,-20,-10,-10,-10,-10,-10,-10,-20]
        self.rook_table = [ 0,  0,  0,  5,  5,  0,  0,  0,-5,  0,  0,  0,  0,  0,  0, -5,-5,  0,  0,  0,  0,  0,  0, -5,-5,  0,  0,  0,  0,  0,  0, -5,-5,  0,  0,  0,  0,  0,  0, -5,-5,  0,  0,  0,  0,  0,  0, -5, 5, 10, 10, 10, 10, 10, 10,  5, 0,  0,  0,  0,  0,  0,  0,  0]
        self.queen_table = [-20,-10,-10, -5, -5,-10,-10,-20,-10,  0,  0,  0,  0,  0,  0,-10,-10,  0,  5,  5,  5,  5,  0,-10, -5,  0,  5,  5,  5,  5,  0, -5,  0,  0,  5,  5,  5,  5,  0, -5,-10,  5,  5,  5,  5,  5,  0,-10,-10,  0,  5,  0,  0,  0,  0,-10,-20,-10,-10, -5, -5,-10,-10,-20]
        self.king_mg_table = [-30,-40,-40,-50,-50,-40,-40,-30,-30,-40,-40,-50,-50,-40,-40,-30,-30,-40,-40,-50,-50,-40,-40,-30,-30,-40,-40,-50,-50,-40,-40,-30,-20,-30,-30,-40,-40,-30,-30,-20,-10,-20,-20,-20,-20,-20,-20,-10, 20, 20,  0,  0,  0,  0, 20, 20, 20, 30, 10,  0,  0, 10, 30, 20]
        self.king_eg_table = [-50,-40,-30,-20,-20,-30,-40,-50,-30,-20,-10,  0,  0,-10,-20,-30,-30,-10, 20, 30, 30, 20,-10,-30,-30,-10, 30, 40, 40, 30,-10,-30,-30,-10, 30, 40, 40, 30,-10,-30,-30,-10, 20, 30, 30, 20,-10,-30,-30,-30,  0,  0,  0,  0,-30,-30,-50,-30,-30,-30,-30,-30,-30,-50]

    def get_opening_move(self, board: chess.Board) -> Optional[chess.Move]:
        """
        Get the next move from our fixed opening sequence based on whose turn it is.
        """
        if self.move_count < 7:  # First 7 moves from opening book
            # Choose the appropriate move list based on which color is to move
            if board.turn == chess.WHITE:
                move_uci = self.opening_moves_white[self.move_count]
            else:
                move_uci = self.opening_moves_black[self.move_count]
            
            move = chess.Move.from_uci(move_uci)
            
            if move in board.legal_moves:
                self.move_count += 1
                return move
        
        return None

    def get_game_phase(self, board: chess.Board) -> float:
        piece_count = 0
        for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            piece_count += len(board.pieces(piece_type, chess.WHITE))
            piece_count += len(board.pieces(piece_type, chess.BLACK))
        phase = max(0, 24 - piece_count) / 24.0
        return phase

    def evaluate_board(self, board: chess.Board) -> int:
        if board.is_checkmate(): return -99999 if board.turn else 99999
        if board.is_stalemate() or board.is_insufficient_material(): return 0
        game_phase = self.get_game_phase(board)
        white_eval = 0
        black_eval = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None: continue
            value = self.piece_values[piece.piece_type]
            if piece.piece_type == chess.PAWN: pos_value = self.pawn_table[square]
            elif piece.piece_type == chess.KNIGHT: pos_value = self.knight_table[square]
            elif piece.piece_type == chess.BISHOP: pos_value = self.bishop_table[square]
            elif piece.piece_type == chess.ROOK: pos_value = self.rook_table[square]
            elif piece.piece_type == chess.QUEEN: pos_value = self.queen_table[square]
            elif piece.piece_type == chess.KING:
                mg_value = self.king_mg_table[square]
                eg_value = self.king_eg_table[square]
                pos_value = int((1 - game_phase) * mg_value + game_phase * eg_value)
            if piece.color == chess.WHITE: white_eval += value + pos_value
            else:
                flipped_square = chess.square_mirror(square)
                if piece.piece_type == chess.PAWN: pos_value = self.pawn_table[flipped_square]
                elif piece.piece_type == chess.KNIGHT: pos_value = self.knight_table[flipped_square]
                elif piece.piece_type == chess.BISHOP: pos_value = self.bishop_table[flipped_square]
                elif piece.piece_type == chess.ROOK: pos_value = self.rook_table[flipped_square]
                elif piece.piece_type == chess.QUEEN: pos_value = self.queen_table[flipped_square]
                elif piece.piece_type == chess.KING:
                    mg_value = self.king_mg_table[flipped_square]
                    eg_value = self.king_eg_table[flipped_square]
                    pos_value = int((1 - game_phase) * mg_value + game_phase * eg_value)
                black_eval += value + pos_value
        score = white_eval - black_eval
        return score if board.turn == chess.WHITE else -score

    def order_moves(self, board: chess.Board) -> List[chess.Move]:
        moves = list(board.legal_moves)
        def move_score(move):
            score = 0
            if board.is_capture(move):
                victim = board.piece_at(move.to_square)
                attacker = board.piece_at(move.from_square)
                if victim and attacker: score += 10 * self.piece_values[victim.piece_type] - self.piece_values[attacker.piece_type]
            if move.promotion: score += self.piece_values[move.promotion]
            return score
        return sorted(moves, key=move_score, reverse=True)

    def quiescence_search(self, board: chess.Board, alpha: float, beta: float) -> int:
        stand_pat_score = self.evaluate_board(board)
        if stand_pat_score >= beta: return beta
        if alpha < stand_pat_score: alpha = stand_pat_score
        capture_moves = [move for move in self.order_moves(board) if board.is_capture(move) or move.promotion]
        for move in capture_moves:
            board.push(move)
            score = -self.quiescence_search(board, -beta, -alpha)
            board.pop()
            if score >= beta: return beta
            if score > alpha: alpha = score
        return alpha

    def minimax(self, board: chess.Board, depth: int, alpha: float, beta: float) -> Tuple[int, Optional[chess.Move]]:
        alpha_orig = alpha
        zobrist_hash = chess.polyglot.zobrist_hash(board)
        if zobrist_hash in self.transposition_table:
            stored_depth, stored_score, stored_move = self.transposition_table[zobrist_hash]
            if stored_depth >= depth: return stored_score, stored_move
        if depth == 0 or board.is_game_over():
            score = self.quiescence_search(board, alpha, beta)
            return score, None
        best_move = None
        max_eval = float('-inf')
        for move in self.order_moves(board):
            board.push(move)
            eval_score, _ = self.minimax(board, depth - 1, -beta, -alpha)
            eval_score = -eval_score
            board.pop()
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
            alpha = max(alpha, eval_score)
            if alpha >= beta: break
        self.transposition_table[zobrist_hash] = (depth, max_eval, best_move)
        return max_eval, best_move

    def get_best_move(self, board: chess.Board) -> Optional[chess.Move]:
        if board.is_game_over():
            return None

        # --- ENFORCE OPENING BOOK ---
        if self.move_count < 7:
            opening_move = self.get_opening_move(board)
            if opening_move:
                print(f"Agent plays '{opening_move.uci()}' from its opening book.")
                return opening_move

        # --- FALLBACK TO MINIMAX SEARCH ---
        print("Agent is thinking... (using minimax search)")
        self.transposition_table.clear()
        _, best_move = self.minimax(board, self.depth, float('-inf'), float('inf'))

        return best_move