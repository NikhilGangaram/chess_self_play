import chess
import random
from typing import Optional, Dict, List, Tuple

class MinimaxAgent:
    def __init__(self, depth: int = 4):
        self.depth = depth
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }
        
        # Enhanced position tables
        self.pawn_table = [
            0,  0,  0,  0,  0,  0,  0,  0,
            50, 50, 50, 50, 50, 50, 50, 50,
            10, 10, 20, 30, 30, 20, 10, 10,
            5,  5, 10, 27, 27, 10,  5,  5,
            0,  0,  0, 25, 25,  0,  0,  0,
            5, -5,-10,  0,  0,-10, -5,  5,
            5, 10, 10,-25,-25, 10, 10,  5,
            0,  0,  0,  0,  0,  0,  0,  0
        ]
        
        self.knight_table = [
            -50,-40,-30,-30,-30,-30,-40,-50,
            -40,-20,  0,  0,  0,  0,-20,-40,
            -30,  0, 10, 15, 15, 10,  0,-30,
            -30,  5, 15, 20, 20, 15,  5,-30,
            -30,  0, 15, 20, 20, 15,  0,-30,
            -30,  5, 10, 15, 15, 10,  5,-30,
            -40,-20,  0,  5,  5,  0,-20,-40,
            -50,-40,-30,-30,-30,-30,-40,-50,
        ]
        
        self.bishop_table = [
            -20,-10,-10,-10,-10,-10,-10,-20,
            -10,  0,  0,  0,  0,  0,  0,-10,
            -10,  0,  5, 10, 10,  5,  0,-10,
            -10,  5,  5, 10, 10,  5,  5,-10,
            -10,  0, 10, 10, 10, 10,  0,-10,
            -10, 10, 10, 10, 10, 10, 10,-10,
            -10,  5,  0,  0,  0,  0,  5,-10,
            -20,-10,-10,-10,-10,-10,-10,-20,
        ]
        
        self.rook_table = [
            0,  0,  0,  0,  0,  0,  0,  0,
            5, 10, 10, 10, 10, 10, 10,  5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            0,  0,  0,  5,  5,  0,  0,  0
        ]
        
        # Opening book - solid, classical openings
        self.opening_book = {
            # Italian Game
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1": ["e2e4"],
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2": ["g1f3"],
            "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3": ["f1c4"],
            "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4": ["d2d3", "c2c3"],
            
            # Ruy Lopez
            "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3": ["f1b5"],
            "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3": ["a7a6"],
            "r1bqkbnr/1ppp1ppp/p1n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4": ["b5a4"],
            
            # Queen's Gambit
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1": ["d2d4"],
            "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2": ["c2c4"],
            "rnbqkbnr/ppp2ppp/8/3pp3/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3": ["b1c3", "g1f3"],
            
            # Sicilian Defense (for Black)
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2": ["g1f3"],
            "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2": ["g1f3"],
            
            # King's Indian Defense
            "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2": ["g1f3"],
            "rnbqkb1r/ppp1pp1p/5np1/3p4/3P4/5N2/PPP1PPPP/RNBQKB1R w KQkq - 0 4": ["c2c4"],
        }
    
    def get_opening_move(self, board: chess.Board) -> Optional[chess.Move]:
        """Get a move from the opening book if available"""
        board_fen = board.fen()
        
        # Check if current position is in opening book
        if board_fen in self.opening_book:
            move_options = self.opening_book[board_fen]
            move_uci = random.choice(move_options)
            try:
                move = chess.Move.from_uci(move_uci)
                if move in board.legal_moves:
                    return move
            except:
                pass
        
        return None
    
    def order_moves(self, board: chess.Board, moves: List[chess.Move]) -> List[chess.Move]:
        """Order moves for better alpha-beta pruning (best moves first)"""
        def move_priority(move):
            priority = 0
            
            # Captures (highest priority)
            if board.is_capture(move):
                captured_piece = board.piece_at(move.to_square)
                if captured_piece:
                    # MVV-LVA: Most Valuable Victim - Least Valuable Attacker
                    attacker_value = self.piece_values.get(board.piece_at(move.from_square).piece_type, 0)
                    victim_value = self.piece_values.get(captured_piece.piece_type, 0)
                    priority += 10000 + victim_value - attacker_value
            
            # Checks
            board.push(move)
            if board.is_check():
                priority += 5000
            board.pop()
            
            # Promotions
            if move.promotion:
                priority += 8000
            
            # Castle
            if board.is_castling(move):
                priority += 3000
            
            # Central squares
            to_square = move.to_square
            if to_square in [chess.E4, chess.E5, chess.D4, chess.D5]:
                priority += 100
            elif to_square in [chess.C4, chess.C5, chess.F4, chess.F5,
                             chess.D3, chess.D6, chess.E3, chess.E6]:
                priority += 50
            
            return priority
        
        return sorted(moves, key=move_priority, reverse=True)
    
    def evaluate_board(self, board: chess.Board) -> float:
        """Enhanced board evaluation"""
        if board.is_checkmate():
            return -999999 if board.turn else 999999
        
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        
        score = 0
        
        # Material and positional evaluation
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None:
                continue
                
            piece_value = self.piece_values[piece.piece_type]
            
            # Positional bonuses
            position_bonus = 0
            if piece.piece_type == chess.PAWN:
                position_bonus = self.pawn_table[square if piece.color else 63 - square]
            elif piece.piece_type == chess.KNIGHT:
                position_bonus = self.knight_table[square if piece.color else 63 - square]
            elif piece.piece_type == chess.BISHOP:
                position_bonus = self.bishop_table[square if piece.color else 63 - square]
            elif piece.piece_type == chess.ROOK:
                position_bonus = self.rook_table[square if piece.color else 63 - square]
            
            total_value = piece_value + position_bonus
            
            if piece.color == chess.WHITE:
                score += total_value
            else:
                score -= total_value
        
        # King safety
        white_king_square = board.king(chess.WHITE)
        black_king_square = board.king(chess.BLACK)
        
        if white_king_square and black_king_square:
            # Penalize exposed kings in endgame
            piece_count = len(board.piece_map())
            if piece_count < 12:  # Endgame
                # King should be active in endgame
                white_king_center_distance = abs(chess.square_file(white_king_square) - 3.5) + abs(chess.square_rank(white_king_square) - 3.5)
                black_king_center_distance = abs(chess.square_file(black_king_square) - 3.5) + abs(chess.square_rank(black_king_square) - 3.5)
                score -= white_king_center_distance * 10
                score += black_king_center_distance * 10
        
        # Mobility bonus
        legal_moves_count = len(list(board.legal_moves))
        if board.turn == chess.WHITE:
            score += legal_moves_count * 10
        else:
            score -= legal_moves_count * 10
        
        # Check bonus
        if board.is_check():
            score += 50 if board.turn == chess.BLACK else -50
        
        return score
    
    def minimax(self, board: chess.Board, depth: int, alpha: float, beta: float, maximizing: bool) -> float:
        """Enhanced minimax with move ordering"""
        if depth == 0 or board.is_game_over():
            return self.evaluate_board(board)
        
        legal_moves = list(board.legal_moves)
        legal_moves = self.order_moves(board, legal_moves)  # Move ordering for better pruning
        
        if maximizing:
            max_eval = float('-inf')
            for move in legal_moves:
                board.push(move)
                eval_score = self.minimax(board, depth - 1, alpha, beta, False)
                board.pop()
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Alpha-beta pruning
            return max_eval
        else:
            min_eval = float('inf')
            for move in legal_moves:
                board.push(move)
                eval_score = self.minimax(board, depth - 1, alpha, beta, True)
                board.pop()
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha-beta pruning
            return min_eval
    
    def get_best_move(self, board: chess.Board) -> Optional[chess.Move]:
        """Get the best move, using opening book first, then search"""
        if board.is_game_over():
            return None
        
        # Try opening book first (for first ~10 moves)
        if board.fullmove_number <= 10:
            opening_move = self.get_opening_move(board)
            if opening_move:
                return opening_move
        
        # Fall back to minimax search
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        
        # Order moves for better search
        legal_moves = self.order_moves(board, legal_moves)
        
        best_move = None
        best_value = float('-inf') if board.turn == chess.WHITE else float('inf')
        
        for move in legal_moves:
            board.push(move)
            move_value = self.minimax(board, self.depth - 1, float('-inf'), float('inf'), 
                                    not board.turn)
            board.pop()
            
            if board.turn == chess.WHITE:  # Maximizing for white
                if move_value > best_value:
                    best_value = move_value
                    best_move = move
            else:  # Minimizing for black
                if move_value < best_value:
                    best_value = move_value
                    best_move = move
        
        return best_move 