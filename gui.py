import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QLabel, QComboBox, QGroupBox
from PyQt6.QtSvgWidgets import QSvgWidget
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QMouseEvent
import chess
import chess.svg
from agents.minimax_agent import MinimaxAgent

class ClickableSvgWidget(QSvgWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_gui = parent
        self.board_size = 350  # SVG board size
    
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            # Get click position relative to widget
            x = event.position().x()
            y = event.position().y()
            
            # Convert to board coordinates (assuming square widget)
            widget_size = min(self.width(), self.height())
            board_x = (x / widget_size) * 8
            board_y = (y / widget_size) * 8
            
            # Convert to chess square (flip y-axis since chess board is flipped)
            file = int(board_x)
            rank = 7 - int(board_y)
            
            if 0 <= file <= 7 and 0 <= rank <= 7:
                square = chess.square(file, rank)
                if self.parent_gui:
                    self.parent_gui.on_square_clicked(square)
        
        super().mousePressEvent(event)

class ChessGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Chess Board GUI - PyQt")
        self.setGeometry(100, 100, 600, 800)
        
        # Game state
        self.board = chess.Board()
        self.selected_square = None
        self.legal_moves = []
        
        # AI and game mode
        self.ai = MinimaxAgent(depth=3)
        self.game_mode = "Human vs Human"  # "Human vs Human", "Human vs AI", "AI vs AI"
        self.human_color = chess.WHITE  # For Human vs AI mode
        self.ai_thinking = False
        self.game_paused = False
        
        # Timer for AI moves
        self.ai_timer = QTimer()
        self.ai_timer.timeout.connect(self.make_ai_move)
        self.ai_timer.setSingleShot(True)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Game mode selection
        mode_group = QGroupBox("")
        mode_layout = QHBoxLayout(mode_group)
        
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Human vs Human", "Human vs AI", "AI vs AI"])
        self.mode_combo.currentTextChanged.connect(self.on_mode_changed)
        mode_layout.addWidget(QLabel("Mode:"))
        mode_layout.addWidget(self.mode_combo)
        
        self.color_combo = QComboBox()
        self.color_combo.addItems(["Play as White", "Play as Black"])
        self.color_combo.currentTextChanged.connect(self.on_color_changed)
        self.color_combo.setEnabled(False)  # Initially disabled
        mode_layout.addWidget(QLabel("Color:"))
        mode_layout.addWidget(self.color_combo)
        
        layout.addWidget(mode_group)
        
        # SVG Widget for the chess board
        self.svg_widget = ClickableSvgWidget(self)
        self.svg_widget.setFixedSize(400, 400)
        layout.addWidget(self.svg_widget, alignment=Qt.AlignmentFlag.AlignCenter)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        refresh_btn = QPushButton("Refresh Board")
        refresh_btn.clicked.connect(self.update_board)
        button_layout.addWidget(refresh_btn)
        
        reset_btn = QPushButton("Reset Game")
        reset_btn.clicked.connect(self.reset_game)
        button_layout.addWidget(reset_btn)
        
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.clicked.connect(self.toggle_pause)
        button_layout.addWidget(self.pause_btn)
        
        export_btn = QPushButton("Export SVG")
        export_btn.clicked.connect(self.export_svg)
        button_layout.addWidget(export_btn)
        
        layout.addLayout(button_layout)
        
        # Game info
        self.info_label = QLabel()
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info_label.setStyleSheet("margin: 5px; font-weight: bold;")
        layout.addWidget(self.info_label)
        
        # Status label
        self.status_label = QLabel("Click a piece to see its possible moves")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("margin: 10px;")
        layout.addWidget(self.status_label)
        
        # Initial board render
        self.update_board()
        self.update_info()
        
        # Start AI if needed
        self.check_ai_turn()
    
    def toggle_pause(self):
        """Toggle game pause state"""
        self.game_paused = not self.game_paused
        
        if self.game_paused:
            self.pause_btn.setText("Resume")
            self.ai_timer.stop()  # Stop AI timer if running
            self.status_label.setText("Game Paused - Click Resume to continue")
        else:
            self.pause_btn.setText("Pause")
            if self.game_mode == "Human vs Human":
                self.status_label.setText("Game resumed. Click a piece to see its moves.")
            elif self.game_mode == "AI vs AI":
                self.status_label.setText("Game resumed. AI vs AI mode - Watch them play!")
            else:  # Human vs AI
                color_str = "White" if self.human_color == chess.WHITE else "Black"
                self.status_label.setText(f"Game resumed. You are playing as {color_str}")
            
            # Check if AI should move after resuming
            self.check_ai_turn()

    def on_mode_changed(self, mode):
        """Handle game mode changes"""
        self.game_mode = mode
        self.color_combo.setEnabled(mode == "Human vs AI")
        
        if mode == "Human vs AI":
            self.on_color_changed(self.color_combo.currentText())
        
        self.reset_game()
    
    def on_color_changed(self, color_text):
        """Handle color selection for Human vs AI mode"""
        self.human_color = chess.WHITE if color_text == "Play as White" else chess.BLACK
        self.reset_game()
    
    def is_human_turn(self):
        """Check if it's the human's turn"""
        if self.game_mode == "Human vs Human":
            return True
        elif self.game_mode == "AI vs AI":
            return False
        else:  # Human vs AI
            return self.board.turn == self.human_color
    
    def check_ai_turn(self):
        """Check if AI should make a move and schedule it"""
        if (not self.game_paused and not self.is_human_turn() and 
            not self.board.is_game_over() and not self.ai_thinking):
            self.ai_thinking = True
            self.status_label.setText("AI is thinking...")
            self.ai_timer.start(500)  # Small delay to show "thinking" message
    
    def make_ai_move(self):
        """Make an AI move"""
        if self.board.is_game_over():
            self.ai_thinking = False
            return
        
        best_move = self.ai.get_best_move(self.board)
        if best_move:
            move_notation = self.board.san(best_move)  # Get notation before pushing
            self.board.push(best_move)
            self.status_label.setText(f"AI played: {move_notation}")
            self.selected_square = None
            self.legal_moves = []
            self.update_board()
            self.update_info()
        
        self.ai_thinking = False
        
        # Check if AI should make another move (for AI vs AI)
        self.check_ai_turn()

    def on_square_clicked(self, square):
        """Handle square clicks for piece selection and movement"""
        # Don't allow human interaction during AI thinking, when it's not human's turn, or when paused
        if self.ai_thinking or not self.is_human_turn() or self.game_paused:
            return
            
        piece = self.board.piece_at(square)
        
        # If we have a piece selected and this click is a legal move
        if self.selected_square is not None:
            move = chess.Move(self.selected_square, square)
            
            # Check for promotion (simplified - always promote to queen)
            if (self.board.piece_at(self.selected_square) and 
                self.board.piece_at(self.selected_square).piece_type == chess.PAWN):
                if (chess.square_rank(square) == 7 and self.board.turn == chess.WHITE) or \
                   (chess.square_rank(square) == 0 and self.board.turn == chess.BLACK):
                    move = chess.Move(self.selected_square, square, promotion=chess.QUEEN)
            
            # Try to make the move
            if move in self.board.legal_moves:
                move_notation = self.board.san(move)  # Get notation before pushing
                self.board.push(move)
                self.selected_square = None
                self.legal_moves = []
                self.status_label.setText(f"Played: {move_notation}")
                self.update_board()
                self.update_info()
                
                # Check if AI should move next
                self.check_ai_turn()
                return
            else:
                # Invalid move, but check if clicking on own piece
                if piece and piece.color == self.board.turn:
                    self.select_piece(square, piece)
                    self.update_board()
                    return
                else:
                    # Deselect
                    self.selected_square = None
                    self.legal_moves = []
                    self.status_label.setText("Invalid move. Click a piece to see its moves.")
        
        # If clicking on a piece of the current player's color
        elif piece and piece.color == self.board.turn:
            self.select_piece(square, piece)
        
        # If clicking on empty square or opponent piece with no selection
        else:
            self.selected_square = None
            self.legal_moves = []
            self.status_label.setText("Click a piece to see its possible moves")
        
        self.update_board()
    
    def select_piece(self, square, piece):
        """Select a piece and highlight its legal moves"""
        self.selected_square = square
        self.legal_moves = [move for move in self.board.legal_moves if move.from_square == square]
        
        piece_name = chess.piece_name(piece.piece_type).title()
        move_count = len(self.legal_moves)
        self.status_label.setText(f"Selected {piece_name} on {chess.square_name(square)} - {move_count} possible moves")
    
    def get_board_svg(self):
        """Generate SVG for current board position with highlights"""
        # Prepare highlighting
        fill_dict = {}
        arrows = []
        
        # Highlight selected square
        if self.selected_square is not None:
            fill_dict[self.selected_square] = "#ffff0066"  # Yellow for selected piece
        
        # Highlight legal move destinations
        for move in self.legal_moves:
            fill_dict[move.to_square] = "#00ff0066"  # Green for possible moves
        
        return chess.svg.board(
            self.board,
            fill=fill_dict,
            size=350,
        )
    
    def update_board(self):
        """Update the chess board display"""
        try:
            svg_data = self.get_board_svg()
            
            # Load SVG directly into widget
            self.svg_widget.load(svg_data.encode('utf-8'))
            
        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")
    
    def update_info(self):
        """Update game information display"""
        turn = "White" if self.board.turn == chess.WHITE else "Black"
        move_num = self.board.fullmove_number
        
        # Check game status
        if self.board.is_checkmate():
            winner = "Black" if self.board.turn == chess.WHITE else "White"
            self.info_label.setText(f"Checkmate! {winner} wins!")
        elif self.board.is_stalemate():
            self.info_label.setText("Stalemate! Game is a draw.")
        elif self.board.is_check():
            self.info_label.setText(f"Move {move_num} - {turn}'s turn (IN CHECK)")
        else:
            self.info_label.setText(f"Move {move_num} - {turn}'s turn")
    
    def reset_game(self):
        """Reset the board to starting position"""
        self.board = chess.Board()
        self.selected_square = None
        self.legal_moves = []
        self.ai_thinking = False
        self.ai_timer.stop()
        
        # Reset pause state
        self.game_paused = False
        self.pause_btn.setText("Pause")
        
        if self.game_mode == "Human vs Human":
            self.status_label.setText("Game reset. Click a piece to see its moves.")
        elif self.game_mode == "AI vs AI":
            self.status_label.setText("Game reset. AI vs AI mode - Watch them play!")
        else:  # Human vs AI
            color_str = "White" if self.human_color == chess.WHITE else "Black"
            self.status_label.setText(f"Game reset. You are playing as {color_str}")
        
        self.update_board()
        self.update_info()
        
        # Start AI if needed
        self.check_ai_turn()
    
    def export_svg(self):
        """Export current board as SVG file"""
        try:
            svg_data = self.get_board_svg()
            with open("exported_board.svg", "w") as f:
                f.write(svg_data)
            self.status_label.setText("SVG exported as 'exported_board.svg'")
        except Exception as e:
            self.status_label.setText(f"Export error: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ChessGUI()
    window.show()
    sys.exit(app.exec()) 