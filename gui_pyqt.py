import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QLabel
from PyQt6.QtSvgWidgets import QSvgWidget
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QMouseEvent
import chess
import chess.svg

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
        self.setGeometry(100, 100, 600, 700)
        
        # Game state
        self.board = chess.Board()
        self.selected_square = None
        self.legal_moves = []
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Title
        title = QLabel("Chess Board Visualization")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin: 10px;")
        layout.addWidget(title)
        
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
    
    def on_square_clicked(self, square):
        """Handle square clicks for piece selection and movement"""
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
                self.board.push(move)
                self.selected_square = None
                self.legal_moves = []
                self.status_label.setText(f"Moved to {chess.square_name(square)}")
                self.update_board()
                self.update_info()
                return
            else:
                # Invalid move, but check if clicking on own piece
                if piece and piece.color == self.board.turn:
                    self.select_piece(square, piece)
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
        self.status_label.setText("Game reset. Click a piece to see its moves.")
        self.update_board()
        self.update_info()
    
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