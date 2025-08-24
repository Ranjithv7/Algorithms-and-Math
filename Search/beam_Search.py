import numpy as np
import time
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import heapq
from collections import defaultdict

class PieceType(Enum):
    PAWN = 'P'
    ROOK = 'R'
    KNIGHT = 'N'
    BISHOP = 'B'
    QUEEN = 'Q'
    KING = 'K'

class Color(Enum):
    WHITE = 'white'
    BLACK = 'black'

@dataclass(frozen=True)
class Position:
    row: int
    col: int
    
    def is_valid(self) -> bool:
        return 0 <= self.row < 8 and 0 <= self.col < 8
    
    def __add__(self, other):
        return Position(self.row + other.row, self.col + other.col)

@dataclass(frozen=True)
class Piece:
    piece_type: PieceType
    color: Color
    
    def __str__(self):
        symbol = self.piece_type.value
        return symbol if self.color == Color.WHITE else symbol.lower()

@dataclass(frozen=True)
class Move:
    """
    Represents a chess move.
    
    WHY IMMUTABLE?
    - Allows hashing for transposition tables
    - Prevents accidental modification during search
    - Enables easy undo/redo operations
    """
    from_pos: Position
    to_pos: Position
    piece: Piece
    captured_piece: Optional[Piece] = None
    
    def __str__(self):
        return f"{self.piece}{chr(97+self.from_pos.col)}{self.from_pos.row+1}-{chr(97+self.to_pos.col)}{self.to_pos.row+1}"

class ChessBoard:
    """
    Simplified chess board representation for AI demonstration.
    
    DESIGN GOALS:
    - Fast move generation (critical for beam search performance)
    - Efficient position evaluation
    - Easy board state copying for search tree
    """
    
    def __init__(self):
        self.board = [[None for _ in range(8)] for _ in range(8)]
        self.to_move = Color.WHITE
        self.move_history = []
        self._setup_initial_position()
    
    def _setup_initial_position(self):
        """Set up standard chess starting position"""
        
        # Pawns
        for col in range(8):
            self.board[1][col] = Piece(PieceType.PAWN, Color.WHITE)
            self.board[6][col] = Piece(PieceType.PAWN, Color.BLACK)
        
        # Major pieces
        piece_order = [PieceType.ROOK, PieceType.KNIGHT, PieceType.BISHOP, PieceType.QUEEN,
                      PieceType.KING, PieceType.BISHOP, PieceType.KNIGHT, PieceType.ROOK]
        
        for col, piece_type in enumerate(piece_order):
            self.board[0][col] = Piece(piece_type, Color.WHITE)
            self.board[7][col] = Piece(piece_type, Color.BLACK)
    
    def get_piece(self, pos: Position) -> Optional[Piece]:
        """Get piece at position"""
        if not pos.is_valid():
            return None
        return self.board[pos.row][pos.col]
    
    def set_piece(self, pos: Position, piece: Optional[Piece]):
        """Set piece at position"""
        if pos.is_valid():
            self.board[pos.row][pos.col] = piece
    
    def copy(self) -> 'ChessBoard':
        """Create deep copy of board state"""
        new_board = ChessBoard()
        new_board.board = [[self.board[r][c] for c in range(8)] for r in range(8)]
        new_board.to_move = self.to_move
        new_board.move_history = self.move_history.copy()
        return new_board
    
    def make_move(self, move: Move) -> 'ChessBoard':
        """
        Make a move and return new board state.
        
        WHY RETURN NEW BOARD?
        - Immutable operations for search tree
        - Easy to undo moves by using previous board state
        - Prevents bugs from modifying shared state
        """
        new_board = self.copy()
        
        # Remove piece from source
        new_board.set_piece(move.from_pos, None)
        
        # Place piece at destination
        new_board.set_piece(move.to_pos, move.piece)
        
        # Switch turn
        new_board.to_move = Color.BLACK if self.to_move == Color.WHITE else Color.WHITE
        
        # Add to move history
        new_board.move_history.append(move)
        
        return new_board
    
    def get_legal_moves(self, color: Color) -> List[Move]:
        """
        Generate all legal moves for given color.
        
        SIMPLIFIED FOR DEMO:
        - Basic piece movement rules
        - No castling, en passant, or check detection
        - Focus on beam search algorithm, not chess complexity
        """
        moves = []
        
        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                if piece and piece.color == color:
                    pos = Position(row, col)
                    moves.extend(self._get_piece_moves(pos, piece))
        
        return moves
    
    def _get_piece_moves(self, pos: Position, piece: Piece) -> List[Move]:
        """Get legal moves for piece at position"""
        if piece.piece_type == PieceType.PAWN:
            return self._get_pawn_moves(pos, piece)
        elif piece.piece_type == PieceType.ROOK:
            return self._get_rook_moves(pos, piece)
        elif piece.piece_type == PieceType.KNIGHT:
            return self._get_knight_moves(pos, piece)
        elif piece.piece_type == PieceType.BISHOP:
            return self._get_bishop_moves(pos, piece)
        elif piece.piece_type == PieceType.QUEEN:
            return self._get_queen_moves(pos, piece)
        elif piece.piece_type == PieceType.KING:
            return self._get_king_moves(pos, piece)
        return []
    
    def _get_pawn_moves(self, pos: Position, piece: Piece) -> List[Move]:
        """Generate pawn moves"""
        moves = []
        direction = 1 if piece.color == Color.WHITE else -1
        
        # Forward move
        new_pos = Position(pos.row + direction, pos.col)
        if new_pos.is_valid() and self.get_piece(new_pos) is None:
            moves.append(Move(pos, new_pos, piece))
            
            # Double move from starting position
            if ((piece.color == Color.WHITE and pos.row == 1) or 
                (piece.color == Color.BLACK and pos.row == 6)):
                double_pos = Position(pos.row + 2*direction, pos.col)
                if double_pos.is_valid() and self.get_piece(double_pos) is None:
                    moves.append(Move(pos, double_pos, piece))
        
        # Capture moves
        for col_offset in [-1, 1]:
            capture_pos = Position(pos.row + direction, pos.col + col_offset)
            if capture_pos.is_valid():
                target = self.get_piece(capture_pos)
                if target and target.color != piece.color:
                    moves.append(Move(pos, capture_pos, piece, target))
        
        return moves
    
    def _get_sliding_moves(self, pos: Position, piece: Piece, directions: List[Position]) -> List[Move]:
        """Generate moves for sliding pieces (rook, bishop, queen)"""
        moves = []
        
        for direction in directions:
            current = pos
            while True:
                current = current + direction
                if not current.is_valid():
                    break
                
                target = self.get_piece(current)
                if target is None:
                    # Empty square
                    moves.append(Move(pos, current, piece))
                elif target.color != piece.color:
                    # Capture
                    moves.append(Move(pos, current, piece, target))
                    break
                else:
                    # Own piece blocks
                    break
        
        return moves
    
    def _get_rook_moves(self, pos: Position, piece: Piece) -> List[Move]:
        directions = [Position(0,1), Position(0,-1), Position(1,0), Position(-1,0)]
        return self._get_sliding_moves(pos, piece, directions)
    
    def _get_bishop_moves(self, pos: Position, piece: Piece) -> List[Move]:
        directions = [Position(1,1), Position(1,-1), Position(-1,1), Position(-1,-1)]
        return self._get_sliding_moves(pos, piece, directions)
    
    def _get_queen_moves(self, pos: Position, piece: Piece) -> List[Move]:
        rook_dirs = [Position(0,1), Position(0,-1), Position(1,0), Position(-1,0)]
        bishop_dirs = [Position(1,1), Position(1,-1), Position(-1,1), Position(-1,-1)]
        return self._get_sliding_moves(pos, piece, rook_dirs + bishop_dirs)
    
    def _get_knight_moves(self, pos: Position, piece: Piece) -> List[Move]:
        moves = []
        knight_moves = [
            Position(2,1), Position(2,-1), Position(-2,1), Position(-2,-1),
            Position(1,2), Position(1,-2), Position(-1,2), Position(-1,-2)
        ]
        
        for move_offset in knight_moves:
            new_pos = pos + move_offset
            if new_pos.is_valid():
                target = self.get_piece(new_pos)
                if target is None or target.color != piece.color:
                    moves.append(Move(pos, new_pos, piece, target))
        
        return moves
    
    def _get_king_moves(self, pos: Position, piece: Piece) -> List[Move]:
        moves = []
        king_moves = [
            Position(1,0), Position(-1,0), Position(0,1), Position(0,-1),
            Position(1,1), Position(1,-1), Position(-1,1), Position(-1,-1)
        ]
        
        for move_offset in king_moves:
            new_pos = pos + move_offset
            if new_pos.is_valid():
                target = self.get_piece(new_pos)
                if target is None or target.color != piece.color:
                    moves.append(Move(pos, new_pos, piece, target))
        
        return moves

class ChessEvaluator:
    """
    Chess position evaluation function.
    
    CRITICAL FOR BEAM SEARCH:
    - Must give good relative rankings of positions
    - Faster evaluation = better beam search performance
    - Quality of evaluation directly impacts move quality
    """
    
    # Piece values (standard chess values)
    PIECE_VALUES = {
        PieceType.PAWN: 100,
        PieceType.KNIGHT: 320,
        PieceType.BISHOP: 330,
        PieceType.ROOK: 500,
        PieceType.QUEEN: 900,
        PieceType.KING: 20000
    }
    
    # Position bonuses for pieces (simplified)
    CENTER_BONUS = 10  # Bonus for central squares
    DEVELOPMENT_BONUS = 15  # Bonus for developed pieces
    
    def evaluate_position(self, board: ChessBoard, for_color: Color) -> float:
        """
        Evaluate chess position from perspective of given color.
        
        EVALUATION COMPONENTS:
        1. Material balance (piece values)
        2. Positional factors (center control, development)
        3. King safety (simplified)
        
        HIGHER SCORE = BETTER FOR for_color
        """
        score = 0.0
        
        # Material evaluation
        score += self._evaluate_material(board, for_color)
        
        # Positional evaluation
        score += self._evaluate_position_factors(board, for_color)
        
        return score
    
    def _evaluate_material(self, board: ChessBoard, for_color: Color) -> float:
        """Count material advantage"""
        score = 0.0
        
        for row in range(8):
            for col in range(8):
                piece = board.board[row][col]
                if piece:
                    value = self.PIECE_VALUES[piece.piece_type]
                    if piece.color == for_color:
                        score += value
                    else:
                        score -= value
        
        return score
    
    def _evaluate_position_factors(self, board: ChessBoard, for_color: Color) -> float:
        """Evaluate positional factors"""
        score = 0.0
        
        for row in range(8):
            for col in range(8):
                piece = board.board[row][col]
                if piece:
                    multiplier = 1 if piece.color == for_color else -1
                    
                    # Center control bonus
                    if 2 <= row <= 5 and 2 <= col <= 5:
                        score += self.CENTER_BONUS * multiplier
                    
                    # Development bonus (knights and bishops not on back rank)
                    if piece.piece_type in [PieceType.KNIGHT, PieceType.BISHOP]:
                        back_rank = 0 if piece.color == Color.WHITE else 7
                        if row != back_rank:
                            score += self.DEVELOPMENT_BONUS * multiplier
        
        return score

@dataclass
class SearchNode:
    """
    Node in the game tree search.
    
    WHY THESE FIELDS:
    - board: Current game state
    - depth: How deep in search tree
    - evaluation: Position evaluation score
    - best_move: Move that led to this position
    - parent: For move sequence reconstruction
    """
    board: ChessBoard
    depth: int
    evaluation: float
    best_move: Optional[Move] = None
    parent: Optional['SearchNode'] = None
    
    def __lt__(self, other):
        # Higher evaluation is better (max heap behavior)
        return self.evaluation > other.evaluation

class BeamSearchChessAI:
    """
    Chess AI using Beam Search for move selection.
    
    ALGORITHM STRATEGY:
    - At each depth level, keep only k best positions
    - Expand each kept position by trying all legal moves
    - Evaluate resulting positions and keep best k again
    - Continue until target depth or time limit
    
    BEAM SEARCH BENEFITS:
    - Handles massive branching factor (~35 moves per position)
    - Time-constrained decision making
    - Focuses on most promising lines of play
    """
    
    def __init__(self, beam_width: int = 5, max_depth: int = 4):
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.evaluator = ChessEvaluator()
        
        # Search statistics
        self.nodes_evaluated = 0
        self.moves_considered = 0
        self.pruned_moves = 0
        self.search_time = 0.0
    
    def get_best_move(self, board: ChessBoard, time_limit: float = 5.0) -> Tuple[Move, Dict]:
        """
        Find best move using beam search with time limit.
        
        TIME-CONSTRAINED SEARCH:
        - Stop search when time limit approached
        - Return best move found so far
        - Demonstrates real-time decision making
        """
        start_time = time.time()
        
        # Reset statistics
        self.nodes_evaluated = 0
        self.moves_considered = 0
        self.pruned_moves = 0
        
        # Initialize beam with current position
        current_beam = [SearchNode(
            board=board,
            depth=0,
            evaluation=self.evaluator.evaluate_position(board, board.to_move)
        )]
        
        best_move = None
        best_evaluation = float('-inf')
        
        # Iterative deepening with beam search
        for depth in range(1, self.max_depth + 1):
            # Check time limit
            if time.time() - start_time > time_limit * 0.9:  # Use 90% of time limit
                print(f"⏰ Time limit reached at depth {depth}")
                break
            
            print(f"🔍 Searching depth {depth} with beam width {self.beam_width}")
            
            # Beam search at current depth
            new_beam = []
            
            for node in current_beam:
                # Generate all legal moves
                legal_moves = node.board.get_legal_moves(node.board.to_move)
                self.moves_considered += len(legal_moves)
                
                # Evaluate each move
                move_evaluations = []
                for move in legal_moves:
                    new_board = node.board.make_move(move)
                    evaluation = self.evaluator.evaluate_position(new_board, board.to_move)
                    self.nodes_evaluated += 1
                    
                    new_node = SearchNode(
                        board=new_board,
                        depth=depth,
                        evaluation=evaluation,
                        best_move=move,
                        parent=node
                    )
                    move_evaluations.append(new_node)
                
                # Keep only top beam_width moves from this position
                move_evaluations.sort(key=lambda n: n.evaluation, reverse=True)
                new_beam.extend(move_evaluations[:self.beam_width])
                self.pruned_moves += max(0, len(move_evaluations) - self.beam_width)
            
            # Keep only top beam_width positions overall
            new_beam.sort(key=lambda n: n.evaluation, reverse=True)
            current_beam = new_beam[:self.beam_width]
            
            # Update best move if we found something better
            if current_beam and current_beam[0].evaluation > best_evaluation:
                best_evaluation = current_beam[0].evaluation
                best_move = self._reconstruct_best_move(current_beam[0])
            
            print(f"   Best evaluation: {best_evaluation:.1f}")
            print(f"   Beam size: {len(current_beam)}")
        
        self.search_time = time.time() - start_time
        
        # Compile search statistics
        stats = {
            'search_time': self.search_time,
            'nodes_evaluated': self.nodes_evaluated,
            'moves_considered': self.moves_considered,
            'pruned_moves': self.pruned_moves,
            'final_evaluation': best_evaluation,
            'search_depth': depth,
            'beam_width': self.beam_width,
            'pruning_ratio': self.pruned_moves / max(1, self.moves_considered)
        }
        
        return best_move, stats
    
    def _reconstruct_best_move(self, leaf_node: SearchNode) -> Move:
        """Reconstruct the first move in the best sequence"""
        current = leaf_node
        while current.parent is not None:
            if current.parent.depth == 0:  # Parent is root
                return current.best_move
            current = current.parent
        return current.best_move

class MinimaxChessAI:
    """
    Traditional Minimax AI for comparison.
    
    ALGORITHM STRATEGY:
    - Exhaustive search to fixed depth
    - Alternating maximization/minimization
    - Guaranteed optimal play within search depth
    
    COMPARISON PURPOSE:
    - Show optimal vs approximate trade-off
    - Demonstrate computational cost difference
    - Validate beam search move quality
    """
    
    def __init__(self, max_depth: int = 3):
        self.max_depth = max_depth
        self.evaluator = ChessEvaluator()
        
        # Search statistics
        self.nodes_evaluated = 0
        self.search_time = 0.0
    
    def get_best_move(self, board: ChessBoard, time_limit: float = 5.0) -> Tuple[Move, Dict]:
        """Find best move using minimax algorithm"""
        start_time = time.time()
        self.nodes_evaluated = 0
        
        legal_moves = board.get_legal_moves(board.to_move)
        if not legal_moves:
            return None, {'error': 'No legal moves'}
        
        best_move = legal_moves[0]
        best_value = float('-inf')
        
        for move in legal_moves:
            # Check time limit
            if time.time() - start_time > time_limit:
                break
                
            new_board = board.make_move(move)
            value = self._minimax(new_board, self.max_depth - 1, False, board.to_move)
            
            if value > best_value:
                best_value = value
                best_move = move
        
        self.search_time = time.time() - start_time
        
        stats = {
            'search_time': self.search_time,
            'nodes_evaluated': self.nodes_evaluated,
            'final_evaluation': best_value,
            'search_depth': self.max_depth
        }
        
        return best_move, stats
    
    def _minimax(self, board: ChessBoard, depth: int, maximizing: bool, for_color: Color) -> float:
        """Minimax algorithm implementation"""
        self.nodes_evaluated += 1
        
        if depth == 0:
            return self.evaluator.evaluate_position(board, for_color)
        
        legal_moves = board.get_legal_moves(board.to_move)
        if not legal_moves:
            # Simplified: treat no moves as neutral
            return 0.0
        
        if maximizing:
            max_eval = float('-inf')
            for move in legal_moves:
                new_board = board.make_move(move)
                eval_score = self._minimax(new_board, depth - 1, False, for_color)
                max_eval = max(max_eval, eval_score)
            return max_eval
        else:
            min_eval = float('inf')
            for move in legal_moves:
                new_board = board.make_move(move)
                eval_score = self._minimax(new_board, depth - 1, True, for_color)
                min_eval = min(min_eval, eval_score)
            return min_eval

def create_interesting_position() -> ChessBoard:
    """
    Create an interesting chess position for testing.
    
    POSITION FEATURES:
    - Material roughly equal
    - Multiple good move options
    - Tactical opportunities
    - Tests evaluation function quality
    """
    board = ChessBoard()
    
    # Clear the board
    for row in range(8):
        for col in range(8):
            board.board[row][col] = None
    
    # Set up a mid-game position
    # White pieces
    board.set_piece(Position(0, 4), Piece(PieceType.KING, Color.WHITE))
    board.set_piece(Position(0, 0), Piece(PieceType.ROOK, Color.WHITE))
    board.set_piece(Position(2, 3), Piece(PieceType.QUEEN, Color.WHITE))
    board.set_piece(Position(1, 1), Piece(PieceType.PAWN, Color.WHITE))
    board.set_piece(Position(1, 2), Piece(PieceType.PAWN, Color.WHITE))
    board.set_piece(Position(1, 5), Piece(PieceType.PAWN, Color.WHITE))
    board.set_piece(Position(3, 4), Piece(PieceType.KNIGHT, Color.WHITE))
    
    # Black pieces  
    board.set_piece(Position(7, 4), Piece(PieceType.KING, Color.BLACK))
    board.set_piece(Position(7, 7), Piece(PieceType.ROOK, Color.BLACK))
    board.set_piece(Position(5, 2), Piece(PieceType.QUEEN, Color.BLACK))
    board.set_piece(Position(6, 1), Piece(PieceType.PAWN, Color.BLACK))
    board.set_piece(Position(6, 3), Piece(PieceType.PAWN, Color.BLACK))
    board.set_piece(Position(6, 6), Piece(PieceType.PAWN, Color.BLACK))
    board.set_piece(Position(4, 5), Piece(PieceType.KNIGHT, Color.BLACK))
    
    board.to_move = Color.WHITE
    return board

def visualize_board(board: ChessBoard):
    """Simple ASCII visualization of chess board"""
    print("   a b c d e f g h")
    print("  ┌─────────────────┐")
    
    for row in range(7, -1, -1):  # Display from rank 8 to 1
        print(f"{row+1} │", end="")
        for col in range(8):
            piece = board.board[row][col]
            if piece:
                print(f" {piece}", end="")
            else:
                print(" .", end="")
        print(f" │ {row+1}")
    
    print("  └─────────────────┘")
    print("   a b c d e f g h")
    print(f"\nTo move: {board.to_move.value}")

def run_chess_ai_comparison():
    """
    Complete comparison between Beam Search and Minimax chess AI.
    
    COMPARISON METRICS:
    - Move quality (evaluation scores)
    - Search speed (nodes per second)
    - Time efficiency
    - Pruning effectiveness
    """
    print("♟️  CHESS AI: BEAM SEARCH VS MINIMAX COMPARISON")
    print("=" * 60)
    
    # Create test position
    board = create_interesting_position()
    
    print("🏁 Starting Position:")
    visualize_board(board)
    
    print(f"\n📊 Legal moves available: {len(board.get_legal_moves(board.to_move))}")
    
    # Test different beam widths
    beam_widths = [1, 3, 5, 10]
    time_limit = 3.0  # 3 seconds per move
    
    results = {}
    
    print(f"\n🔍 BEAM SEARCH ANALYSIS (time limit: {time_limit}s)")
    print("=" * 50)
    
    for beam_width in beam_widths:
        print(f"\n⚡ Beam Width: {beam_width}")
        
        ai = BeamSearchChessAI(beam_width=beam_width, max_depth=6)
        move, stats = ai.get_best_move(board, time_limit)
        
        if move:
            print(f"✅ Best move: {move}")
            print(f"   Evaluation: {stats['final_evaluation']:.1f}")
            print(f"   Search time: {stats['search_time']:.3f}s")
            print(f"   Nodes evaluated: {stats['nodes_evaluated']:,}")
            print(f"   Moves considered: {stats['moves_considered']:,}")
            print(f"   Pruned moves: {stats['pruned_moves']:,}")
            print(f"   Pruning ratio: {stats['pruning_ratio']:.1%}")
            print(f"   Nodes/second: {stats['nodes_evaluated']/max(stats['search_time'], 0.001):,.0f}")
            
            results[f'beam_{beam_width}'] = {
                'move': move,
                'stats': stats,
                'ai_type': 'beam_search'
            }
        else:
            print("❌ No move found")
    
    # Test Minimax for comparison
    print(f"\n🔍 MINIMAX ANALYSIS (optimal within depth)")
    print("=" * 45)
    
    minimax_ai = MinimaxChessAI(max_depth=4)
    move, stats = minimax_ai.get_best_move(board, time_limit)
    
    if move:
        print(f"✅ Best move: {move}")
        print(f"   Evaluation: {stats['final_evaluation']:.1f}")
        print(f"   Search time: {stats['search_time']:.3f}s")
        print(f"   Nodes evaluated: {stats['nodes_evaluated']:,}")
        print(f"   Search depth: {stats['search_depth']}")
        print(f"   Nodes/second: {stats['nodes_evaluated']/max(stats['search_time'], 0.001):,.0f}")
        
        results['minimax'] = {
            'move': move,
            'stats': stats,
            'ai_type': 'minimax'
        }
    
    # Analysis and comparison
    print(f"\n📈 PERFORMANCE COMPARISON")
    print("=" * 40)
    
    if results:
        print(f"{'Algorithm':<15} {'Move':<12} {'Eval':<8} {'Time':<8} {'Nodes':<10} {'N/sec':<8}")
        print("-" * 70)
        
        for name, result in results.items():
            stats = result['stats']
            move_str = str(result['move'])[:11]
            eval_str = f"{stats.get('final_evaluation', 0):.1f}"
            time_str = f"{stats.get('search_time', 0):.3f}s"
            nodes_str = f"{stats.get('nodes_evaluated', 0):,}"
            nps_str = f"{stats.get('nodes_evaluated', 0)/max(stats.get('search_time', 0.001), 0.001):,.0f}"
            
            print(f"{name:<15} {move_str:<12} {eval_str:<8} {time_str:<8} {nodes_str:<10} {nps_str:<8}")
    
    # Key insights
    print(f"\n💡 KEY INSIGHTS:")
    print(f"   🎯 BEAM SEARCH BENEFITS:")
    print(f"      - Handles time pressure effectively")
    print(f"      - Scalable performance (adjust beam width)")
    print(f"      - High pruning ratio reduces search space")
    print(f"      - Good move quality with fraction of computation")
    print(f"   ")
    print(f"   ⚖️  MINIMAX BENEFITS:")
    print(f"      - Guaranteed optimal within search depth")
    print(f"      - No risk of missing good moves due to evaluation errors")
    print(f"      - More predictable search behavior")
    print(f"   ")
    print(f"   🔄 TRADE-OFF ANALYSIS:")
    if 'beam_5' in results and 'minimax' in results:
        beam_stats = results['beam_5']['stats']
        minimax_stats = results['minimax']['stats']
        
        time_ratio = minimax_stats['search_time'] / beam_stats['search_time']
        node_ratio = minimax_stats['nodes_evaluated'] / beam_stats['nodes_evaluated']
        
        print(f"      - Beam Search (k=5) vs Minimax:")
        print(f"        * Time efficiency: {time_ratio:.1f}× faster")
        print(f"        * Node efficiency: {node_ratio:.1f}× fewer nodes")
        print(f"        * Evaluation difference: {abs(beam_stats['final_evaluation'] - minimax_stats['final_evaluation']):.1f} points")
    
    return results

def demonstrate_beam_width_effects():
    """Show how beam width affects move quality and computation time"""
    print(f"\n🔬 BEAM WIDTH SENSITIVITY ANALYSIS")
    print("=" * 45)
    
    board = create_interesting_position()
    beam_widths = [1, 2, 3, 5, 8, 12, 20]
    
    print(f"{'Beam Width':<12} {'Move Quality':<12} {'Time (s)':<10} {'Efficiency':<12}")
    print("-" * 50)
    
    baseline_time = None
    baseline_eval = None
    
    for beam_width in beam_widths:
        ai = BeamSearchChessAI(beam_width=beam_width, max_depth=4)
        move, stats = ai.get_best_move(board, time_limit=2.0)
        
        if move and stats:
            eval_score = stats['final_evaluation']
            search_time = stats['search_time']
            
            if baseline_time is None:
                baseline_time = search_time
                baseline_eval = eval_score
            
            time_ratio = search_time / baseline_time
            eval_improvement = eval_score - baseline_eval
            efficiency = eval_improvement / max(time_ratio, 0.1)
            
            print(f"{beam_width:<12} {eval_score:<12.1f} {search_time:<10.3f} {efficiency:<12.1f}")
    
    print(f"\n💡 Observations:")
    print(f"   - Diminishing returns as beam width increases")
    print(f"   - Sweet spot typically around k=3-8 for chess")
    print(f"   - Efficiency (quality/time) peaks at moderate beam widths")

if __name__ == "__main__":
    # Run the comprehensive comparison
    print("🚀 Starting Chess AI Analysis...")
    
    results = run_chess_ai_comparison()
    
    print(f"\n" + "="*60)
    demonstrate_beam_width_effects()
    
    print(f"\n🎯 BEAM SEARCH CHESS AI MASTERY ACHIEVED!")
    print(f"   - Implemented time-constrained chess AI")
    print(f"   - Demonstrated approximation vs optimality trade-off") 
    print(f"   - Analyzed beam width effects on performance")
    print(f"   - Compared with traditional minimax approach")
    print(f"   - Applied beam search to real-time decision making")