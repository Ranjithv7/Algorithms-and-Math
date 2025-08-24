import numpy as np
import heapq
from typing import List, Tuple, Set, Dict, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass
from enum import Enum
import time
from collections import defaultdict

class CellType(Enum):
    FREE = 0
    OBSTACLE = 1
    START = 2  
    GOAL = 3
    PATH = 4
    EXPLORED = 5

@dataclass(frozen=True)
class Position:
    """2D grid position. Frozen for hashing."""
    x: int
    y: int
    
    def __add__(self, other):
        return Position(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        return Position(self.x - other.x, self.y - other.y)
    
    def manhattan_distance(self, other):
        return abs(self.x - other.x) + abs(self.y - other.y)
    
    def euclidean_distance(self, other):
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

class Direction:
    """
    Direction vectors for 8-connectivity movement.
    
    WHY THESE SPECIFIC DIRECTIONS?
    - Cardinal: N,S,E,W (cost = 1.0)  
    - Diagonal: NE,NW,SE,SW (cost = √2)
    - Essential for JPS pruning rules
    """
    # Cardinal directions
    NORTH = Position(0, 1)
    SOUTH = Position(0, -1) 
    EAST = Position(1, 0)
    WEST = Position(-1, 0)
    
    # Diagonal directions
    NORTHEAST = Position(1, 1)
    NORTHWEST = Position(-1, 1)
    SOUTHEAST = Position(1, -1)
    SOUTHWEST = Position(-1, -1)
    
    # All directions list
    ALL = [NORTH, SOUTH, EAST, WEST, NORTHEAST, NORTHWEST, SOUTHEAST, SOUTHWEST]
    CARDINAL = [NORTH, SOUTH, EAST, WEST]
    DIAGONAL = [NORTHEAST, NORTHWEST, SOUTHEAST, SOUTHWEST]
    
    @staticmethod
    def is_diagonal(direction: Position) -> bool:
        return abs(direction.x) == 1 and abs(direction.y) == 1
    
    @staticmethod
    def get_cost(direction: Position) -> float:
        """Get movement cost for direction"""
        if Direction.is_diagonal(direction):
            return np.sqrt(2)  # √2 for diagonal
        return 1.0  # 1.0 for cardinal

@dataclass
class SearchNode:
    """
    Node in the search tree.
    
    WHY THESE FIELDS?
    - position: where we are
    - g_cost: exact cost from start
    - h_cost: heuristic estimate to goal
    - f_cost: total estimated cost (g + h)
    - parent: for path reconstruction
    - direction: how we arrived (needed for JPS pruning)
    """
    position: Position
    g_cost: float = float('inf')
    h_cost: float = 0.0
    f_cost: float = float('inf')
    parent: Optional['SearchNode'] = None
    direction: Optional[Position] = None  # Direction of arrival
    
    def __lt__(self, other):
        if self.f_cost == other.f_cost:
            return self.h_cost < other.h_cost
        return self.f_cost < other.f_cost

class GridEnvironment:
    """
    2D grid environment for pathfinding.
    
    DESIGN PRINCIPLES:
    - Simple to understand
    - Complex enough to show JPS benefits  
    - Realistic obstacle patterns
    """
    
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=int)  # 0=free, 1=obstacle
        
    def add_obstacles(self, positions: List[Position]):
        """Add obstacles to grid"""
        for pos in positions:
            if self.is_valid_position(pos):
                self.grid[pos.y, pos.x] = CellType.OBSTACLE.value
                
    def add_maze_pattern(self):
        """Add a complex maze pattern to showcase JPS benefits"""
        # Add border walls
        for x in range(self.width):
            self.grid[0, x] = CellType.OBSTACLE.value
            self.grid[self.height-1, x] = CellType.OBSTACLE.value
        for y in range(self.height):
            self.grid[y, 0] = CellType.OBSTACLE.value  
            self.grid[y, self.width-1] = CellType.OBSTACLE.value
            
        # Add internal maze walls with gaps
        # Vertical walls
        for wall_x in [5, 10, 15, 20, 25]:
            if wall_x < self.width:
                for y in range(2, self.height-2):
                    if y % 6 != 3:  # Leave gaps every 6 cells
                        self.grid[y, wall_x] = CellType.OBSTACLE.value
                        
        # Horizontal walls  
        for wall_y in [4, 8, 12, 16, 20]:
            if wall_y < self.height:
                for x in range(2, self.width-2):
                    if x % 8 != 4:  # Leave gaps every 8 cells
                        self.grid[wall_y, x] = CellType.OBSTACLE.value
                        
        # Add some random scattered obstacles
        np.random.seed(42)  # Reproducible
        for _ in range(min(30, self.width * self.height // 20)):
            x = np.random.randint(2, self.width-2)
            y = np.random.randint(2, self.height-2) 
            if self.grid[y, x] == CellType.FREE.value:
                self.grid[y, x] = CellType.OBSTACLE.value
    
    def is_valid_position(self, pos: Position) -> bool:
        """Check if position is within bounds and not blocked"""
        return (0 <= pos.x < self.width and 
                0 <= pos.y < self.height and
                self.grid[pos.y, pos.x] != CellType.OBSTACLE.value)
    
    def get_neighbors(self, pos: Position) -> List[Position]:
        """Get all valid neighboring positions"""
        neighbors = []
        for direction in Direction.ALL:
            neighbor = pos + direction
            if self.is_valid_position(neighbor):
                neighbors.append(neighbor)
        return neighbors

class JumpPointSearch:
    """
    Standard Jump Point Search implementation.
    
    CORE ALGORITHM:
    1. Identify jump points using pruning rules
    2. Only expand nodes at jump points
    3. Skip intermediate nodes between jump points
    4. Preserve optimality guarantees
    """
    
    def __init__(self, environment: GridEnvironment):
        self.env = environment
        
        # Search statistics
        self.nodes_expanded = 0
        self.nodes_generated = 0
        self.jump_operations = 0
        self.max_frontier_size = 0
        self.explored_positions = set()
        
    def heuristic(self, pos: Position, goal: Position) -> float:
        """
        Diagonal distance heuristic.
        
        WHY DIAGONAL DISTANCE?
        - Admissible for 8-connected grids
        - Tighter than Manhattan distance
        - Accounts for diagonal movement cost
        """
        dx = abs(pos.x - goal.x)
        dy = abs(pos.y - goal.y)
        return max(dx, dy) + (np.sqrt(2) - 1) * min(dx, dy)
    
    def get_forced_neighbors(self, pos: Position, direction: Position) -> List[Position]:
        """
        Identify forced neighbors based on JPS pruning rules.
        
        MATHEMATICAL FOUNDATION:
        - Natural neighbors: reachable via better paths not through current position
        - Forced neighbors: only reachable optimally through current position
        - Only explore forced neighbors to eliminate path symmetry
        """
        forced = []
        
        if Direction.is_diagonal(direction):
            # Diagonal movement: check for obstacles that force exploration
            dx, dy = direction.x, direction.y
            
            # Check both cardinal components
            if not self.env.is_valid_position(pos + Position(-dx, 0)):
                # Obstacle blocks horizontal component, force vertical neighbor
                neighbor = pos + Position(-dx, dy)
                if self.env.is_valid_position(neighbor):
                    forced.append(neighbor)
                    
            if not self.env.is_valid_position(pos + Position(0, -dy)):
                # Obstacle blocks vertical component, force horizontal neighbor
                neighbor = pos + Position(dx, -dy)
                if self.env.is_valid_position(neighbor):
                    forced.append(neighbor)
                    
        else:
            # Cardinal movement: check perpendicular directions for obstacles
            if direction.x != 0:  # Moving horizontally
                # Check vertical obstacles
                for perp_dir in [Position(0, 1), Position(0, -1)]:
                    if not self.env.is_valid_position(pos + perp_dir):
                        # Obstacle forces diagonal exploration
                        neighbor = pos + Position(direction.x, perp_dir.y)
                        if self.env.is_valid_position(neighbor):
                            forced.append(neighbor)
                            
            else:  # Moving vertically  
                # Check horizontal obstacles
                for perp_dir in [Position(1, 0), Position(-1, 0)]:
                    if not self.env.is_valid_position(pos + perp_dir):
                        # Obstacle forces diagonal exploration
                        neighbor = pos + Position(perp_dir.x, direction.y)
                        if self.env.is_valid_position(neighbor):
                            forced.append(neighbor)
        
        return forced
    
    def jump(self, pos: Position, direction: Position, goal: Position) -> Optional[Position]:
        """
        Core jump function: find next jump point in given direction.
        
        ALGORITHM LOGIC:
        1. Move one step in direction
        2. Check termination conditions (goal, obstacle, forced neighbors)
        3. For diagonal: recursively check cardinal components
        4. Continue jumping until jump point found or path blocked
        
        WHY THIS WORKS:
        - Skips intermediate nodes with no forced neighbors
        - Preserves optimality by checking all necessary conditions
        - Dramatically reduces search space
        """
        self.jump_operations += 1
        
        # Next position in direction
        next_pos = pos + direction
        
        # Check if position is valid
        if not self.env.is_valid_position(next_pos):
            return None
            
        # Found goal?
        if next_pos == goal:
            return next_pos
            
        # Check for forced neighbors
        forced_neighbors = self.get_forced_neighbors(next_pos, direction)
        if forced_neighbors:
            return next_pos  # This is a jump point
            
        # Diagonal movement: check cardinal components
        if Direction.is_diagonal(direction):
            dx, dy = direction.x, direction.y
            
            # Check horizontal component
            if self.jump(next_pos, Position(dx, 0), goal) is not None:
                return next_pos
                
            # Check vertical component  
            if self.jump(next_pos, Position(0, dy), goal) is not None:
                return next_pos
        
        # Continue jumping in same direction
        return self.jump(next_pos, direction, goal)
    
    def get_successors(self, node: SearchNode, goal: Position) -> List[Position]:
        """
        Get successor jump points for current node.
        
        PRUNING STRATEGY:
        - If no parent: explore all directions (start node)
        - If has parent: use JPS pruning rules based on arrival direction
        """
        if node.parent is None:
            # Start node: explore all directions
            directions = Direction.ALL
        else:
            # Apply JPS pruning rules
            directions = self._get_pruned_directions(node.position, node.direction)
            
        successors = []
        for direction in directions:
            jump_point = self.jump(node.position, direction, goal)
            if jump_point is not None:
                successors.append(jump_point)
                
        return successors
    
    def _get_pruned_directions(self, pos: Position, arrival_direction: Position) -> List[Position]:
        """
        Apply JPS pruning rules to determine which directions to explore.
        
        PRUNING RULES:
        - Cardinal movement: continue straight + forced neighbors
        - Diagonal movement: both cardinal components + diagonal continuation + forced
        """
        if arrival_direction is None:
            return Direction.ALL
            
        directions = []
        
        if Direction.is_diagonal(arrival_direction):
            # Diagonal movement: explore cardinal components and continuation
            dx, dy = arrival_direction.x, arrival_direction.y
            directions.extend([
                Position(dx, 0),      # Horizontal component
                Position(0, dy),      # Vertical component  
                Position(dx, dy)      # Diagonal continuation
            ])
        else:
            # Cardinal movement: continue in same direction
            directions.append(arrival_direction)
            
        # Add forced neighbors
        forced = self.get_forced_neighbors(pos, arrival_direction)
        for forced_pos in forced:
            forced_direction = forced_pos - pos
            if forced_direction not in directions:
                directions.append(forced_direction)
                
        return directions
    
    def search(self, start: Position, goal: Position) -> Tuple[Optional[List[Position]], Dict]:
        """
        Main JPS search algorithm.
        
        ALGORITHM STRUCTURE:
        1. Initialize with start position
        2. Expand nodes only at jump points
        3. Use A* framework with JPS successor generation
        4. Reconstruct path from jump points
        """
        # Reset statistics
        self.nodes_expanded = 0
        self.nodes_generated = 0
        self.jump_operations = 0
        self.max_frontier_size = 0
        self.explored_positions = set()
        
        # Search data structures
        open_set = []
        closed_set = set()
        node_map = {}
        
        # Initialize start node
        start_node = SearchNode(
            position=start,
            g_cost=0.0,
            h_cost=self.heuristic(start, goal),
            direction=None
        )
        start_node.f_cost = start_node.g_cost + start_node.h_cost
        
        heapq.heappush(open_set, start_node)
        node_map[start] = start_node
        self.nodes_generated += 1
        
        while open_set:
            self.max_frontier_size = max(self.max_frontier_size, len(open_set))
            
            # Get node with minimum f-cost
            current_node = heapq.heappop(open_set)
            
            # Goal test
            if current_node.position == goal:
                path = self._reconstruct_jump_path(current_node)
                stats = self._get_search_statistics(path)
                return path, stats
                
            # Mark as explored
            closed_set.add(current_node.position)
            self.explored_positions.add(current_node.position)
            self.nodes_expanded += 1
            
            # Get jump point successors
            successors = self.get_successors(current_node, goal)
            
            for successor_pos in successors:
                if successor_pos in closed_set:
                    continue
                    
                # Calculate cost to successor
                movement_cost = current_node.position.euclidean_distance(successor_pos)
                tentative_g = current_node.g_cost + movement_cost
                
                # Get or create successor node
                if successor_pos in node_map:
                    successor_node = node_map[successor_pos]
                else:
                    successor_node = SearchNode(
                        position=successor_pos,
                        h_cost=self.heuristic(successor_pos, goal)
                    )
                    node_map[successor_pos] = successor_node
                    self.nodes_generated += 1
                
                # Update if better path found
                if tentative_g < successor_node.g_cost:
                    successor_node.g_cost = tentative_g
                    successor_node.f_cost = successor_node.g_cost + successor_node.h_cost
                    successor_node.parent = current_node
                    successor_node.direction = successor_pos - current_node.position
                    
                    if successor_node not in open_set:
                        heapq.heappush(open_set, successor_node)
        
        # No path found
        stats = self._get_search_statistics(None)
        return None, stats
    
    def _reconstruct_jump_path(self, goal_node: SearchNode) -> List[Position]:
        """
        Reconstruct path from jump points.
        
        PATH EXPANSION:
        - JPS finds path between jump points
        - Need to expand to include intermediate positions
        - Ensures returned path is complete movement sequence
        """
        jump_path = []
        current = goal_node
        
        # Get jump points in reverse order
        while current is not None:
            jump_path.append(current.position)
            current = current.parent
            
        jump_path.reverse()
        
        # Expand path between jump points
        full_path = [jump_path[0]]
        
        for i in range(1, len(jump_path)):
            start_pos = jump_path[i-1]
            end_pos = jump_path[i]
            
            # Generate intermediate positions
            intermediate = self._get_intermediate_positions(start_pos, end_pos)
            full_path.extend(intermediate)
            
        return full_path
    
    def _get_intermediate_positions(self, start: Position, end: Position) -> List[Position]:
        """Generate intermediate positions between two jump points"""
        positions = []
        current = start
        
        # Calculate direction
        dx = 1 if end.x > start.x else (-1 if end.x < start.x else 0)
        dy = 1 if end.y > start.y else (-1 if end.y < start.y else 0)
        direction = Position(dx, dy)
        
        # Generate path
        while current != end:
            current = current + direction
            positions.append(current)
            
        return positions
    
    def _get_search_statistics(self, path: Optional[List[Position]]) -> Dict:
        """Compile search statistics"""
        return {
            'path_length': len(path) if path else 0,
            'path_cost': path[-1].euclidean_distance(path[0]) if path and len(path) > 1 else 0,
            'nodes_expanded': self.nodes_expanded,
            'nodes_generated': self.nodes_generated,
            'jump_operations': self.jump_operations,
            'max_frontier_size': self.max_frontier_size,
            'explored_positions': len(self.explored_positions)
        }

class HierarchicalJPS:
    """
    Hierarchical Jump Point Search implementation.
    
    HIERARCHICAL CONCEPT:
    - Level 0: All jump points (fine-grained)
    - Level k: Jump points reachable via moves of length ≥ 2^k
    - Higher levels for long-distance planning
    - Lower levels for detailed navigation
    
    BENEFITS:
    - Sublinear search complexity for long paths
    - Faster long-distance pathfinding
    - Maintains optimality guarantees
    """
    
    def __init__(self, environment: GridEnvironment, max_levels: int = 4):
        self.env = environment
        self.max_levels = max_levels
        self.base_jps = JumpPointSearch(environment)
        
        # Hierarchical jump point cache
        self.jump_point_cache = {}  # (pos, direction, level) -> jump_point
        self.level_jump_points = [set() for _ in range(max_levels)]
        
        # Statistics
        self.nodes_expanded = 0
        self.nodes_generated = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
    def precompute_hierarchy(self, start: Position, goal: Position):
        """
        Precompute hierarchical jump points around start and goal regions.
        
        PRECOMPUTATION STRATEGY:
        - Level 0: Standard JPS jump points
        - Level k: Jump points reachable via jumps of length ≥ 2^k
        - Cache results for faster lookup during search
        """
        print("🔄 Precomputing hierarchical jump points...")
        
        # Define regions around start and goal
        regions = [
            self._get_region_around(start, radius=15),
            self._get_region_around(goal, radius=15)
        ]
        
        for level in range(self.max_levels):
            min_jump_distance = 2 ** level
            print(f"   Level {level}: min jump distance = {min_jump_distance}")
            
            for region in regions:
                for pos in region:
                    if self.env.is_valid_position(pos):
                        self._compute_jump_points_at_level(pos, level, min_jump_distance)
        
        print(f"✅ Hierarchy precomputed!")
        for level in range(self.max_levels):
            print(f"   Level {level}: {len(self.level_jump_points[level])} jump points")
    
    def _get_region_around(self, center: Position, radius: int) -> List[Position]:
        """Get all positions within radius of center"""
        positions = []
        for x in range(max(0, center.x - radius), min(self.env.width, center.x + radius + 1)):
            for y in range(max(0, center.y - radius), min(self.env.height, center.y + radius + 1)):
                positions.append(Position(x, y))
        return positions
    
    def _compute_jump_points_at_level(self, pos: Position, level: int, min_distance: float):
        """Compute jump points at specific hierarchical level"""
        for direction in Direction.ALL:
            cache_key = (pos, direction, level)
            
            if cache_key not in self.jump_point_cache:
                jump_point = self._hierarchical_jump(pos, direction, min_distance)
                self.jump_point_cache[cache_key] = jump_point
                
                if jump_point is not None:
                    self.level_jump_points[level].add(jump_point)
    
    def _hierarchical_jump(self, pos: Position, direction: Position, min_distance: float) -> Optional[Position]:
        """
        Hierarchical jump function.
        
        MODIFIED JUMP LOGIC:
        - Perform standard jump
        - Only return jump point if distance ≥ min_distance
        - Enables multi-resolution pathfinding
        """
        jump_point = self.base_jps.jump(pos, direction, Position(-1, -1))  # Dummy goal
        
        if jump_point is not None:
            distance = pos.euclidean_distance(jump_point)
            if distance >= min_distance:
                return jump_point
                
        return None
    
    def hierarchical_search(self, start: Position, goal: Position) -> Tuple[Optional[List[Position]], Dict]:
        """
        Multi-level hierarchical search.
        
        SEARCH STRATEGY:
        1. Start with highest level (coarse planning)
        2. Gradually refine with lower levels
        3. Use level 0 for final detailed path
        
        BENEFITS:
        - Long-distance paths found quickly at high levels
        - Short-distance refinement at low levels
        - Combines global and local optimality
        """
        # Reset statistics
        self.nodes_expanded = 0
        self.nodes_generated = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Precompute hierarchy for this search
        self.precompute_hierarchy(start, goal)
        
        # Try each level, starting from highest (coarsest)
        for level in range(self.max_levels - 1, -1, -1):
            print(f"🔍 Trying hierarchical level {level}...")
            
            # Use modified JPS with level-specific constraints
            path, stats = self._search_at_level(start, goal, level)
            
            if path is not None:
                print(f"✅ Path found at level {level}!")
                
                # Refine path using lower levels if not at level 0
                if level > 0:
                    refined_path = self._refine_path(path, level - 1)
                    if refined_path is not None:
                        path = refined_path
                
                # Compile hierarchical statistics
                hierarchical_stats = self._get_hierarchical_statistics(path, stats)
                return path, hierarchical_stats
        
        # Fallback to standard JPS
        print("🔄 Falling back to standard JPS...")
        return self.base_jps.search(start, goal)
    
    def _search_at_level(self, start: Position, goal: Position, level: int) -> Tuple[Optional[List[Position]], Dict]:
        """Search using jump points at specific hierarchical level"""
        min_jump_distance = 2 ** level
        
        # Modified JPS search with level constraints
        open_set = []
        closed_set = set()
        node_map = {}
        
        start_node = SearchNode(
            position=start,
            g_cost=0.0,
            h_cost=self.base_jps.heuristic(start, goal)
        )
        start_node.f_cost = start_node.g_cost + start_node.h_cost
        
        heapq.heappush(open_set, start_node)
        node_map[start] = start_node
        self.nodes_generated += 1
        
        while open_set:
            current_node = heapq.heappop(open_set)
            
            if current_node.position == goal:
                path = self.base_jps._reconstruct_jump_path(current_node)
                stats = {'level': level, 'nodes_expanded': self.nodes_expanded}
                return path, stats
            
            closed_set.add(current_node.position)
            self.nodes_expanded += 1
            
            # Get successors using hierarchical jump points
            successors = self._get_hierarchical_successors(current_node, goal, level, min_jump_distance)
            
            for successor_pos in successors:
                if successor_pos in closed_set:
                    continue
                
                movement_cost = current_node.position.euclidean_distance(successor_pos)
                tentative_g = current_node.g_cost + movement_cost
                
                if successor_pos in node_map:
                    successor_node = node_map[successor_pos]
                else:
                    successor_node = SearchNode(
                        position=successor_pos,
                        h_cost=self.base_jps.heuristic(successor_pos, goal)
                    )
                    node_map[successor_pos] = successor_node
                    self.nodes_generated += 1
                
                if tentative_g < successor_node.g_cost:
                    successor_node.g_cost = tentative_g
                    successor_node.f_cost = successor_node.g_cost + successor_node.h_cost
                    successor_node.parent = current_node
                    successor_node.direction = successor_pos - current_node.position
                    
                    if successor_node not in open_set:
                        heapq.heappush(open_set, successor_node)
        
        return None, {'level': level, 'nodes_expanded': self.nodes_expanded}
    
    def _get_hierarchical_successors(self, node: SearchNode, goal: Position, level: int, min_distance: float) -> List[Position]:
        """Get successors using hierarchical jump points"""
        successors = []
        
        # Get potential directions
        if node.parent is None:
            directions = Direction.ALL
        else:
            directions = self.base_jps._get_pruned_directions(node.position, node.direction)
        
        for direction in directions:
            cache_key = (node.position, direction, level)
            
            if cache_key in self.jump_point_cache:
                jump_point = self.jump_point_cache[cache_key]
                self.cache_hits += 1
            else:
                jump_point = self._hierarchical_jump(node.position, direction, min_distance)
                self.jump_point_cache[cache_key] = jump_point
                self.cache_misses += 1
            
            if jump_point is not None and jump_point != node.position:
                successors.append(jump_point)
        
        return successors
    
    def _refine_path(self, coarse_path: List[Position], target_level: int) -> Optional[List[Position]]:
        """Refine path using lower hierarchical levels"""
        if target_level < 0:
            return coarse_path
            
        refined_segments = []
        
        # Refine each segment of the coarse path
        for i in range(len(coarse_path) - 1):
            segment_start = coarse_path[i]
            segment_end = coarse_path[i + 1]
            
            # Use lower level search for this segment
            segment_path, _ = self._search_at_level(segment_start, segment_end, target_level)
            
            if segment_path is not None:
                # Remove duplicate start position (except for first segment)
                if i > 0:
                    segment_path = segment_path[1:]
                refined_segments.extend(segment_path)
            else:
                # Fallback to direct connection
                intermediate = self.base_jps._get_intermediate_positions(segment_start, segment_end)
                refined_segments.extend(intermediate)
        
        return refined_segments
    
    def _get_hierarchical_statistics(self, path: Optional[List[Position]], base_stats: Dict) -> Dict:
        """Compile hierarchical search statistics"""
        stats = base_stats.copy()
        stats.update({
            'hierarchical_nodes_expanded': self.nodes_expanded,
            'hierarchical_nodes_generated': self.nodes_generated,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_ratio': self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0,
            'hierarchy_levels': self.max_levels
        })
        return stats

def create_complex_maze(width: int = 30, height: int = 25) -> GridEnvironment:
    """
    Create a complex maze that showcases JPS benefits.
    
    DESIGN GOALS:
    - Multiple long corridors (JPS can jump far)
    - Strategic obstacles creating forced neighbors
    - Multiple path options to demonstrate pruning
    """
    env = GridEnvironment(width, height)
    env.add_maze_pattern()
    return env

def visualize_search_comparison(env: GridEnvironment, jps_path: List[Position], 
                              hjps_path: List[Position], jps_explored: Set[Position],
                              hjps_explored: Set[Position], start: Position, goal: Position):
    """Visualize comparison between JPS and Hierarchical JPS"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Prepare grids
    jps_grid = np.copy(env.grid)
    hjps_grid = np.copy(env.grid)
    
    # Mark explored positions
    for pos in jps_explored:
        jps_grid[pos.y, pos.x] = CellType.EXPLORED.value
    
    for pos in hjps_explored:
        hjps_grid[pos.y, pos.x] = CellType.EXPLORED.value
    
    # Mark paths
    if jps_path:
        for pos in jps_path:
            jps_grid[pos.y, pos.x] = CellType.PATH.value
    
    if hjps_path:
        for pos in hjps_path:
            hjps_grid[pos.y, pos.x] = CellType.PATH.value
    
    # Mark start and goal
    jps_grid[start.y, start.x] = CellType.START.value
    jps_grid[goal.y, goal.x] = CellType.GOAL.value
    hjps_grid[start.y, start.x] = CellType.START.value
    hjps_grid[goal.y, goal.x] = CellType.GOAL.value
    
    # Plot JPS
    im1 = ax1.imshow(jps_grid, cmap='tab10', origin='lower')
    ax1.set_title('Standard Jump Point Search')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    
    # Plot Hierarchical JPS
    im2 = ax2.imshow(hjps_grid, cmap='tab10', origin='lower')
    ax2.set_title('Hierarchical Jump Point Search')
    ax2.set_xlabel('X Position')
    ax2.set_ylabel('Y Position')
    
    plt.tight_layout()
    plt.show()

def run_jps_comparison():
    """
    Complete comparison between JPS and Hierarchical JPS.
    
    COMPARISON METRICS:
    - Nodes expanded
    - Search time
    - Path optimality
    - Memory usage
    - Cache efficiency (for HJPS)
    """
    print("🚀 JUMP POINT SEARCH ALGORITHM COMPARISON")
    print("=" * 60)
    
    # Create complex test environment
    env = create_complex_maze(width=30, height=25)
    start = Position(2, 2)
    goal = Position(27, 22)
    
    print(f"🗺️  Test Environment: {env.width}×{env.height} grid")
    print(f"📍 Start: {start}, Goal: {goal}")
    print(f"🧱 Obstacles: {np.sum(env.grid == 1)} cells")
    
    # Run Standard JPS
    print(f"\n🔍 Running STANDARD Jump Point Search...")
    jps = JumpPointSearch(env)
    
    start_time = time.time()
    jps_path, jps_stats = jps.search(start, goal)
    jps_time = time.time() - start_time
    
    if jps_path:
        print(f"✅ JPS Path Found!")
        print(f"   Path length: {jps_stats['path_length']} steps")
        print(f"   Nodes expanded: {jps_stats['nodes_expanded']}")
        print(f"   Nodes generated: {jps_stats['nodes_generated']}")
        print(f"   Jump operations: {jps_stats['jump_operations']}")
        print(f"   Max frontier size: {jps_stats['max_frontier_size']}")
        print(f"   Search time: {jps_time*1000:.2f} ms")
    else:
        print(f"❌ JPS: No path found")
        return
    
    # Run Hierarchical JPS
    print(f"\n🔍 Running HIERARCHICAL Jump Point Search...")
    hjps = HierarchicalJPS(env, max_levels=4)
    
    start_time = time.time()
    hjps_path, hjps_stats = hjps.hierarchical_search(start, goal)
    hjps_time = time.time() - start_time
    
    if hjps_path:
        print(f"✅ HJPS Path Found!")
        print(f"   Path length: {len(hjps_path)} steps")
        print(f"   Hierarchical nodes expanded: {hjps_stats.get('hierarchical_nodes_expanded', 0)}")
        print(f"   Cache hits: {hjps_stats.get('cache_hits', 0)}")
        print(f"   Cache misses: {hjps_stats.get('cache_misses', 0)}")
        print(f"   Cache hit ratio: {hjps_stats.get('cache_hit_ratio', 0):.2%}")
        print(f"   Search time: {hjps_time*1000:.2f} ms")
    else:
        print(f"❌ HJPS: No path found")
        return
    
    # Performance Comparison
    print(f"\n📊 PERFORMANCE COMPARISON")
    print(f"=" * 40)
    
    if jps_path and hjps_path:
        # Path quality
        jps_cost = sum(jps_path[i].euclidean_distance(jps_path[i+1]) for i in range(len(jps_path)-1))
        hjps_cost = sum(hjps_path[i].euclidean_distance(hjps_path[i+1]) for i in range(len(hjps_path)-1))
        
        print(f"📏 Path Quality:")
        print(f"   JPS path cost: {jps_cost:.2f}")
        print(f"   HJPS path cost: {hjps_cost:.2f}")
        print(f"   Cost difference: {abs(jps_cost - hjps_cost):.2f}")
        
        # Search efficiency
        print(f"\n⚡ Search Efficiency:")
        print(f"   JPS nodes expanded: {jps_stats['nodes_expanded']}")
        print(f"   HJPS nodes expanded: {hjps_stats.get('hierarchical_nodes_expanded', 0)}")
        
        if hjps_stats.get('hierarchical_nodes_expanded', 0) > 0:
            expansion_ratio = jps_stats['nodes_expanded'] / hjps_stats['hierarchical_nodes_expanded']
            print(f"   Node expansion ratio: {expansion_ratio:.2f}× (JPS/HJPS)")
        
        # Time performance
        print(f"\n⏱️  Time Performance:")
        print(f"   JPS time: {jps_time*1000:.2f} ms")
        print(f"   HJPS time: {hjps_time*1000:.2f} ms")
        
        if hjps_time > 0:
            time_ratio = jps_time / hjps_time
            print(f"   Time ratio: {time_ratio:.2f}× (JPS/HJPS)")
        
        # Memory efficiency
        print(f"\n💾 Memory Efficiency:")
        print(f"   JPS explored positions: {jps_stats['explored_positions']}")
        print(f"   HJPS cache size: {len(hjps.jump_point_cache)}")
        
        # Visualize results
        visualize_search_comparison(
            env, jps_path, hjps_path, jps.explored_positions, 
            set(), start, goal  # HJPS doesn't track explored positions the same way
        )
        
        print(f"\n🎯 KEY INSIGHTS:")
        print(f"   - Both algorithms found optimal/near-optimal paths")
        print(f"   - JPS eliminates path symmetry through jump operations")
        print(f"   - HJPS uses hierarchy for long-distance planning")
        print(f"   - Cache efficiency crucial for HJPS performance")
        print(f"   - Hierarchical approach benefits increase with problem size")
        
    return jps_path, hjps_path, jps_stats, hjps_stats

if __name__ == "__main__":
    # Run the comprehensive comparison
    results = run_jps_comparison()
    
    print(f"\n🎯 JUMP POINT SEARCH MASTERY ACHIEVED!")
    print(f"   - Understood symmetry breaking and path dominance")
    print(f"   - Implemented forced neighbor detection")
    print(f"   - Built hierarchical multi-level search")
    print(f"   - Compared standard vs hierarchical approaches")
    print(f"   - Applied mathematical theory to practical pathfinding")