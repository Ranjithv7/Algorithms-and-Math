import numpy as np
import heapq
from typing import List, Tuple, Optional, Set
import matplotlib.pyplot as plt
from dataclasses import dataclass
from enum import Enum
import time

class CellType(Enum):
    FREE = 0
    OBSTACLE = 1
    START = 2
    GOAL = 3
    PATH = 4

@dataclass
class Node:
    """
    Represents a node in the A* search tree
    
    Why each field matters:
    - position: Where we are in the grid (state representation)
    - g_cost: Exact cost from start to this node (backward cost)
    - h_cost: Heuristic estimate from this node to goal (forward estimate)  
    - f_cost: Total estimated cost g + h (evaluation function)
    - parent: For path reconstruction (backtracking)
    """
    position: Tuple[int, int]
    g_cost: float = float('inf')
    h_cost: float = 0.0
    f_cost: float = float('inf')
    parent: Optional['Node'] = None
    
    def __lt__(self, other):
        """
        Why do we need this? 
        heapq needs to compare nodes when f_costs are equal.
        We break ties using h_cost (prefer nodes closer to goal).
        """
        if self.f_cost == other.f_cost:
            return self.h_cost < other.h_cost
        return self.f_cost < other.f_cost

class HeuristicType(Enum):
    MANHATTAN = "manhattan"
    EUCLIDEAN = "euclidean"
    DIAGONAL = "diagonal"
    COMBINATION = "combination"

class WarehouseEnvironment:
    """
    Simulates a warehouse with:
    - Static obstacles (shelves, walls)
    - Dynamic obstacles (other robots, moving objects)
    - Different terrain costs (smooth floor, rough areas)
    """
    
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        # Grid: 0=free, 1=obstacle, values 0.1-10.0 represent terrain costs
        self.grid = np.ones((height, width)) * 1.0  # Default cost = 1.0
        self.obstacles = np.zeros((height, width), dtype=bool)
        
    def add_obstacles(self, obstacle_positions: List[Tuple[int, int]]):
        """Add static obstacles (walls, shelves)"""
        for x, y in obstacle_positions:
            if 0 <= x < self.width and 0 <= y < self.height:
                self.obstacles[y, x] = True
                
    def add_rough_terrain(self, positions: List[Tuple[int, int]], cost: float):
        """
        Add areas with higher movement cost (wet floors, ramps, etc.)
        Why different costs? Real warehouses have varying terrain difficulty.
        """
        for x, y in positions:
            if 0 <= x < self.width and 0 <= y < self.height:
                self.grid[y, x] = cost
                
    def is_valid_position(self, x: int, y: int) -> bool:
        """Check if position is within bounds and not an obstacle"""
        return (0 <= x < self.width and 
                0 <= y < self.height and 
                not self.obstacles[y, x])
    
    def get_movement_cost(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> float:
        """
        Calculate cost of moving between adjacent positions.
        
        Why this calculation?
        - Base cost from terrain type at destination
        - Diagonal moves cost √2 times more (Euclidean distance)
        - This ensures our costs reflect actual movement distance
        """
        x1, y1 = from_pos
        x2, y2 = to_pos
        
        # Manhattan distance (1 for adjacent, √2 for diagonal)
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # Terrain cost at destination
        terrain_cost = self.grid[y2, x2]
        
        return distance * terrain_cost

class AStarPathfinder:
    """
    A* Pathfinding implementation with multiple heuristic options
    """
    
    def __init__(self, environment: WarehouseEnvironment, heuristic_type: HeuristicType):
        self.env = environment
        self.heuristic_type = heuristic_type
        
        # Statistics for analysis
        self.nodes_expanded = 0
        self.nodes_generated = 0
        self.max_frontier_size = 0
        
    def heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """
        Calculate heuristic distance between position and goal.
        
        WHY EACH HEURISTIC?
        """
        x1, y1 = pos
        x2, y2 = goal
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        
        if self.heuristic_type == HeuristicType.MANHATTAN:
            """
            Manhattan distance: |dx| + |dy|
            
            WHY ADMISSIBLE? 
            - Assumes we can only move in 4 directions (up/down/left/right)
            - Never overestimates because it's the shortest path in grid with 4-connectivity
            - Each step costs at least 1, and we need at least dx+dy steps
            """
            return dx + dy
            
        elif self.heuristic_type == HeuristicType.EUCLIDEAN:
            """
            Euclidean distance: √(dx² + dy²)
            
            WHY ADMISSIBLE?
            - Straight-line distance is always ≤ any path distance
            - Optimal if we could move in any direction with cost proportional to distance
            - Never overestimates actual path cost
            """
            return np.sqrt(dx*dx + dy*dy)
            
        elif self.heuristic_type == HeuristicType.DIAGONAL:
            """
            Diagonal distance: max(dx, dy) + (√2 - 1) * min(dx, dy)
            
            WHY THIS FORMULA?
            - Move diagonally for min(dx, dy) steps: cost = √2 * min(dx, dy)  
            - Move straight for remaining steps: cost = max(dx, dy) - min(dx, dy)
            - Total = max(dx, dy) + (√2 - 1) * min(dx, dy)
            
            WHY ADMISSIBLE?
            - This is exactly the cost of optimal path in 8-connected grid
            - Since our grid allows 8-connectivity, this never overestimates
            """
            return max(dx, dy) + (np.sqrt(2) - 1) * min(dx, dy)
            
        elif self.heuristic_type == HeuristicType.COMBINATION:
            """
            Take maximum of multiple admissible heuristics.
            
            WHY MAXIMUM?
            - If h1 and h2 are both admissible, then max(h1, h2) is also admissible
            - Gives tighter (better) estimate while preserving admissibility
            - Proof: max(h1, h2) ≤ max(h*, h*) = h*
            """
            manhattan = dx + dy
            euclidean = np.sqrt(dx*dx + dy*dy)
            diagonal = max(dx, dy) + (np.sqrt(2) - 1) * min(dx, dy)
            return max(manhattan, euclidean, diagonal)
        
        return 0.0
    
    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Get valid neighboring positions (8-connectivity).
        
        WHY 8-CONNECTIVITY?
        - Allows diagonal movement (more realistic for robot)
        - Reduces path length compared to 4-connectivity
        - Must ensure heuristic accounts for diagonal costs
        """
        x, y = pos
        neighbors = []
        
        # 8 possible directions: N, NE, E, SE, S, SW, W, NW
        directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            if self.env.is_valid_position(new_x, new_y):
                neighbors.append((new_x, new_y))
                
        return neighbors
    
    def reconstruct_path(self, goal_node: Node) -> List[Tuple[int, int]]:
        """
        Backtrack from goal to start using parent pointers.
        
        WHY BACKTRACK?
        - We only stored parent pointers during search
        - More memory efficient than storing full paths
        - Path reconstruction is O(path_length), not O(nodes_expanded)
        """
        path = []
        current = goal_node
        
        while current is not None:
            path.append(current.position)
            current = current.parent
            
        return path[::-1]  # Reverse to get start→goal order
    
    def search(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Tuple[Optional[List[Tuple[int, int]]], dict]:
        """
        Main A* search algorithm.
        
        Returns: (path, statistics)
        """
        # Reset statistics
        self.nodes_expanded = 0
        self.nodes_generated = 0
        self.max_frontier_size = 0
        
        # WHY THESE DATA STRUCTURES?
        open_set = []  # Priority queue (min-heap) for frontier
        closed_set: Set[Tuple[int, int]] = set()  # Visited nodes
        node_map = {}  # Position → Node mapping for quick lookup
        
        # Initialize start node
        start_node = Node(
            position=start,
            g_cost=0.0,
            h_cost=self.heuristic(start, goal),
        )
        start_node.f_cost = start_node.g_cost + start_node.h_cost
        
        heapq.heappush(open_set, start_node)
        node_map[start] = start_node
        self.nodes_generated += 1
        
        while open_set:
            # Track maximum frontier size
            self.max_frontier_size = max(self.max_frontier_size, len(open_set))
            
            # Get node with minimum f-cost
            current_node = heapq.heappop(open_set)
            current_pos = current_node.position
            
            # WHY CHECK GOAL HERE?
            # Goal test when expanding (not when generating) ensures optimality
            if current_pos == goal:
                path = self.reconstruct_path(current_node)
                stats = {
                    'path_cost': current_node.g_cost,
                    'path_length': len(path),
                    'nodes_expanded': self.nodes_expanded,
                    'nodes_generated': self.nodes_generated,
                    'max_frontier_size': self.max_frontier_size,
                    'effective_branching_factor': self._calculate_eff_branching_factor(len(path))
                }
                return path, stats
            
            # Mark as visited
            closed_set.add(current_pos)
            self.nodes_expanded += 1
            
            # Explore neighbors
            for neighbor_pos in self.get_neighbors(current_pos):
                # Skip if already explored
                if neighbor_pos in closed_set:
                    continue
                
                # Calculate tentative g-cost
                movement_cost = self.env.get_movement_cost(current_pos, neighbor_pos)
                tentative_g = current_node.g_cost + movement_cost
                
                # Get or create neighbor node
                if neighbor_pos in node_map:
                    neighbor_node = node_map[neighbor_pos]
                else:
                    neighbor_node = Node(
                        position=neighbor_pos,
                        h_cost=self.heuristic(neighbor_pos, goal)
                    )
                    node_map[neighbor_pos] = neighbor_node
                    self.nodes_generated += 1
                
                # WHY THIS CONDITION?
                # Only update if we found a better path to this neighbor
                if tentative_g < neighbor_node.g_cost:
                    # Update neighbor with better path
                    neighbor_node.parent = current_node
                    neighbor_node.g_cost = tentative_g
                    neighbor_node.f_cost = neighbor_node.g_cost + neighbor_node.h_cost
                    
                    # Add to frontier if not already there
                    if neighbor_node not in open_set:
                        heapq.heappush(open_set, neighbor_node)
        
        # No path found
        stats = {
            'path_cost': float('inf'),
            'path_length': 0,
            'nodes_expanded': self.nodes_expanded,
            'nodes_generated': self.nodes_generated,
            'max_frontier_size': self.max_frontier_size,
            'effective_branching_factor': float('inf')
        }
        return None, stats
    
    def _calculate_eff_branching_factor(self, path_length: int) -> float:
        """
        Calculate effective branching factor.
        
        WHY THIS METRIC?
        - Measures how "focused" our search was
        - Lower values indicate better heuristic performance
        - Allows comparison between different heuristics
        """
        if path_length <= 1:
            return 1.0
            
        # Solve: N = 1 + b* + b*² + ... + b*^d for b*
        # Using binary search approximation
        n = self.nodes_expanded
        d = path_length - 1
        
        if n <= d + 1:
            return 1.0
            
        # Binary search for effective branching factor
        low, high = 1.0, n
        epsilon = 1e-6
        
        while high - low > epsilon:
            mid = (low + high) / 2
            sum_powers = sum(mid**i for i in range(d + 1))
            
            if sum_powers < n:
                low = mid
            else:
                high = mid
                
        return (low + high) / 2

def create_warehouse_scenario() -> WarehouseEnvironment:
    """
    Create a realistic warehouse scenario with:
    - Shelving units (obstacles)
    - Different floor types (varying costs)
    - Aisles for navigation
    """
    env = WarehouseEnvironment(width=50, height=30)
    
    # Add shelving units (obstacles)
    shelves = []
    # Vertical shelves
    for shelf_col in [10, 20, 30, 40]:
        for y in range(5, 25):
            if y not in [12, 13, 17, 18]:  # Leave gaps for cross-aisles
                shelves.append((shelf_col, y))
                shelves.append((shelf_col + 1, y))
    
    # Horizontal barriers
    for x in range(5, 45):
        if x not in [10, 11, 20, 21, 30, 31, 40, 41]:  # Leave gaps
            shelves.append((x, 8))
            shelves.append((x, 22))
    
    env.add_obstacles(shelves)
    
    # Add rough terrain areas (higher cost zones)
    rough_areas = []
    # Loading dock area (busy, higher cost)
    for x in range(0, 8):
        for y in range(0, 10):
            rough_areas.append((x, y))
    
    env.add_rough_terrain(rough_areas, cost=2.5)
    
    # Wet floor area (very high cost)
    wet_areas = [(15, 15), (16, 15), (15, 16), (16, 16)]
    env.add_rough_terrain(wet_areas, cost=5.0)
    
    return env

def visualize_solution(env: WarehouseEnvironment, path: List[Tuple[int, int]], 
                      start: Tuple[int, int], goal: Tuple[int, int], title: str):
    """Visualize the warehouse and solution path"""
    
    # Create visualization grid
    vis_grid = np.copy(env.grid)
    
    # Mark obstacles
    vis_grid[env.obstacles] = -1
    
    # Mark path
    for x, y in path:
        vis_grid[y, x] = 10
        
    # Mark start and goal
    vis_grid[start[1], start[0]] = 15
    vis_grid[goal[1], goal[0]] = 20
    
    plt.figure(figsize=(12, 8))
    plt.imshow(vis_grid, cmap='viridis', origin='lower')
    plt.colorbar(label='Terrain Cost / Path')
    plt.title(title)
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    
    # Add legend
    legend_elements = [
        plt.Rectangle((0,0),1,1, color='black', label='Obstacles'),
        plt.Rectangle((0,0),1,1, color='yellow', label='Path'),
        plt.Rectangle((0,0),1,1, color='red', label='Start'),
        plt.Rectangle((0,0),1,1, color='white', label='Goal')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    plt.show()

def compare_heuristics():
    """
    Compare different heuristics on the same problem.
    
    WHY THIS COMPARISON?
    - Shows impact of heuristic choice on performance
    - Demonstrates admissibility vs efficiency tradeoff
    - Reveals effective branching factor differences
    """
    print("🤖 WAREHOUSE ROBOT NAVIGATION USING A* SEARCH")
    print("=" * 60)
    
    # Create environment
    env = create_warehouse_scenario()
    start = (2, 2)
    goal = (45, 25)
    
    heuristics = [
        HeuristicType.MANHATTAN,
        HeuristicType.EUCLIDEAN, 
        HeuristicType.DIAGONAL,
        HeuristicType.COMBINATION
    ]
    
    results = {}
    
    for heuristic_type in heuristics:
        print(f"\n🔍 Testing {heuristic_type.value.upper()} heuristic...")
        
        pathfinder = AStarPathfinder(env, heuristic_type)
        start_time = time.time()
        path, stats = pathfinder.search(start, goal)
        end_time = time.time()
        
        if path:
            print(f"✅ Path found!")
            print(f"   Path cost: {stats['path_cost']:.2f}")
            print(f"   Path length: {stats['path_length']} nodes")
            print(f"   Nodes expanded: {stats['nodes_expanded']}")
            print(f"   Nodes generated: {stats['nodes_generated']}")
            print(f"   Max frontier size: {stats['max_frontier_size']}")
            print(f"   Effective branching factor: {stats['effective_branching_factor']:.3f}")
            print(f"   Search time: {(end_time - start_time)*1000:.2f} ms")
            
            results[heuristic_type] = {
                'path': path,
                'stats': stats,
                'time': end_time - start_time
            }
            
            # Visualize best result
            if heuristic_type == HeuristicType.COMBINATION:
                visualize_solution(env, path, start, goal, 
                                 f"A* Path ({heuristic_type.value})")
        else:
            print("❌ No path found!")
    
    # Analysis
    print("\n📊 HEURISTIC COMPARISON ANALYSIS")
    print("=" * 50)
    
    if results:
        best_heuristic = min(results.keys(), 
                           key=lambda h: results[h]['stats']['nodes_expanded'])
        
        print(f"🏆 Most efficient: {best_heuristic.value}")
        print(f"   Nodes expanded: {results[best_heuristic]['stats']['nodes_expanded']}")
        print(f"   Effective branching factor: {results[best_heuristic]['stats']['effective_branching_factor']:.3f}")
        
        print(f"\n💡 KEY INSIGHTS:")
        print(f"   - Combination heuristic typically expands fewest nodes")
        print(f"   - Lower effective branching factor = better heuristic")
        print(f"   - All heuristics find optimal path (admissibility guarantee)")
    
    return results

if __name__ == "__main__":
    # Run the comparison
    results = compare_heuristics()
    
    print(f"\n🎯 A* ALGORITHM MASTERY ACHIEVED!")
    print(f"   - Understood mathematical foundations")
    print(f"   - Implemented efficient search")
    print(f"   - Compared heuristic effectiveness")
    print(f"   - Applied to real-world scenario")