import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, MultiPolygon
from shapely.ops import unary_union
from tqdm import tqdm
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
import time, heapq, copy, operator, math
from collections import deque



# Helper functions
def arrange_vertices_ccw(vertices):
    vertices = np.array(vertices)
    centroid = np.mean(vertices, axis=0)
    angles = np.arctan2(vertices[:, 1] - centroid[1], vertices[:, 0] - centroid[0])
    sorted_indices = np.argsort(angles)
    sorted_vertices = vertices[sorted_indices]
    return sorted_vertices.tolist()
        
def are_collinear(vertices):
    if len(vertices) < 3:
        return True  # Less than 3 points are trivially collinear           
    x1, y1 = vertices[0]
    x2, y2 = vertices[1]
    # Calculate the coefficients of the line equation (y - y1) = m(x - x1)
    # where m = (y2 - y1) / (x2 - x1) if x2 != x1
    if x2 != x1:
        m = (y2 - y1) / (x2 - x1)
        c = y1 - m * x1
    else:
        # If x2 == x1, the line is vertical
        return all(x == x1 for x, _ in vertices)
    # Check if all other points lie on this line
    for x, y in vertices[2:]:
        if not np.isclose(y, m * x + c, atol=1e-6):
            return False
    return True
        
def has_duplicates(points, atol=1e-6):
    seen = set()
    for point in points:
        # Convert each point to a tuple rounded to avoid floating-point issues
        rounded_point = tuple(np.round(point, decimals=int(-np.log10(atol))))
        if rounded_point in seen:
            return True
        seen.add(rounded_point)
    return False

def bresenham_line(x0, y0, x1, y1):
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        steep = dy > dx

        # Swap roles of x and y if the line is steep
        if steep:
            x0, y0 = y0, x0
            x1, y1 = y1, x1

        # Ensure the line is drawn from left to right
        if x0 > x1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)

        err = dx // 2
        ystep = 1 if y0 < y1 else -1
        y = y0

        for x in range(x0, x1 + 1):
            coord = (y, x) if steep else (x, y)
            points.append(coord)
            err -= dy
            if err < 0:
                y += ystep
                err += dx

        return points


class Environment:
    def __init__(self, size_x, size_y, n_obs=3, resolution=0.1, random_seed=None):
        """
        Initialize the 2D environment.
        
        :param width: Width of the environment
        :param height: Height of the environment
        """
        self.size_x = size_x
        self.size_y = size_y
        self.resolution = resolution
        self.obs = []
        self.cspace, self.x_grid, self.y_grid = None, None, None
        self.start, self.goal = None, None
        self.origins = []

        if random_seed is not None:
            np.random.seed(random_seed)

        self.generate_obstacles(n_obs) # Generate self.obs
        start_time = time.time()
        self.generate_cspace(resolution) # Generate self.cspace, self.x_grid, self.y_grid
        end_time = time.time()
        self.cspace_generation_time = end_time - start_time
        self.generate_start_goal() # Generate self.start, self.goal
        self.current_pos = self.start


    def generate_cspace(self, resolution=0.1):
        # Generate a grid of xy values with resolution
        x_values = np.arange(0, self.size_x + resolution, resolution)
        y_values = np.arange(0, self.size_y + resolution, resolution)
    
        # Initialize the C-space matrix with True (assuming all configurations collision free initially)
        self.cspace = np.ones((len(x_values), len(y_values)), dtype=bool)

        for i, x in enumerate(tqdm(x_values, desc="Generating configuration space", 
                                   bar_format='{desc}: {bar}|{percentage:3.0f}%')):
            for j, y in enumerate(y_values):
                if self.check_collision((x, y), buffer=resolution/2):
                    self.cspace[i, j] = False
    
        # Create a meshgrid for theta values to facilitate visualization or further processing
        self.x_grid, self.y_grid = np.meshgrid(x_values, y_values, indexing='ij')


    def check_collision(self, point, buffer=0.05):
        point = Point(point)
        buffered_obs = [obs.buffer(buffer) for obs in self.obs]
        combined_buffered_obs = unary_union(buffered_obs)
        return combined_buffered_obs.intersects(point)
    
    
    def check_line_collision(self, p1, p2, step=100, buffer=0.1):
        x1, y1 = p1
        x2, y2 = p2
        
        # Linear interpolation
        t = np.linspace(0, 1, step)
        xi = x1 + t * (x2 - x1)
        yi = y1 + t * (y2 - y1)

        for x, y in zip(xi, yi):
            if self.check_collision((x, y), buffer):
                return True
        return False


    def add_polyobs(self, vertices):
        polygon = Polygon(vertices)

        if not polygon.is_valid:
            print("Fail to add obstacles! Polygon is not valid.")
            return
        
        self.obs.append(polygon)


    def generate_obstacles(self, n_obs, max_vertices=6):
        # Generate a random number of random polygon obstacles.

        for _ in range(n_obs):
            # Randomly generate the number of vertices for the polygon
            n_vertices = np.random.randint(3, max_vertices + 1)

            # Randomly generate a center point for the polygon
            center_x = np.random.uniform(1, self.size_x - 1)
            center_y = np.random.uniform(1, self.size_y - 1)
            
            while True:
                # Generate vertices around the center point
                vertices = []
                for i in range(n_vertices):
                    angle = np.random.uniform(0, 2*np.pi)
                    radius = np.random.uniform(min(self.size_x, self.size_y)/5, max(self.size_x, self.size_y)/3)
                    x = center_x + radius * np.cos(angle)
                    y = center_y + radius * np.sin(angle)
                    # Ensure vertices are within bounds
                    x = max(1, min(x, self.size_x - 1))
                    y = max(1, min(y, self.size_y - 1))
                    vertices.append((x, y))

                if are_collinear(vertices) or has_duplicates(vertices):
                    continue

                vertices = arrange_vertices_ccw(vertices)
                polygon = Polygon(vertices)
                intersect = False
                for obs in self.obs:
                    if obs.intersects(polygon):
                        intersect = True
                        break
                if not intersect:
                    break
  
            self.add_polyobs(vertices)


    def generate_start_goal(self):
        # generate random start & goal nodes
        max_attempts = 100
        found_start = False
        found_goal = False
        for _ in range(max_attempts):
            x_start = np.random.rand()*self.size_x
            y_start = np.random.rand()*self.size_y
            if not self.check_collision((x_start, y_start)):
                found_start = True
                break
        for _ in range(max_attempts):
            x_goal = np.random.rand()*self.size_x
            y_goal = np.random.rand()*self.size_y
            if not self.check_collision((x_goal, y_goal)):
                found_goal = True
                break
        if found_start and found_goal:
            self.start, self.goal = (x_start, y_start), (x_goal, y_goal)
        else:
            print("Fail to generate start and goal!")
    

    def xy_to_idx(self, xy):
        # Convert raw (x, y) coordinates into corresponding cspace idx (i, j)
        return round(xy[0] / self.resolution), round(xy[1] / self.resolution)
    

    def idx_to_xy(self, idx):
        return self.x_grid[idx], self.y_grid[idx]
        

    def is_valid_move(self, idx):
        if 0 <= idx[0] < self.cspace.shape[0] and 0 <= idx[1] < self.cspace.shape[1]:
            return self.cspace[idx]
        return False
    

    def ifReachGoal(self, xy):
        xy = np.array(xy)
        return np.linalg.norm(xy - np.array(self.goal)) < 0.15


    def check_line_coverage(self, p1, p2):
        covered_grid = set()
        idx1 = self.xy_to_idx(p1)
        idx2 = self.xy_to_idx(p2)
        idx_list = bresenham_line(idx1[0], idx1[1], idx2[0], idx2[1])
        for idx in idx_list:
            if self.is_valid_move(idx):
                covered_grid.add(idx)
            else: # Collision
                return covered_grid, False
        return covered_grid, True


    def evaluate_performance(self, sol_path, visited_idx, total_visited_idx, output_file='performance_metrics.txt'):
        path_cost = len(total_visited_idx) - 1
        if np.sum(self.cspace) > 0:
            area_coverage = len(visited_idx)/np.sum(self.cspace)*100
        else:
            area_coverage = 100
        reached_goal = self.ifReachGoal(sol_path[-1]) if sol_path else False
        complete = reached_goal and abs(area_coverage - 100) < 1e-8

        # Compile results into a dictionary
        performance_metrics = {
            'path_cost': path_cost,
            'area_coverage': area_coverage,
            'reached_goal': reached_goal,
            'complete': complete
        }

        # Write the performance metrics to a text file
        with open(output_file, 'w') as f:
            f.write(f"Path Cost: {performance_metrics['path_cost']}\n")
            f.write(f"Area Coverage: {performance_metrics['area_coverage']:.2f}%\n")
            f.write(f"Reached Goal: {'Yes' if performance_metrics['reached_goal'] else 'No'}\n")
            f.write(f"Complete: {'Yes' if performance_metrics['complete'] else 'No'}\n")

        return performance_metrics


    def visualize(self, sol_path, strips=None, waypoints=None):
        """
        Visualize the environment with obstacles and robot's position.
        """
        # Calculate the centers of the grid cells
        dx = self.x_grid[1, 0] - self.x_grid[0, 0]
        dy = self.y_grid[0, 1] - self.y_grid[0, 0]
        collide = False
        visited_idx = set()
        visited_idx.add(self.xy_to_idx(sol_path[1]))
        total_visited_idx = [self.xy_to_idx(sol_path[1])]
        
        fig, ax = plt.subplots(figsize=(5, 5))

        for i in range(len(sol_path)):
            ax.clear()
            ax.set_xlim(0, self.size_x)
            ax.set_ylim(0, self.size_y)
            if i > 0:
                idx1 = self.xy_to_idx(sol_path[i-1])
                idx2 = self.xy_to_idx(sol_path[i])
                idx_list = bresenham_line(idx1[0], idx1[1], idx2[0], idx2[1])
                for idx in idx_list:
                    if self.is_valid_move(idx):
                        visited_idx.add(idx)
                        total_visited_idx.append(idx)
                        last_valid_idx = idx
                    else:
                        print("Collision occur!")
                        print(idx)
                        collide = True
                        collision_idx = last_valid_idx
                        break
        
            # Plot the obstacles
            for obs in self.obs:
                x, y = obs.exterior.xy
                ax.plot(x, y, 'r')
            
            # Plot the robot's current position
            ax.plot(self.start[0], self.start[1], 'y^', markersize=12)
            ax.plot(self.goal[0], self.goal[1], 'g*', markersize=12)

            for i in range(len(self.origins)):
                ax.plot(self.origins[i][0], self.origins[i][1], '*', markersize=4)

            self.current_pos = sol_path[i]
            ax.plot(self.current_pos[0], self.current_pos[1], 'bo')
            if i > 0:
                self.previous_pos = sol_path[i - 1]
                ax.plot(self.previous_pos[0], self.previous_pos[1], 'bo', alpha=0.5)
                ax.plot((self.previous_pos[0], self.current_pos[0]), (self.previous_pos[1], self.current_pos[1]), 'b', alpha=0.5)

            # Plot the C-space grid with light lines
            for i in range(self.x_grid.shape[0]):
                ax.plot(self.x_grid[i, :] - dx/2, self.y_grid[i, :], 'k-', lw=0.5, alpha=0.15)
            for j in range(self.x_grid.shape[1]):
                ax.plot(self.x_grid[:, j], self.y_grid[:, j] - dy/2, 'k-', lw=0.5, alpha=0.15)

            occupied_cells = np.where(~self.cspace)
            for i, j in zip(*occupied_cells):
                rect = plt.Rectangle((self.x_grid[i, j] - dx / 2,
                                    self.y_grid[i, j] - dy / 2),
                                    dx, dy,
                                    facecolor='red', edgecolor='none', alpha=0.25)
                ax.add_patch(rect)

            for idx in visited_idx:
                rect = plt.Rectangle((self.x_grid[idx[0], idx[1]] - dx / 2,
                                    self.y_grid[idx[0], idx[1]] - dy / 2),
                                    dx, dy,
                                    facecolor='green', edgecolor='none', alpha=0.25)
                ax.add_patch(rect)

            if strips is not None:
                for strip in strips:
                    if isinstance(strip, MultiPolygon):
                        print('Catch error')
                        for i, polygon in enumerate(strip.geoms):
                            x, y = polygon.exterior.xy
                            print(f"polygon {i}, {x, y}")
                    x, y = strip.exterior.xy
                    ax.plot(x, y, 'm')

            if waypoints is not None:
                # Normalize the indices of waypoints to use the full range of the colormap
                norm = Normalize(vmin=0, vmax=len(waypoints) - 1)
                cmap = get_cmap('plasma')

                # Plot each waypoint with its corresponding color from the colormap
                for idx, (x, y) in enumerate(waypoints):
                    color = cmap(norm(idx))
                    ax.plot(x, y, 'd', color=color, alpha=0.3)
            
            plt.draw()
            plt.pause(0.01)

            if collide: break
            
        print(self.evaluate_performance(sol_path, visited_idx, total_visited_idx))
        plt.show()

    def bfs(self):
        """
        Perform BFS to find the shortest path from initial_state to goal_state.

        Args:
            initial_state (np.ndarray): The starting state as a 2D array.
            goal_state (np.ndarray): The goal state as a 2D array.

        Returns:
            list: The sequence of coordinates representing moves from the initial blank position to the goal blank position.
        """
        # Convert states to tuples for hashability
        initial_tuple = tuple(map(tuple, self.cspace))
        goal_tuple = tuple(map(tuple, np.zeros([len(self.cspace), len(self.cspace[0])], dtype=bool)))

        # Find the initial position of the blank (0)
        initial_pos_idx = tuple(self.xy_to_idx(self.start))

        # BFS setup
        queue = deque([(initial_tuple, initial_pos_idx)])  # Each entry is (state, position)
        visited = {tuple([initial_tuple, initial_pos_idx]): None}  # Maps state to parent and move taken

        while queue:
            current_state, current_pos_idx = queue.popleft()

            # Check if we've reached the goal
            if (current_state == goal_tuple and current_pos_idx == self.xy_to_idx(self.goal)):
                path = []
                trace_state = current_state
                trace_pos_idx = current_pos_idx
                while trace_state != initial_tuple:
                    parent_state, parent_idx = visited[trace_state, trace_pos_idx]
                    path.append((self.x_grid[parent_idx[0]][parent_idx[1]], self.y_grid[parent_idx[0]][parent_idx[1]]))
                    trace_state = parent_state
                    trace_pos_idx = parent_idx
                return path[::-1]  # Reverse the moves

            # Generate neighbors and add them to the queue
            neighbors = []
            n_rows, n_cols = np.array(current_state).shape
            x_idx, y_idx = current_pos_idx

            # Possible moves: up, down, left, right
            moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            for dx, dy in moves:
                new_x_idx, new_y_idx = x_idx + dx, y_idx + dy

                # print(self.is_valid_move((new_x_idx, new_y_idx)))
                # print(self.check_collision((self.x_grid[new_x_idx][new_y_idx], self.y_grid[new_x_idx][new_y_idx])))

                if 0 <= new_x_idx < n_rows and 0 <= new_y_idx < n_cols and self.is_valid_move((new_x_idx, new_y_idx)) and not self.check_collision((self.x_grid[new_x_idx][new_y_idx], self.y_grid[new_x_idx][new_y_idx])):
                    # Create a new state by swapping the blank space
                    new_state = np.array(current_state)
                    new_state[x_idx][y_idx] = 0
                    neighbors.append([new_state, (new_x_idx, new_y_idx)])

            # print(neighbors)

            for neighbor, new_idx in neighbors:
                neighbor_tuple = tuple(map(tuple, neighbor))
                new_idx_tuple = tuple(new_idx)
                if (tuple([neighbor_tuple, new_idx]) not in visited):
                    visited[tuple([neighbor_tuple, new_idx_tuple])] = tuple([current_state, current_pos_idx])
                    queue.append([neighbor_tuple, new_idx])

        return []  # No path found`


    def spiral_wall_following(self):
        """
        Execute the cleaning algorithm.
        :return: List of positions in the order they were visited.
        """

        grid_state = copy.deepcopy(self.cspace)
        spiral_grid_state = copy.deepcopy(self.cspace)
        rows, cols = self.cspace.shape
        goal_state = np.zeros([rows, cols], dtype=bool)
        current_pos = self.xy_to_idx(self.start)
        direction_index = 0  # Initial direction (right)
        directions = [
            (0, 1),  # right
            (1, 0),  # down
            (0, -1), # left
            (-1, 0)  # up
        ]
        clockwise = True
        path = []

        while not np.array_equal(grid_state, goal_state):

            while True:
                if np.array_equal(grid_state, goal_state):
                    break

                # print(current_pos)
                # print(direction_index)
                spiral_grid_state[current_pos[0]][current_pos[1]] = 0
                grid_state[current_pos[0]][current_pos[1]] = 0
                path.append((self.x_grid[current_pos[0]][current_pos[1]], self.y_grid[current_pos[0]][current_pos[1]]))

                # Calculate next position
                op_relative = operator.add
                op_reverse = operator.sub
                if not clockwise:
                    op_relative = operator.sub
                    op_reverse = operator.add

                new_pos = (current_pos[0] + directions[op_relative(direction_index, 1) % 4][0], current_pos[1] + directions[op_relative(direction_index, 1) % 4][1]) 
                if not self.is_valid_move(new_pos) or spiral_grid_state[new_pos[0]][new_pos[1]] == 0:
                    new_pos = (current_pos[0] + directions[direction_index][0], current_pos[1] + directions[direction_index][1])
                    if not self.is_valid_move(new_pos) or spiral_grid_state[new_pos[0]][new_pos[1]] == 0:
                        new_pos = (current_pos[0] + directions[op_reverse(direction_index, 1) % 4][0], current_pos[1] + directions[op_reverse(direction_index, 1) % 4][1])
                        if not self.is_valid_move(new_pos) or spiral_grid_state[new_pos[0]][new_pos[1]] == 0:
                            new_pos = (current_pos[0] + directions[op_relative(direction_index, 2) % 4][0], current_pos[1] + directions[op_relative(direction_index, 2) % 4][1])
                            direction_index = op_relative(direction_index, 2) % 4
                        else:
                            direction_index = op_reverse(direction_index, 1) % 4
                            clockwise = not clockwise
                else:
                    direction_index = op_relative(direction_index, 1) % 4

                if(not self.is_valid_move(new_pos) or spiral_grid_state[new_pos[0]][new_pos[1]] == 0):
                    break
                current_pos = new_pos

            if np.array_equal(goal_state, grid_state):
                break

            #reset start point
            max_attempts = 100
            for _ in range(max_attempts):
                x_start = np.random.rand()*self.size_x
                y_start = np.random.rand()*self.size_y
                if not self.check_collision((x_start, y_start)):
                    break

            new_start = self.xy_to_idx((x_start, y_start))
            # self.origins.append((self.x_grid[new_start[0]], self.y_grid[new_start[1]]))
            visited = {}
            neighbors = []

            manhattan_distance =  math.sqrt((current_pos[0] - new_start[0]) ** 2 + (current_pos[1] - new_start[1]) ** 2)
            heapq.heappush(neighbors, tuple((manhattan_distance, current_pos)))
            visited[current_pos] = None
            initial_pos = current_pos
            while neighbors:
                current_pos = heapq.heappop(neighbors)[1]
                if current_pos == new_start:
                    temp_path = []
                    trace = current_pos
                    while trace != initial_pos:
                        parent = visited[trace]
                        grid_state[trace[0]][trace[1]] = 0
                        temp_path.append((self.x_grid[parent[0]][parent[1]], self.y_grid[parent[0]][parent[1]]))
                        trace = parent
                    for i in range(len(temp_path)):
                        path.append(temp_path[len(temp_path) - i - 1])
                    break

                for dx, dy in directions:
                    new_x_idx = current_pos[0] + dx
                    new_y_idx = current_pos[1] + dy

                    if self.is_valid_move((new_x_idx, new_y_idx)) and (new_x_idx, new_y_idx) not in visited:
                        new_pos = (new_x_idx, new_y_idx)
                        visited[new_pos] = current_pos
                        manhattan_distance =  math.sqrt((new_x_idx - new_start[0]) ** 2 + (new_y_idx - new_start[1]) ** 2)
                        heapq.heappush(neighbors, tuple((manhattan_distance, new_pos)))

            spiral_grid_state = copy.deepcopy(self.cspace)
            # print(grid_state)

        return path
    

# Example Usage
if __name__ == '__main__':
    # Create a 20x20 environment
    env = Environment(5, 5, 3, 0.2, 1)
    # env.add_polyobs([(2, 2), (2, 3), (3, 3), (3, 2)])
    # env.generate_cspace()
    # env.generate_start_goal()
    start_time = time.time()
    sol_path = env.spiral_wall_following()
    end_time = time.time()
    total_time = end_time - start_time + env.cspace_generation_time
    # for dy in np.arange(0, 2, 0.1):
    #     sol_path.append((env.start[0], env.start[1] + dy))
    
    # Add polygon obstacles
    # env.add_polyobs([(3, 3), (7, 3), (7, 7), (3, 7)])  # Square obstacle
    # env.add_polyobs([(12, 12), (16, 12), (16, 16), (12, 16)])  # Another square
    
    # Visualize the environment
    print(total_time)
    # print(sol_path)
    env.visualize(sol_path)
    