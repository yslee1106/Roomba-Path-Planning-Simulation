def heuritics(self, grid_state, goal_state, current_idx):
        # Number of uncovered spaces
        uncovered_count = np.sum(grid_state != goal_state)

        # Distance to the closest uncovered space
        rows, cols = grid_state.shape
        row_idx, col_idx = current_idx
        distances = []

        for i in range(rows):
            i = i + 1
            row_idx = row_idx - i
            col_idx = col_idx - i

            for dr in range(i + 2):
                if 0 < row_idx + dr < rows and 0 < col_idx + i + 1 < cols:
                    if not grid_state[row_idx + dr][col_idx] == goal_state[row_idx + dr][col_idx] or not grid_state[row_idx + dr][col_idx + i + 1] == goal_state[row_idx + dr][col_idx + i + 1]:
                        distances.append(i)
                        break
                else:
                    break

            if len(distances) > 0: break

            for dc in range(i + 2):
                if 0 < row_idx + i + 1 < rows and 0 < col_idx + dc < cols:
                    if not grid_state[row_idx][col_idx + dc] == goal_state[row_idx][col_idx + dc] or not grid_state[row_idx + i + 1][col_idx + dc] == goal_state[row_idx + i + 1][col_idx + dc]:
                        distances.append(i)
                        break
                else:
                    break
            
            if len(distances) > 0: break

        if len(distances) == 0: distances.append((rows + cols) / 2)

        closest_distance = min(distances) if distances else 0

        # Combine metrics (higher heuristic for worse states)
        return uncovered_count + closest_distance

 def heuritics(self, grid_state, goal_state, current_idx):
        # Number of uncovered spaces
        uncovered_count = np.sum(grid_state != goal_state)

        # Distance to the closest uncovered space
        rows, cols = grid_state.shape
        r, c = current_idx
        distances = []

        for i in range(r):
            idx = r - i
            if grid_state[idx][c] != goal_state[idx][c]:
                distances.append(i)
                break

        for i in range(rows - r):
            idx = r + i
            if grid_state[idx][c] != goal_state[idx][c]:
                distances.append(i)
                break

        for i in range(c):
            idx = c - i
            if grid_state[r][idx] != goal_state[r][idx]:
                distances.append(i)
                break

        for i in range(cols - c):
            idx = c + i
            if grid_state[r][idx] != goal_state[r][idx]:
                distances.append(i)
                break

        if len(distances) == 0: distances.append(0)
        closest_distance = min(distances) if distances else 0


        return uncovered_count + (0.25 * closest_distance)



    def aStar(self):
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
        goal_state = np.zeros([len(self.cspace), len(self.cspace[0])], dtype=bool)
        goal_tuple = tuple(map(tuple, goal_state))
        goal_idx = self.xy_to_idx(self.goal)

        # Find the initial position of the blank (0)
        initial_pos_idx = tuple(self.xy_to_idx(self.start))
        
        # Heuristics Setup
        initial_cost = self.heuritics(self.cspace, goal_state, initial_pos_idx)

        # BFS setup
        priority_queue = []  # Each entry is (state, (cost, depth, grid_state, robot_index_position))
        heapq.heappush(priority_queue, (initial_cost, initial_tuple, initial_pos_idx))
        visited = {tuple([initial_tuple, initial_pos_idx]): None}  # Maps state to parent and move taken

        while priority_queue:
            current_cost, current_state, current_pos_idx = heapq.heappop(priority_queue)

            # Check if we've reached the goal
            if (np.all(current_state == goal_state) and current_pos_idx == self.xy_to_idx(self.goal)):
                path = []
                path.append((self.x_grid[current_pos_idx[0]][current_pos_idx[1]], self.y_grid[current_pos_idx[0]][current_pos_idx[1]]))
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
                new_x_idx = x_idx + dx
                new_y_idx = y_idx + dy

                # print(self.is_valid_move((new_x_idx, new_y_idx)))
                # print(self.check_collision((self.x_grid[new_x_idx][new_y_idx], self.y_grid[new_x_idx][new_y_idx])))

                if 0 <= new_x_idx < n_rows and 0 <= new_y_idx < n_cols and self.is_valid_move((new_x_idx, new_y_idx)) and not self.check_collision((self.x_grid[new_x_idx][new_y_idx], self.y_grid[new_x_idx][new_y_idx])):
                    new_state = np.array(current_state)
                    new_state[new_x_idx][new_y_idx] = 0
                    neighbors.append([new_state, (new_x_idx, new_y_idx)])

            # print(len(neighbors))

            for neighbor_state, new_idx in neighbors:
                neighbor_state_tuple = tuple(map(tuple, neighbor_state))
                new_idx_tuple = tuple(new_idx)
                new_cost = self.heuritics(neighbor_state, goal_state, new_idx)
                if (tuple([neighbor_state_tuple, new_idx]) not in visited):
                    visited[tuple([neighbor_state_tuple, new_idx_tuple])] = tuple([current_state, current_pos_idx])
                    heapq.heappush(priority_queue, (new_cost, neighbor_state_tuple, new_idx))

        return []  # No path found`