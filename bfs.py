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