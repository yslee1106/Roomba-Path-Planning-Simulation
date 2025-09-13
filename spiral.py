def spiral_wall_following(self):
        """
        Execute the cleaning algorithm.
        :return: List of positions in the order they were visited.
        """

        grid_state = copy.deepcopy(self.cspace)
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

        while grid_state[current_pos[0]][current_pos[1]] != 0:
            grid_state[current_pos[0]][current_pos[1]] = 0
            path.append((self.x_grid[current_pos[0]][current_pos[1]], self.y_grid[current_pos[0]][current_pos[1]]))

            # Calculate next position
            op_relative = operator.add
            op_reverse = operator.sub
            if not clockwise:
                op_relative = operator.sub
                op_reverse = operator.add

            new_pos = (current_pos[0] + directions[op_relative(direction_index, 1) % 4][0], current_pos[1] + directions[op_relative(direction_index, 1) % 4][1]) 
            if self.check_collision((self.x_grid[new_pos[0]][new_pos[1]], self.y_grid[new_pos[0]][new_pos[1]])) or grid_state[new_pos[0]][new_pos[1]] == 0:
                new_pos = (current_pos[0] + directions[direction_index][0], current_pos[1] + directions[direction_index][1])
                if not (0 <= new_pos[0] < rows) or not (0 <= new_pos[1] < cols):
                    new_pos = (current_pos[0] + directions[op_reverse(direction_index, 1) % 4][0], current_pos[1] + directions[op_reverse(direction_index, 1) % 4][1])
                    if not (0 <= new_pos[0] < rows) or not (0 <= new_pos[1] < cols):
                        new_pos = (current_pos[0] + directions[op_relative(direction_index, 1) % 4][0], current_pos[1] + directions[op_relative(direction_index, 1) % 4][1])
                        direction_index = op_relative(direction_index, 1) % 4
                    else:
                        direction_index = op_reverse(direction_index, 1) % 4
                        clockwise = not clockwise
                elif self.check_collision((self.x_grid[new_pos[0]][new_pos[1]], self.y_grid[new_pos[0]][new_pos[1]])) or grid_state[new_pos[0]][new_pos[1]] == 0:
                    new_pos = (current_pos[0] + directions[op_reverse(direction_index, 1) % 4][0], current_pos[1] + directions[op_reverse(direction_index, 1) % 4][1])
                    direction_index = op_reverse(direction_index, 1) % 4
            else:
                direction_index = op_relative(direction_index, 1) % 4

            current_pos = new_pos

        return path