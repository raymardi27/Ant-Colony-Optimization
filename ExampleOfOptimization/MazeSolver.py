from manim import *
import numpy as np
from collections import deque

class MazeSolverScene(Scene):
    def construct(self):
        # Maze configuration
        maze = [
            "##########",
            "#S#     F#",
            "# # #### #",
            "# #    # #",
            "# #### # #",
            "#      # #",
            "##########",
        ]
        
        cell_size = 0.5
        maze_grid, maze_shift = self.create_maze(maze, cell_size)
        self.play(Create(maze_grid))
        self.wait(1)
        
        # Find start and finish positions
        start_pos, finish_pos = self.get_positions(maze)
        
        # Place start and finish dots
        start_dot = Dot(color=GREEN).move_to(self.maze_to_coords(*start_pos, cell_size, maze_shift))
        finish_dot = Dot(color=RED).move_to(self.maze_to_coords(*finish_pos, cell_size, maze_shift))
        self.play(FadeIn(start_dot), FadeIn(finish_dot))
        self.wait(1)
        
        # Find and draw the path using BFS
        path = self.find_path(maze)
        if path:
            path_lines = VGroup()
            for i in range(len(path)-1):
                start_point = self.maze_to_coords(path[i][0], path[i][1], cell_size, maze_shift)
                end_point = self.maze_to_coords(path[i+1][0], path[i+1][1], cell_size, maze_shift)
                line = Line(start_point, end_point, color=BLUE, stroke_width=4)
                path_lines.add(line)
                self.play(Create(line), run_time=0.2)
        
        # Wait so you can see the final path
        self.wait(3)
    
    def create_maze(self, maze, cell_size):
        """Creates a visual representation of the maze and returns the maze VGroup and its shift."""
        grid = VGroup()
        for i, row in enumerate(maze):
            for j, char in enumerate(row):
                if char == "#":
                    square = Square(side_length=cell_size, color=WHITE, fill_color=BLACK, fill_opacity=1)
                    square.move_to([j * cell_size, -i * cell_size, 0])
                    grid.add(square)
        
        # Center the maze
        grid_shift = -grid.get_center()
        grid.shift(grid_shift)
        
        return grid, grid_shift
    
    def maze_to_coords(self, i, j, cell_size, maze_shift):
        """Convert maze cell (i, j) to scene coordinates."""
        x = j * cell_size
        y = -i * cell_size
        return np.array([x, y, 0]) + maze_shift
    
    def get_positions(self, maze):
        """Find (i,j) coordinates of S and F."""
        start = None
        finish = None
        for i, row in enumerate(maze):
            for j, char in enumerate(row):
                if char == "S":
                    start = (i, j)
                elif char == "F":
                    finish = (i, j)
        return start, finish
    
    def find_path(self, maze):
        """Finds a path using BFS."""
        start, finish = self.get_positions(maze)
        if not start or not finish:
            return []
        
        queue = deque([start])
        visited = {start: None}
        
        while queue:
            current = queue.popleft()
            if current == finish:
                break
            neighbors = self.get_neighbors(current, maze)
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited[neighbor] = current
                    queue.append(neighbor)
        
        # Reconstruct the path if found
        path = []
        current = finish
        while current is not None:
            path.append(current)
            current = visited.get(current)
        
        path.reverse()
        return path if path and path[0] == start else []
    
    def get_neighbors(self, position, maze):
        directions = [(-1,0),(1,0),(0,-1),(0,1)]
        neighbors = []
        i, j = position
        for di, dj in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < len(maze) and 0 <= nj < len(maze[0]):
                if maze[ni][nj] != "#":
                    neighbors.append((ni, nj))
        return neighbors
