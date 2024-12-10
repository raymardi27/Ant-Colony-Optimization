from manim import *
import numpy as np
import random

class AntColonyOptimizationScene(Scene):
    def construct(self):
        # Parameters
        num_paths = 4
        num_ants = 18
        iterations = 10
        path_color = BLUE
        pheromone_color = YELLOW

        # Create Nest (Start) and Food (Goal)
        nest = Dot(LEFT*5, color=GREEN)
        food = Dot(RIGHT*5, color=RED)
        nest_label = Text("Nest", font_size=24).next_to(nest, DOWN)
        food_label = Text("Food", font_size=24).next_to(food, DOWN)

        self.play(FadeIn(nest), FadeIn(food), Write(nest_label), Write(food_label))
        self.wait(1)

        # Generate random paths (cubic Bezier curves or piecewise lines)
        # We'll generate some random control points to create distinct wiggly paths
        paths = []
        for _ in range(num_paths):
            # We'll pick random control points in a vertical range for variety
            control_points = [
                nest.get_center(),
                [random.uniform(-2, 2), random.uniform(-3, 3), 0],
                [random.uniform(2, 4), random.uniform(-3, 3), 0],
                food.get_center()
            ]
            path = CubicBezier(*control_points).set_stroke(width=3, color=path_color)
            paths.append(path)
        
        # Group paths and show them
        path_group = VGroup(*paths)
        self.play(Create(path_group))
        self.wait(1)

        # Initialize pheromone levels
        # We'll store pheromones as a numeric value and visually represent them by line thickness or brightness
        pheromones = np.ones(num_paths) * 1.0  # start all equal

        # Function to update path appearances based on pheromones
        def update_paths():
            max_pher = max(pheromones)
            min_pher = min(pheromones)
            # Normalize pheromone values for visual appearance
            # Let's map pheromone amount to stroke width or brightness
            for i, p in enumerate(paths):
                # Increase thickness and shift color toward YELLOW as pheromones increase
                # Normalized factor:
                factor = (pheromones[i] - min_pher) / (max_pher - min_pher + 1e-6)
                new_width = 3 + factor * 5
                # Interpolate color from BLUE (low pheromone) to YELLOW (high pheromone)
                new_color = interpolate_color(BLUE, YELLOW, factor)
                p.set_stroke(width=new_width, color=new_color)

        update_paths()
        self.wait(1)

        # For simulating ants choosing paths:
        # Initial probability for each path is uniform. After each iteration, 
        # probability is proportional to pheromones.

        # Helper function to choose a path index based on pheromones
        alpha = 1.0
        def choose_path():
            # Compute a weighted probability using pheromones^alpha
            weighted_pher = [ph**alpha for ph in pheromones]
            total_pher = sum(weighted_pher)
            r = random.random() * total_pher
            cum = 0
            for i, wph in enumerate(weighted_pher):
                cum += wph
                if r <= cum:
                    return i
            return len(pheromones)-1
        # def choose_path():
        #     # Probability ~ pheromones[i]
        #     total_pher = sum(pheromones)
        #     r = random.random() * total_pher
        #     cum = 0
        #     for i, ph in enumerate(pheromones):
        #         cum += ph
        #         if r <= cum:
        #             return i
        #     return len(pheromones)-1

        # Represent ants as small dots that move along chosen paths
        ant_agents = VGroup()
        for _ in range(num_ants):
            ant = Dot(color=WHITE, radius=0.05).move_to(nest.get_center())
            ant_agents.add(ant)
        self.play(FadeIn(ant_agents))
        self.wait(1)

        # Run ACO iterations
        for iteration in range(iterations):
            chosen_paths = []
            moves = []
            # Each ant chooses a path and moves along it
            for ant in ant_agents:
                path_idx = choose_path()
                chosen_paths.append(path_idx)
                # Animate ant along the chosen path
                path = paths[path_idx]
                # We parametrize the path as a VMobject. Ant moves from start=0 to end=1
                anim = MoveAlongPath(ant, path, run_time=2)
                moves.append(anim)

            self.play(*moves)
            self.wait(0.5)

            # Ants reached the food. They deposit pheromones on the chosen paths.
            # Increase the pheromone level of each chosen path.
            for p_idx in chosen_paths:
                # Add some pheromone proportional to how many ants took this path
                pheromones[p_idx] += 1.0

            # Update path visuals
            update_paths()
            self.wait(1)

            # Move ants back to nest (to simulate another iteration)
            # In a real ACO, ants would start again from the nest
            # Here, we simply teleport them back for the next iteration
            for ant in ant_agents:
                ant.move_to(nest.get_center())

        # After all iterations, show that one path is now favored
        self.wait(2)
        self.play(FadeOut(ant_agents), FadeOut(path_group), FadeOut(nest), FadeOut(food), FadeOut(nest_label), FadeOut(food_label))
        self.wait(1)
