from manim import *
import numpy as np
import random

class TSPComparisonScene(Scene):
    def construct(self):
        # Set seed for reproducibility
        random.seed(42)
        np.random.seed(42)

        # Title and definition
        title = Text("Traveling Salesman Problem (TSP)", font_size=40).to_edge(UP)
        definition = Text("Given a set of cities, find the shortest route visiting all cities exactly once and returning to the start.", font_size=24).next_to(title, DOWN)
        self.play(Write(title), run_time=1.5)
        self.play(Write(definition), run_time=3)
        self.wait(1)
        self.play(FadeOut(definition), title.animate.to_edge(UP).scale(0.8))

        # Generate cities
        num_cities = 5
        # Random points (x,y) within a certain area
        coords = np.array([[random.uniform(-4,4), random.uniform(-2,2), 0] for _ in range(num_cities)])
        
        # Create city dots and labels
        cities = VGroup()
        for i,(x,y,z) in enumerate(coords):
            dot = Dot([x,y,z], color=BLUE)
            label = Text(str(i), font_size=20).next_to(dot, DOWN)
            city_group = VGroup(dot, label)
            cities.add(city_group)

        self.play(LaggedStart(*[FadeIn(c) for c in cities], lag_ratio=0.3))
        self.wait(1)

        # Draw all edges (Complete graph)
        edges = VGroup()
        for i in range(num_cities):
            for j in range(i+1, num_cities):
                line = Line(coords[i], coords[j], color=GRAY)
                edges.add(line)

        self.play(Create(edges), run_time=2)
        self.wait(1)

        # We'll show NN on the left and ACO on the right. Let's shift the scene:
        # Move everything slightly left to leave room for side-by-side demonstration
        # We'll clone the cities and edges for the ACO side
        left_shift = LEFT*3
        right_shift = RIGHT*3

        # Original cities and edges represent the "middle"
        # Let's clone for two sides
        nn_cities = cities.copy().shift(left_shift)
        nn_edges = edges.copy().shift(left_shift)
        aco_cities = cities.copy().shift(right_shift)
        aco_edges = edges.copy().shift(right_shift)

        self.play(
            FadeOut(cities), FadeOut(edges),
            FadeIn(nn_cities), FadeIn(nn_edges),
            FadeIn(aco_cities), FadeIn(aco_edges)
        )

        # Labels for each side
        nn_label = Text("Nearest Neighbor", font_size=24).next_to(nn_cities, UP).shift(UP*0.5)
        aco_label = Text("Ant Colony Optimization", font_size=24).next_to(aco_cities, UP).shift(UP*0.5)

        self.play(Write(nn_label), Write(aco_label))
        self.wait(1)

        # Function to compute distance matrix
        def dist_matrix(coords):
            n = len(coords)
            d = np.zeros((n,n))
            for i in range(n):
                for j in range(n):
                    if i != j:
                        d[i,j] = np.linalg.norm(coords[i]-coords[j])
            return d

        dists = dist_matrix(coords)

        # NEAREST NEIGHBOR HEURISTIC
        # Pick a start city
        start = 0
        unvisited = list(range(num_cities))
        unvisited.remove(start)
        nn_path = [start]
        current = start

        while unvisited:
            # find nearest city
            nearest = min(unvisited, key=lambda c: dists[current,c] if c!=current else float('inf'))
            nn_path.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        # Return to start
        nn_path.append(start)

        # Draw NN path
        nn_lines = VGroup()
        for i in range(len(nn_path)-1):
            c1 = nn_path[i]
            c2 = nn_path[i+1]
            line = Line(coords[c1]+[0], coords[c2]+[0], color=YELLOW)
            line.shift(left_shift)
            nn_lines.add(line)

        self.play(Create(nn_lines))
        self.wait(1)

        nn_length = sum(dists[nn_path[i], nn_path[i+1]] for i in range(len(nn_path)-1))

        # ACO SIMULATION (Simplified)
        # Initialize pheromones
        pheromones = np.ones((num_cities,num_cities))
        # Simple symmetric: pheromones[j,i] = pheromones[i,j]
        # We'll do a small number of iterations
        aco_iterations = 10
        num_ants = 8
        evaporation_rate = 0.3
        Q = 1.0

        best_aco_path = None
        best_aco_length = float('inf')

        for iteration in range(aco_iterations):
            all_paths = []
            # Each ant constructs a path
            for _a in range(num_ants):
                visited = [start]
                unvis = set(range(num_cities))
                unvis.remove(start)
                while unvis:
                    current = visited[-1]
                    # Probabilities based on pheromones and 1/distance
                    # alpha=1,beta=2 for instance
                    alpha=1.0
                    beta=2.0
                    weights = []
                    candidates = list(unvis)
                    for nxt in candidates:
                        tau = pheromones[current,nxt]**alpha
                        eta = (1.0/dists[current,nxt])**beta if dists[current,nxt]!=0 else 1e9
                        weights.append(tau*eta)
                    weights = np.array(weights)
                    weights = weights / weights.sum()
                    # Choose next city
                    nxt_city = np.random.choice(candidates, p=weights)
                    visited.append(nxt_city)
                    unvis.remove(nxt_city)
                visited.append(start) # return to start

                length = sum(dists[visited[i],visited[i+1]] for i in range(len(visited)-1))
                all_paths.append((visited,length))
                if length < best_aco_length:
                    best_aco_length = length
                    best_aco_path = visited
            
            # Update pheromones
            # Evaporation
            pheromones = (1-evaporation_rate)*pheromones
            # Deposit
            for path,length in all_paths:
                deposit_amount = Q/length
                for i in range(len(path)-1):
                    a = path[i]
                    b = path[i+1]
                    pheromones[a,b] += deposit_amount
                    pheromones[b,a] += deposit_amount
        
        # Draw best ACO path
        aco_lines = VGroup()
        for i in range(len(best_aco_path)-1):
            c1 = best_aco_path[i]
            c2 = best_aco_path[i+1]
            line = Line(coords[c1], coords[c2], color=GREEN)
            line.shift(right_shift)
            aco_lines.add(line)

        self.play(Create(aco_lines))
        self.wait(1)

        # Display Path lengths at the bottom
        # We'll show two Text Mobjects: "NN Path Length: X" and "ACO Path Length: Y"
        nn_length_text = Text(f"NN Path Length: {nn_length:.2f}", font_size=24, color=YELLOW).to_edge(DOWN).shift(LEFT*3)
        aco_length_text = Text(f"ACO Path Length: {best_aco_length:.2f}", font_size=24, color=GREEN).to_edge(DOWN).shift(RIGHT*3)

        self.play(Write(nn_length_text), Write(aco_length_text))
        self.wait(2)

        # Finally, zoom out / show everything
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(1)
