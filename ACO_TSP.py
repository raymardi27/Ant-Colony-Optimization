from manim import *
import numpy as np
import random

class TSPACOScene(Scene):
    def construct(self):
        # Seed for reproducibility
        random.seed(42)
        np.random.seed(42)

        # -----------------------
        # INTRO AND DEFINITION
        # -----------------------
        title = Text("Traveling Salesman Problem (TSP)", font_size=36)
        definition = Text(
            "Find the shortest possible route that visits each city exactly once\nand returns to the starting point.",
            font_size=24,
            line_spacing=0.7
        ).next_to(title, DOWN, buff=0.5)

        title.to_edge(UP)
        self.play(Write(title))
        self.wait(1)
        self.play(FadeIn(definition))
        self.wait(2)

        # Fade out definition, keep title at top
        self.play(FadeOut(definition))
        self.wait(1)

        # -----------------------
        # SETUP CITIES AND EDGES
        # -----------------------
        num_cities = 5
        coords = np.array([[random.uniform(-4,4), random.uniform(-2,2),0] for _ in range(num_cities)])

        # Create city dots and labels
        city_dots = VGroup()
        for i, pos in enumerate(coords):
            dot = Dot(pos, color=BLUE)
            label = Text(str(i), font_size=20).next_to(dot, DOWN*0.4)
            city_dots.add(VGroup(dot,label))

        self.play(LaggedStart(*[FadeIn(c) for c in city_dots], lag_ratio=0.2))
        self.wait(1)

        # Draw all edges (Complete graph)
        edges = VGroup()
        for i in range(num_cities):
            for j in range(i+1, num_cities):
                line = Line(coords[i], coords[j], color=GRAY, stroke_width=2)
                edges.add(line)

        self.play(Create(edges))
        self.wait(1)

        # Distance matrix
        def dist_matrix(coords):
            n = len(coords)
            d = np.zeros((n,n))
            for i in range(n):
                for j in range(n):
                    if i!=j:
                        d[i,j] = np.linalg.norm(coords[i]-coords[j])
            return d
        dists = dist_matrix(coords[:,:2]) # only x,y

        # -----------------------
        # NEAREST NEIGHBOR SOLUTION
        # -----------------------
        nn_title = Text("Nearest Neighbor Heuristic", font_size=28)
        nn_title.to_edge(LEFT).shift(UP*2.5)
        self.play(Write(nn_title))
        self.wait(1)

        # Nearest neighbor: pick start = 0
        start = 0
        unvisited = list(range(num_cities))
        unvisited.remove(start)
        nn_path = [start]
        current = start

        highlight_color = YELLOW
        chosen_lines_nn = VGroup()

        # Step-by-step highlight
        for step in range(num_cities-1):
            # Find nearest from current
            nearest = min(unvisited, key=lambda c: dists[current,c])
            unvisited.remove(nearest)
            # Highlight chosen edge
            chosen_edge = Line(coords[current], coords[nearest], color=highlight_color, stroke_width=4)
            chosen_lines_nn.add(chosen_edge)
            self.play(Create(chosen_edge), run_time=1)
            self.wait(0.5)
            nn_path.append(nearest)
            current = nearest

        # Return to start
        end_edge = Line(coords[current], coords[start], color=highlight_color, stroke_width=4)
        chosen_lines_nn.add(end_edge)
        self.play(Create(end_edge), run_time=1)
        nn_path.append(start)
        self.wait(1)

        # Compute NN length
        nn_length = sum(dists[nn_path[i], nn_path[i+1]] for i in range(len(nn_path)-1))
        nn_length_text = Text(f"NN Path Length: {nn_length:.2f}", font_size=24, color=highlight_color)
        nn_length_text.to_edge(DOWN)
        self.play(Write(nn_length_text))
        self.wait(2)

        # -----------------------
        # TRANSITION TO ACO
        # -----------------------
        # Fade out NN solution lines (but keep cities and edges)
        self.play(FadeOut(chosen_lines_nn), FadeOut(nn_length_text))
        self.wait(1)

        aco_title = Text("Ant Colony Optimization (ACO)", font_size=28)
        aco_title.to_edge(RIGHT).shift(UP*2.5)
        self.play(Write(aco_title))
        self.wait(1)

        # Initialize pheromones
        pheromones = np.ones((num_cities,num_cities))
        # Parameters for ACO
        aco_iterations = 5
        num_ants = 6
        evaporation_rate = 0.3
        Q = 1.0
        alpha=1.0
        beta=2.0

        # Ant agents (small dots)
        ant_agents = VGroup()

        # We'll place ants at the start city (0)
        for _ in range(num_ants):
            ant = Dot(coords[start], color=WHITE, radius=0.07)
            ant_agents.add(ant)
        self.play(FadeIn(ant_agents))
        self.wait(1)

        # Function to update edge appearance based on pheromones
        def update_edge_visuals():
            # Map pheromone level to thickness/color
            # We'll map min,max of pheromone to [2,6] width and Blue->Yellow
            all_pher_values = []
            for i in range(num_cities):
                for j in range(i+1,num_cities):
                    all_pher_values.append(pheromones[i,j])
            if not all_pher_values:
                return
            min_pher = min(all_pher_values)
            max_pher = max(all_pher_values)
            for k, e in enumerate(edges):
                # Need to find which cities these are
                # This is hacky: we know edges are in same order as created
                # Let's just recompute i,j from k
                # Actually we can't rely on order easily, let's store them before:
                pass

        # We'll store a mapping of edges to their city pairs
        edge_city_pairs = []
        idx = 0
        for i in range(num_cities):
            for j in range(i+1,num_cities):
                edge_city_pairs.append((i,j))
        # edge_city_pairs aligns with edges in the order they were created

        def refresh_edges():
            min_pher = float('inf')
            max_pher = float('-inf')
            for i in range(num_cities):
                for j in range(i+1, num_cities):
                    val = pheromones[i,j]
                    if val < min_pher:
                        min_pher = val
                    if val > max_pher:
                        max_pher = val
            
            for k, e in enumerate(edges):
                i,j = edge_city_pairs[k]
                val = pheromones[i,j]
                factor = (val - min_pher) / (max_pher - min_pher + 1e-12)
                width = 2 + factor*4
                # Color interpolate BLUE->YELLOW
                # BLUE=(0,0,1), YELLOW=(1,1,0)
                r = factor
                g = factor
                b = 1-factor
                e.set_stroke(width=width, color=rgb_to_color([r,g,b]))

        # Initial pheromone visualization update
        refresh_edges()
        self.wait(1)

        # ACO iterations visualization
        best_aco_path = None
        best_aco_length = float('inf')

        # We'll show ants "walking" in a simplified manner:
        # Each iteration:
        # 1. Each ant constructs a path by choosing next cities based on pheromone and distance.
        # 2. Animate ants moving along chosen edges.
        # 3. Update pheromones and refresh edge visuals.
        # 4. If a better path found, update best path length text.

        aco_length_text = Text("ACO Best Length: N/A", font_size=24, color=GREEN).to_edge(DOWN)
        self.play(Write(aco_length_text))
        self.wait(1)

        for iteration in range(aco_iterations):
            all_paths = []

            # Construct solutions for each ant
            ant_paths = []
            for a_id, ant in enumerate(ant_agents):
                visited = [start]
                unvis = set(range(num_cities))
                unvis.remove(start)
                while unvis:
                    cur = visited[-1]
                    candidates = list(unvis)
                    weights = []
                    for nxt in candidates:
                        tau = pheromones[cur,nxt]**alpha
                        eta = (1/dists[cur,nxt])**beta if dists[cur,nxt]!=0 else 1e9
                        weights.append(tau*eta)
                    weights = np.array(weights)
                    weights /= weights.sum()
                    nxt_city = np.random.choice(candidates, p=weights)
                    visited.append(nxt_city)
                    unvis.remove(nxt_city)
                visited.append(start)

                # Animate the ant along the visited path
                moves = []
                for i in range(len(visited)-1):
                    c1 = visited[i]
                    c2 = visited[i+1]
                    path_line = Line(coords[c1], coords[c2], stroke_opacity=0)
                    # MoveAlongPath animation
                    moves.append(MoveAlongPath(ant, path_line, run_time=0.5))
                self.play(*moves, run_time=0.5*len(moves))

                length = sum(dists[visited[i], visited[i+1]] for i in range(len(visited)-1))
                all_paths.append((visited,length))
                ant_paths.append((visited,length))

            # Update pheromones
            pheromones = (1-evaporation_rate)*pheromones
            for visited,length in all_paths:
                deposit = Q/length
                for i in range(len(visited)-1):
                    a = visited[i]
                    b = visited[i+1]
                    pheromones[a,b] += deposit
                    pheromones[b,a] += deposit

            refresh_edges()

            # Check for best path
            for visited,length in all_paths:
                if length < best_aco_length:
                    best_aco_length = length
                    best_aco_path = visited[:]

            # Update ACO length text
            new_text = Text(f"ACO Best Length: {best_aco_length:.2f}", font_size=24, color=GREEN).to_edge(DOWN)
            self.play(Transform(aco_length_text, new_text))
            self.wait(0.5)

            # Move ants back to start for next iteration
            for ant in ant_agents:
                ant.move_to(coords[start])

        self.wait(1)

        # Final comparison:
        # Show both NN and ACO best lengths together at bottom
        final_compare_text = Text(
            f"NN Length: {nn_length:.2f}   vs   ACO Length: {best_aco_length:.2f}",
            font_size=24, color=WHITE
        ).to_edge(DOWN)
        self.play(ReplacementTransform(aco_length_text, final_compare_text))
        self.wait(2)

        # Show best ACO path clearly
        # Highlight best ACO path on top
        best_aco_lines = VGroup()
        if best_aco_path is not None:
            for i in range(len(best_aco_path)-1):
                a = best_aco_path[i]
                b = best_aco_path[i+1]
                l = Line(coords[a], coords[b], color=GREEN, stroke_width=5)
                best_aco_lines.add(l)
            self.play(Create(best_aco_lines))
            self.wait(2)

        # End
        self.play(*[FadeOut(m) for m in self.mobjects])
        self.wait(1)
