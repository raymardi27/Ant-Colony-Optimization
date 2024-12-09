from manim import *
import numpy as np

class StochasticGradientDescentScene(Scene):
    def construct(self):
        # Title
        title = Text("Stochastic Gradient Descent (SGD)").scale(0.8)
        # self.play(Write(title))
        # self.wait(1)
        # self.play(FadeOut(title))
        
        # Axes
        axes = Axes(
            x_range=[-4, 4, 1],
            y_range=[0, 20, 5],
            axis_config={"include_numbers": True},
        )
        axes_labels = axes.get_axis_labels(x_label="w", y_label="Loss")
        self.play(Create(axes), Write(axes_labels))
        self.wait(1)
        
        # Loss Function: f(w) = w^2
        loss_graph = axes.plot(lambda w: w**3 + w**2 + 3, color=BLUE)
        loss_label = axes.get_graph_label(loss_graph, label="f(w) = w^2", x_val=3.5, direction=UP)
        self.play(Create(loss_graph), Write(loss_label))
        self.wait(1)
        
        # SGD Optimization Steps
        # Initialize weight
        w = 2.0  # Starting point
        dot = Dot(axes.c2p(w, w**3 + w**2 + 3), color=RED)
        self.play(FadeIn(dot))
        self.wait(0.5)
        
        # Learning rate
        lr = 0.2
        
        # Number of iterations
        iterations = 50
        
        for i in range(iterations):
            # Compute gradient: f'(w) = 2w
            grad = 2 * w
            # Update rule: w_new = w - lr * grad
            w_new = w - lr * grad
            # Ensure w_new is within the axes range
            w_new = max(min(w_new, 4), -4)
            # Move dot to new position
            new_point = axes.c2p(w_new, w_new**3 + w_new**2 + 3)
            line = Line(dot.get_center(), new_point, color=GREEN)
            self.play(
                Transform(dot, Dot(new_point, color=RED)),
                GrowFromCenter(line)
            )
            self.wait(0.5)
            w = w_new
        
        # Highlight Minimum
        min_dot = Dot(axes.c2p(0, 0), color=YELLOW)
        self.play(FadeIn(min_dot))
        self.wait(1)
        
        # Draw vertical line from minimum to loss graph
        vert_line = DashedLine(
            start=axes.c2p(0, 0),
            end=axes.c2p(0, 0),
            color=YELLOW
        )
        self.play(Create(vert_line))
        self.wait(1)
        
        # Fade Out
        self.play(FadeOut(axes), FadeOut(loss_graph), FadeOut(loss_label),
                  FadeOut(dot), FadeOut(min_dot), FadeOut(vert_line))
