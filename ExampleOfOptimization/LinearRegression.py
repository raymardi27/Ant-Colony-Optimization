from manim import *
import numpy as np

class LinearRegressionScene(Scene):
    def construct(self):
        # Title
        title = Text("Linear Regression Model").scale(0.8)
        # self.play(Write(title))
        # self.wait(1)
        # self.play(FadeOut(title))
        
        # Axes
        axes = Axes(
            x_range=[0, 10, 1],
            y_range=[0, 25, 2],
            axis_config={"include_numbers": True},
        )
        axes_labels = axes.get_axis_labels(x_label="X", y_label="Y")
        self.play(Create(axes), Write(axes_labels))
        self.wait(1)
        
        # Data Points
        np.random.seed(42)  # For reproducibility
        data_x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        data_y = 2 * data_x + 1 + np.random.randn(len(data_x))  # y = 2x + 1 + noise
        data_points = VGroup(*[
            Dot(axes.c2p(x, y), color=BLUE)
            for x, y in zip(data_x, data_y)
        ])
        self.play(FadeIn(data_points))
        self.wait(1)
        
        # Initial Regression Line
        initial_slope = 0.0
        initial_intercept = 5.0
        reg_line = axes.plot(
            lambda x: initial_slope * x + initial_intercept,
            stroke_color=RED
        )
        reg_label = MathTex("y = 0x + 5").scale(0.7).next_to(axes, DOWN, buff=1)
        self.play(Create(reg_line), Write(reg_label))
        self.wait(1)
        
        # Wobbling Regression Lines
        # Simulate the fitting process with intermediate steps
        fitting_steps = 10
        target_slope = 2.0
        target_intercept = 1.0
        for step in range(1, fitting_steps + 1):
            # Calculate intermediate slope and intercept with slight random fluctuation
            slope_variation = np.random.uniform(-0.2, 0.2)
            intercept_variation = np.random.uniform(-0.2, 0.2)
            intermediate_slope = initial_slope + (target_slope - initial_slope) * (step / fitting_steps) + slope_variation
            intermediate_intercept = initial_intercept + (target_intercept - initial_intercept) * (step / fitting_steps) + intercept_variation
            
            # Create intermediate regression line
            intermediate_reg_line = axes.plot(
                lambda x: intermediate_slope * x + intermediate_intercept,
                stroke_color=YELLOW
            )
            intermediate_reg_label = MathTex(
                f"y = {intermediate_slope:.2f}x + {intermediate_intercept:.2f}"
            ).scale(0.7).next_to(axes, DOWN, buff=1)
            
            # Transform the current regression line to the intermediate line
            self.play(
                Transform(reg_line, intermediate_reg_line),
                Transform(reg_label, intermediate_reg_label),
                run_time=0.3
            )
            self.wait(0.1)
        
        # Final Regression Line
        final_reg_line = axes.plot(
            lambda x: target_slope * x + target_intercept,
            stroke_color=GREEN
        )
        final_reg_label = MathTex(f"y = {target_slope}x + {target_intercept}").scale(0.7).next_to(axes, DOWN, buff=1)
        self.play(
            Transform(reg_line, final_reg_line),
            Transform(reg_label, final_reg_label),
            run_time=0.5
        )
        self.wait(1)
        
        # Highlight Residuals
        residuals = VGroup()
        for dot, x in zip(data_points, data_x):
            y_actual = dot.get_center()[1]
            y_pred = target_slope * x + target_intercept
            line = Line(
                start=dot.get_center(),
                end=axes.c2p(x, y_pred),
                stroke_color=YELLOW
            )
            residuals.add(line)
        self.play(Create(residuals))
        self.wait(2)
        
        
        # Fade Out (Optional: Keep the final state for emphasis)
        self.play(FadeOut(axes), FadeOut(data_points), FadeOut(reg_line),
                  FadeOut(reg_label), FadeOut(residuals))
