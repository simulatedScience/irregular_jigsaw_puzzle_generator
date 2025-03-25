"""
Author: GPT-4o (02.12.2024)
"""

import random
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

def new_connector(max_offset: float = 0.03) -> np.ndarray:
    """
    Generates a new connector with a spline.
    
    Args:
        max_offset (float): maximum disturbance added to each base point.
    
    Returns:
        np.ndarray: array of connector points (shpaoe: (n, 2)).
    """
    # Step 1: Generate fixed pattern of points
    base_points = generate_fixed_pattern()
    # Step 2: Apply random offsets (optional for variation)
    offset_points = add_random_offsets(base_points, max_offset=max_offset)
    # Step 3: Create parametric splines for x(t) and y(t)
    cs_x, cs_y = create_parametric_spline(offset_points)
    # Step 4: Generate dense connector points
    connector_points = generate_connector_points(cs_x, cs_y)
    return np.array(connector_points)

def generate_fixed_pattern(pattern="standard"):
    """
    Generates a fixed pattern of evenly spaced points along the x-axis.
    The y-coordinates start and end at 0, forming a basic wave pattern.
    """
    if pattern.lower() == "standard":
        points = [
            (0., 0.),
            (.3, 0.),
            (.4, .15),
            (.3, .3),
            (.5, .4),
            (.7, .3),
            (.6, .15),
            (.7, 0.),
            (1., 0.)
        ]
    elif pattern.lower() == "swirl":
        points = [
            (0., 0.),
            (.2, -.1),
            (.50, -.3),
            (.70, 0.),
            (.50, .15),
            (.45, .05),
            (.50, 0.), # center point
            (.55, -.05),
            (.50, -.15),
            (.30, 0.),
            (.50, .3),
            # (.5, .3),
            (.8, .1),
            # (.8, 0.),
            # (.9, -.1),
            (1., 0.)
        ]
    elif pattern.lower() == "eagle":
        points = [
            (0., 0.),
            (.1, -.1),
            (.4, -.3),
            (.4, -.1), # start left wing
            (.3, -.05),
            (.2, .1),
            (.3, .1),
            (.4, .1), # head start
            (.5, .25), # center point
            (.6, .25),
            (.65, .2), # beak
            (.55, .15),
            (.55, .1), # head end
            (.7, .1),
            (.8, .1),
            (.7, -.05),
            (.6, -.1), # end right wing
            (.6, -.3),
            (.9, -.1),
            (1., 0.),
        ]
        if random.random() < 0.5: # randomly flip in x-direction
            points = [(1-x, y) for x, y in points]
            points.reverse()
    elif pattern.lower() == "dragon":
        points = [
            (0.0, 0.0),
            (.40, 0.0),
            (.35, .10),
            (.20, .15),
            # (.25, .20),
            (.30, .2),
            (.50, .10),
            (.50, .35),
            (.70, .30),
            (.90, .175),
            (.95, .10),
            (.90, .125),
            (.85, .05),
            (.75, .10),
            (.70, 0.0),
            (1.0, 0.0),
        ]
        if random.random() < 0.5:
            points = [(1-x, y) for x, y in points]
            points.reverse()
    elif not pattern or pattern.lower() in ("none", "random"):
        return generate_fixed_pattern(pattern=random.choice([
            "standard",
            "swirl",
            "eagle",
            "dragon",
        ]))
    return points

def add_random_offsets(points, max_offset=0.1):
    """
    Adds random vertical offsets to y-coordinates and horizontal offsets to x-coordinates.
    
    Parameters:
    - points: List of (x, y) tuples representing the original points.
    - max_offset: Maximum deviation for the coordinates.
    
    Returns:
    - List of (x, y) tuples with random offsets applied.
    """
    return [points[0]] + [(x + random.uniform(-max_offset, max_offset), 
             y + random.uniform(-max_offset, max_offset)) for x, y in points[1:-1]] + [points[-1]]

def create_parametric_spline(points):
    t = np.linspace(0, 1, len(points))  # Parametric variable t
    x, y = zip(*points)
    
    # Calculate slope for boundary condition: (f(b) - f(a)) / (b - a)
    slope_x = (x[-1] - x[0]) / (t[-1] - t[0])
    slope_y = (y[-1] - y[0]) / (t[-1] - t[0])
    
    # Create cubic splines with specified first derivatives at the boundaries
    cs_x = CubicSpline(t, x, bc_type=((1, slope_x), (1, slope_x)))
    cs_y = CubicSpline(t, y, bc_type=((1, slope_y), (1, slope_y)))
    
    return cs_x, cs_y

def generate_connector_points(cs_x, cs_y, num_points=100):
    """
    Generates a set of points along the parametric spline.
    
    Parameters:
    - cs_x, cs_y: CubicSpline objects representing x(t) and y(t).
    - num_points: Number of points to generate along the spline.
    
    Returns:
    - List of (x, y) tuples representing the smooth connector curve.
    """
    t_dense = np.linspace(0, 1, num_points)
    x_dense = cs_x(t_dense)
    y_dense = cs_y(t_dense)
    return list(zip(x_dense, y_dense))

def plot_connector(connector_points, show_plot: bool = True, linestyle="-", **kwargs):
    """
    Plots the given connector using matplotlib.
    
    Parameters:
    - connector_points: List of (x, y) tuples representing the connector points.
    """
    x, y = zip(*connector_points)
    plt.plot(x, y, linestyle=linestyle, **kwargs)
    plt.title("Jigsaw Puzzle Connector with Overhang")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.grid(True)
    plt.axis('equal')
    if show_plot:
        plt.show()

def generate_and_plot_connector(n=1, max_offset=0.03):
    """
    Generates a connector with a spline and plots it.
    This function is suitable for testing the entire pipeline.
    """
    # Step 1: Generate fixed pattern of points
    base_points = generate_fixed_pattern()
    plot_connector(base_points, show_plot=False, label="Base Pattern", alpha=0.7, linestyle="-", marker="o", markersize=5)
    for _ in range(n):
        # Step 2: Apply random offsets (optional for variation)
        offset_points = add_random_offsets(base_points, max_offset=max_offset)
        # Step 3: Create parametric splines for x(t) and y(t)
        cs_x, cs_y = create_parametric_spline(offset_points)
        # Step 4: Generate dense connector points
        connector_points = generate_connector_points(cs_x, cs_y)
        # Step 5: Plot the connector
        # plot_connector(offset_points, show_plot=False, color="#f80", linestyle="", marker="o", markersize=5)
        plot_connector(connector_points, show_plot=False, color="#f80")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    generate_and_plot_connector(max_offset=0.01, n=1)
