import random
import numpy as np
import matplotlib.pyplot as plt
from connector_generation import new_connector

def scale_and_position_connector(
        connector_points: np.ndarray,
        edge_start: np.ndarray,
        edge_end: np.ndarray,
        min_scale=0.5,
        max_scale=1.5,
        random_flip: bool = True) -> np.ndarray:
    """
    Scales and repositions the connector to fit the given edge while respecting scaling limits.

    Parameters:
    - connector_points: List of (x, y) tuples representing the connector.
    - edge_start: Tuple (x_a, y_a) representing the start of the edge.
    - edge_end: Tuple (x_b, y_b) representing the end of the edge.
    - min_scale: Minimum allowable scaling factor for the connector.
    - max_scale: Maximum allowable scaling factor for the connector.

    Returns:
    - List of (x, y) tuples representing the scaled and positioned connector.
    """
    x_start, y_start = edge_start
    x_end, y_end = edge_end
    
    # Calculate edge length
    edge_length: float = np.linalg.norm(edge_end - edge_start)
    
    if edge_length < min_scale: # Edge is too short to fit the connector
        return np.array([edge_start, edge_end]).T

    if random_flip and random.randint(0, 1):
        # Flip y-values
        connector_points[:, 1] *= -1
    # Calculate random scaling factor within bounds
    scale_factor: float = random.uniform(min_scale, min(max_scale, edge_length))
    
    # select random position on the edge
    connector_offset: float = random.uniform(0, edge_length - scale_factor)

    # 1. scale connector
    repositioned_points: np.ndarray = connector_points * scale_factor
    # 2. shift connector by offset
    repositioned_points += np.array([[connector_offset, 0]])
    # 3. rotate connector
    angle = np.arctan2(y_end - y_start, x_end - x_start)
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    repositioned_points = rotation_matrix @ (repositioned_points.T)
    # 4. shift connector to edge
    repositioned_points += edge_start.reshape(2, 1)
    # 5. add edge start and end
    repositioned_points = np.vstack(
        [edge_start,
         repositioned_points.T,
         edge_end]
    ).T

    return repositioned_points

def plot_edge_and_connector(
        edge_start,
        edge_end,
        connector_points,
        show_plot: bool = True,
        **kwargs):
    """
    Plots the edge and the repositioned connector using matplotlib.

    Parameters:
    - edge_start: Tuple (x_a, y_a) representing the start of the edge.
    - edge_end: Tuple (x_b, y_b) representing the end of the edge.
    - connector_points: List of (x, y) tuples representing the scaled and positioned connector.
    """

    # Plot the edge
    # x_edge = [edge_start[0], edge_end[0]]
    # y_edge = [edge_start[1], edge_end[1]]
    # plt.plot(x_edge, y_edge, 'k--', label='Edge')

    # Plot the connector
    x_connector, y_connector = connector_points[0, :], connector_points[1, :]
    plt.plot(
        x_connector,
        y_connector,
        label='Connector',
        **kwargs)
    
    # plt.title("Connector Positioned on Edge")
    # plt.xlabel("X-axis")
    # plt.ylabel("Y-axis")
    # plt.grid(True)
    plt.axis('equal')
    if show_plot:
        plt.legend()
        plt.show()
    
def draw_connector(
        edge_start: np.ndarray = np.array([0, 0]),
        edge_end: np.ndarray = np.array([3, 2]),
        min_scale: float = 0.5,
        max_scale: float = 1.5,
        show_plot: bool = True,
        **kwargs):
    
    connector_points: np.ndarray = new_connector()
    # Position connector on edge
    repositioned_connector: np.ndarray = scale_and_position_connector(
        connector_points,
        edge_start,
        edge_end,
        min_scale=min_scale,
        max_scale=max_scale)
    # Plot result
    plot_edge_and_connector(
        edge_start,
        edge_end,
        repositioned_connector,
        show_plot=False,
        **kwargs)
    if show_plot:
        plt.legend()
        plt.show()
    
if __name__ == "__main__":
    hexagon: np.ndarray = np.array([
        (-1, 0),
        (-0.5, 0.866),
        (0.5, 0.866),
        (1, 0),
        (0.5, -0.866),
        (-0.5, -0.866)
    ])
    pentagon: np.ndarray = np.array([
        (0, 1),
        (-0.951, 0.309),
        (-0.588, -0.809),
        (0.588, -0.809),
        (0.951, 0.309)
    ])
    square = np.array([
        (-1, -1),
        (1, -1),
        (1, 1),
        (-1, 1)
    ])
    triangle = np.array([
        (-1, 0),
        (1, 0),
        (0, 1.7)
    ])
    base_poly = hexagon
    n_pieces = 30
    for x in range(int(np.ceil(n_pieces**.5))):
        for y in range(int(np.floor(n_pieces**.5))):
            piece_offset = np.array([x, y])*5
            polygon: np.ndarray = base_poly + piece_offset
        # polygon: np.ndarray = hexagon
            for p1, p2 in zip(polygon, np.roll(polygon, -1, axis=0)):
                draw_connector(
                    edge_start=np.array(p1),
                    edge_end=np.array(p2),
                    min_scale=0.7,
                    max_scale=1.5,
                    show_plot=False,
                    color="#000")
    # test_connector_placement()
    # plt.legend()
    plt.tight_layout()
    plt.show()