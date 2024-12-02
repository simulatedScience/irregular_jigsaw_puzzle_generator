"""
Given a puzzle as many Voronoi cells, reduce it to a set of edges.
"""
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

from VoronoiCell import VoronoiCell

def reduce_to_edges(
        cells: list[VoronoiCell]
    ) -> np.ndarray:
    """
    Given a puzzle as a list of VoronoiCell objects, return a list of all edges in the polygon cells.

    Args:
        cells (list[VoronoiCell]): List of VoronoiCell objects representing the puzzle.

    Returns:
        np.ndarray: array of n edges (shape: (n, 2)).
    """
    puzzle_edges: set[tuple[tuple[float, float], tuple[float, float]]] = set()
    for cell in cells:
        polygon: Polygon = cell.polygon
        corners = np.array(polygon.exterior.coords)
        edges = np.hstack([corners, np.roll(corners, -1, axis=0)])
        for edge in edges:
            # sort edge points to have consistent representation
            sorted_edge = tuple(
                sorted([
                    tuple(edge[:2]),
                    tuple(edge[2:])
                ])
            )
            
            puzzle_edges.add(sorted_edge)
    # convert puzzle edges to array of shape (n, 4), each row as (x1, y1, x2, y2)
    return np.array(list(puzzle_edges))


def on_puzzle_edge(
    edge_start: np.ndarray,
    edge_end: np.ndarray,
    puzzle_bbox: tuple[float, float, float, float],
    ) -> bool:
    """
    Check if a given edge (both points) is on the boundary of the puzzle.
    
    Args:
        edge_start (np.ndarray): Start of edge (x, y).
        edge_end (np.ndarray): End of edge (x, y).
        puzzle_bbox (tuple[float, float, float, float]): Bounding box of the puzzle (x_min, y_min, x_max, y_max).
    """
    for point in [edge_start, edge_end]:
        if (puzzle_bbox[0] < point[0] < puzzle_bbox[2] and \
                puzzle_bbox[1] < point[1] < puzzle_bbox[3]):
            # point is inside the puzzle
            return False
    return True

def plot_puzzle_edges(edges, color="#000", linewidth=1):
    """
    Plots all the edges from a list of edges using matplotlib.

    Parameters:
    - edges: List of tuples [(x1, y1, x2, y2), ...] representing the edges.
    - color: Color of the edges to be plotted (default is blue).
    - linewidth: Width of the lines representing the edges.
    """
    plt.figure(figsize=(8, 8))

    for (x1, y1), (x2, y2) in edges:
        plt.plot([x1, x2], [y1, y2], color=color, linewidth=linewidth)

    plt.title("Custom irregular Jigsaw Puzzle")
    plt.axis('scaled')
    plt.show()
