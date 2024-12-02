import numpy as np
import matplotlib.pyplot as plt

from shapely.geometry import Polygon, Point, MultiPolygon
from shapely.ops import unary_union

class VoronoiCell:
    """
    Helper class to save Voronoi cell polygons with neighbors and a unique ID.
    """
    def __init__(self, polygon: Polygon, cell_id: int):
        self.polygon: Polygon = polygon
        self.id: int = cell_id
        self.neighbors: set[int] = set()
    
    def __str__(self):
        return f"Cell {self.id} with {len(self.neighbors)} neighbors."

    def set_neighbors(self, neighbors: set[int]):
        """
        Set the neighbors of the cell.

        Args:
            neighbors (set[int]): Set of neighboring cell IDs.
        """
        self.neighbors: set[int] = {int(n) for n in neighbors}

    def plot(self, all_polygons: dict[int, "VoronoiCell"]):
        """
        Plot self and neighbors using matplotlib.
        """
        plot_polygons([self] + [all_polygons[n] for n in self.neighbors])

# --- Debug Plotting Function ---

def debug_plot(polygons: list[Polygon], points: np.ndarray = None):
    """
    Plot the polygons and points for debugging purposes.

    Args:
        polygons (list[Polygon]): List of polygons to plot.
        points (np.ndarray, optional): Points to plot (if any).
    """
    fig, ax = plt.subplots()
    for poly in polygons:
        x, y = poly.exterior.xy
        ax.fill(x, y, alpha=0.5, fc='b', ec='black')
    
    if points is not None:
        ax.scatter(points[:, 0], points[:, 1], color='red', zorder=5)
    
    ax.set_aspect('equal')
    plt.show()


def plot_polygons(polygons: list[VoronoiCell], show_ids: bool = True):
    """
    Plot a list of polygons using matplotlib.

    Args:
        polygons (list[Polygon]): List of polygons to plot.
    """
    fig, ax = plt.subplots()
    for cell in polygons:
        if isinstance(cell.polygon, MultiPolygon):
            # Use unary_union to merge MultiPolygon into a single Polygon
            cell.polygon = unary_union(cell.polygon)
        if isinstance(cell.polygon, MultiPolygon):
            # draw all subpolygons individually in red
            for poly in cell.polygon.geoms:
                x, y = poly.exterior.xy
                ax.fill(x, y, alpha=0.5, fc='#dd0000', ec='black')
                x, y = poly.centroid.xy
                ax.text(x[0], y[0], str(cell.id), fontsize=8, ha='center', va='center')
        else:
            x, y = cell.polygon.exterior.xy
            ax.fill(x, y, alpha=0.5, fc='#5588ff', ec='black')
            # draw id on the center of the cell
            x, y = cell.polygon.centroid.xy
            if show_ids:
                ax.text(x[0], y[0], str(cell.id), fontsize=8, ha='center', va='center')
    ax.set_aspect('equal')
    plt.subplots_adjust(left=0.03, right=0.99, top=0.99, bottom=0.02)
    
    plt.show()
