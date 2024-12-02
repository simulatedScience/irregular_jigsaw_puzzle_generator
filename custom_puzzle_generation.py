"""
This module provides tools to generate custom jigsaw puzzles with irregular pieces using Voronoi diagrams.

Call generate_puzzle() to generate a new puzzle layout.

Author: Sebastian Jost using GPT-4o (13.10.2024)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Polygon, Point, MultiPolygon
from shapely.ops import unary_union
from collections import defaultdict

from VoronoiCell import VoronoiCell, plot_polygons, debug_plot
from reduce_to_edges import reduce_to_edges, plot_puzzle_edges, on_puzzle_edge
from connector_placement import draw_connector

DEBUG = False

FAR_POINT_DISTANCE = 10000

# --- Utility Functions ---

def generate_random_points(grid_size: tuple[int, int], num_points: int) -> np.ndarray:
    """
    Generate random points within each cell of the grid with padding.
    
    Args:
        grid_size (tuple[int, int]): Dimensions of the grid in (rows, columns).
        num_points (int): Number of random points to generate.

    Returns:
        np.ndarray: Array of random points of shape (num_points, 2).
    """
    rows, cols = grid_size
    points = []
    padding = 0.1

    for i in range(rows):
        for j in range(cols):
            for _ in range(num_points):
                x = i + padding + np.random.random() * (1 - 2 * padding)
                y = j + padding + np.random.random() * (1 - 2 * padding)
                points.append([x, y])
    
    # add extreme points along major axes to ensure all relevant cells are closed
    points += [(FAR_POINT_DISTANCE, 0),
               (0, FAR_POINT_DISTANCE),
               (-FAR_POINT_DISTANCE, 0),
               (0, -FAR_POINT_DISTANCE)]

    return np.array(points)

def scale_to_bounds(points: np.ndarray, grid_size: tuple[int, int], width: float, height: float) -> np.ndarray:
    """
    Scale the points to fit within the given width and height.

    Args:
        points (np.ndarray): Array of points of shape (num_points, 2).
        grid_size (tuple[int, int]): Dimensions of the grid in (rows, columns).
        width (float): Width of the puzzle.
        height (float): Height of the puzzle.

    Returns:
        np.ndarray: Array of scaled points of shape (num_points, 2).
    """
    min_x, min_y = 0, 0
    max_x, max_y = grid_size
    scale_x = width / (max_x - min_x)
    scale_y = height / (max_y - min_y)
    points[:, 0] = (points[:, 0] - min_x) * scale_x
    points[:, 1] = (points[:, 1] - min_y) * scale_y
    return points

def generate_voronoi(points: np.ndarray) -> Voronoi:
    """
    Generate the Voronoi diagram from a set of points.

    Args:
        points (np.ndarray): Array of points of shape (num_points, 2).

    Returns:
        Voronoi: Voronoi diagram generated from the points.
    """
    return Voronoi(points)

def clip_voronoi(vor: Voronoi, width: float, height: float) -> list[Polygon]:
    """
    Clip the Voronoi diagram to fit within a bounding box of given width and height.

    Args:
        vor (Voronoi): Voronoi diagram.
        width (float): Width of the puzzle.
        height (float): Height of the puzzle.

    Returns:
        list[Polygon]: List of polygons representing the clipped Voronoi cells.
    """
    bounding_box = Polygon([(0, 0), (0, height), (width, height), (width, 0)])
    polygons = []
    
    for region_index in vor.regions:
        if not region_index or -1 in region_index:  # Ignore open regions
            continue
        region = [vor.vertices[i] for i in region_index]
        poly = Polygon(region)
        clipped_poly = poly.intersection(bounding_box)
        if clipped_poly.is_valid and not clipped_poly.is_empty:
            polygons.append(clipped_poly)

    return polygons

def reduce_to_target_count(cells: dict[int, VoronoiCell], target_count: int, max_index: int) -> tuple[list[VoronoiCell], int]:
    """
    Reduce the number of Voronoi cells to the target count by merging cells.

    Args:
        cells (list[VoronoiCell]): List of Voronoi cells.
        target_count (int): Target number of pieces.
        max_index (int): Current maximum index.

    Returns:
        Tuple: Updated list of cells and the new maximum index.
    """
    merged_cells: dict[int, VoronoiCell] = cells.copy()
    removed_ids: set[int] = set()

    while len(merged_cells) > target_count:
        # Find the cell with the smallest area
        smallest_cell_id = min(merged_cells.keys(), key=lambda cell_id: merged_cells[cell_id].polygon.area)

        # Get the smallest neighbor of the smallest cell
        # try:
        smallest_neighbor_id = min(merged_cells[smallest_cell_id].neighbors, key=lambda cell_id: merged_cells[cell_id].polygon.area)
        smallest_neighbor = merged_cells[smallest_neighbor_id]
        # except Exception as exception:
        #     mycell = merged_cells[smallest_cell_id]
        #     print(f"Cell {smallest_cell_id} has {len(mycell.neighbors)} neighbors.")
        #     print(f"There are {len(merged_cells)} cells in total.")
        #     plot_polygons(list(merged_cells.values()))
        #     raise exception

        # Merge the two cells
        new_cell = merge_cells(merged_cells[smallest_cell_id], smallest_neighbor, max_index, merged_cells)
        max_index += 1
        merged_cells[new_cell.id] = new_cell
        
        # Update neighbors
        update_neighbors(merged_cells, smallest_cell_id, smallest_neighbor.id, new_cell.id)
        
        # Remove the merged cells
        if not smallest_cell_id in removed_ids:
            removed_ids.add(smallest_cell_id)
            del merged_cells[smallest_cell_id]
        else:
            print(f"Cell {smallest_cell_id} already removed.")
        if not smallest_neighbor.id in removed_ids:
            removed_ids.add(smallest_neighbor.id)
            del merged_cells[smallest_neighbor.id]
        else:
            print(f"Cell {smallest_neighbor.id} already removed.")

    print(f"Removed {len(removed_ids)} small cells.")

    return merged_cells, max_index

def get_smallest_neighbor(poly: Polygon, polygons: list[Polygon]) -> Polygon:
    """
    Get the smallest neighboring polygon for a given polygon.
    
    Args:
        poly (Polygon): The polygon to find the neighbor for.
        polygons (list[Polygon]): List of available polygons to choose from.

    Returns:
        Polygon: The smallest neighboring polygon.
    """
    neighbors = [p for p in polygons if p != poly and poly.touches(p)]
    if neighbors:
        return min(neighbors, key=lambda p: p.area)
    return None

def extract_voronoi_cells(vor: Voronoi, width: float, height: float) -> tuple[list[VoronoiCell], dict[int, set]]:
    """
    Extract the Voronoi cells and their neighbor relationships from the Voronoi diagram.

    Args:
        vor (Voronoi): The Voronoi diagram.
        width (float): Puzzle width.
        height (float): Puzzle height.

    Returns:
        Tuple: List of VoronoiCell objects and a dictionary of neighbor relations.
    """
    bounding_box = Polygon([(0, 0), (0, height), (width, height), (width, 0)])
    cells: dict[int, VoronoiCell] = {}
    neighbors_dict = defaultdict(set)

    ignored_points: set[int] = set()

    for point_idx, region_idx in enumerate(vor.point_region):
        region: list[int] = vor.regions[region_idx]
        if not region or -1 in region: # Ignore open regions
            ignored_points.add(point_idx)
            continue
        region_points: list[np.ndarray] = [vor.vertices[i] for i in region]
        poly: Polygon = Polygon(region_points).intersection(bounding_box)
        if poly.is_valid and not poly.is_empty:
            cell: VoronoiCell = VoronoiCell(poly, point_idx)
            cells[cell.id] = cell
    # calculate neighbors from vor.ridge_points
    for (p1, p2) in vor.ridge_points:
        if p1 not in ignored_points and p2 not in ignored_points:
            # check if polygons are still adjacent after clipping
            if not cells[p1].polygon.touches(cells[p2].polygon):
                continue
            # add neighbor to both cells
            cells[p1].neighbors.add(int(p2))
            cells[p2].neighbors.add(int(p1))

    for cell_id, neighbors in neighbors_dict.items():
        cells[cell_id].set_neighbors(neighbors)
    
    # # plot voronoi diagram: numbered points and lines between ridge_points
    # fig, ax = plt.subplots()
    # ridges_x = []
    # ridges_y = []
    # for (p1, p2) in vor.ridge_points:
    #     if p1 not in ignored_points and p2 not in ignored_points:
    #         x1, y1 = vor.points[p1]
    #         x2, y2 = vor.points[p2]
    #         ridges_x.append([x1, x2])
    #         ridges_y.append([y1, y2])
    #         # ax.plot([x1, x2], [y1, y2], 'r-', zorder=-1, alpha=0.2)
    # ridges_x = list(zip(*ridges_x))
    # ridges_y = list(zip(*ridges_y))
    # ax.plot(ridges_x, ridges_y, 'k-', zorder=-1, alpha=0.2)
    # # ridges_x = []
    # # ridges_y = []
    # # for (p1, p2) in vor.ridge_vertices:
    # #     if p1 != -1 and p2 != -1:
    # #         x1, y1 = vor.vertices[p1]
    # #         x2, y2 = vor.vertices[p2]
    # #         ridges_x.append([x1, x2])
    # #         ridges_y.append([y1, y2])
    # #         # ax.plot([x1, x2], [y1, y2], 'r-', zorder=-1, alpha=0.2)
    # # ridges_x = list(zip(*ridges_x))
    # # ridges_y = list(zip(*ridges_y))
    # # ax.plot(ridges_x, ridges_y, 'r-', zorder=-1, alpha=0.2)
    # for i, point in enumerate(vor.points):
    #     ax.text(point[0], point[1], str(i), fontsize=8, ha='center', va='center')
    # plt.legend()
    # plt.show()
    return cells

def merge_polygons(poly1: Polygon, poly2: Polygon) -> Polygon:
    """
    Merge two polygons and return a unified polygon.

    Args:
        poly1 (Polygon): First polygon.
        poly2 (Polygon): Second polygon.

    Returns:
        Polygon: The merged polygon.
    """
    merged = poly1.union(poly2)
    if isinstance(merged, MultiPolygon):
        merged = unary_union(merged)
    return merged

def merge_cells(cell1: VoronoiCell, cell2: VoronoiCell, max_index: int, all_cells: dict[int, VoronoiCell]) -> VoronoiCell:
    """
    Merge two Voronoi cells into a new cell with a unique ID.

    Args:
        cell1 (VoronoiCell): First cell to merge.
        cell2 (VoronoiCell): Second cell to merge.
        max_index (int): Current maximum cell index.

    Returns:
        VoronoiCell: New merged cell.
    """
        
    merged_polygon = merge_polygons(cell1.polygon, cell2.polygon)
    new_cell = VoronoiCell(merged_polygon, max_index + 1)
    new_neighbors = (cell1.neighbors | cell2.neighbors) - {cell1.id, cell2.id}

    if isinstance(merged_polygon, MultiPolygon):
        print(f"Incorrect merging of cells {cell1.id} and {cell2.id} resulted in MultiPolygon {new_cell.id}.")
        print(f"Neighbors of {cell1.id}: {cell1.neighbors}")
        print(f"Neighbors of {cell2.id}: {cell2.neighbors}")
        print(f"New neighbors: {new_neighbors}")
        plot_polygons(list(all_cells.values()))

    new_cell.set_neighbors(new_neighbors)
    if len(new_neighbors) == 0:
        print(f"New cell {new_cell.id} has no neighbors.")
        plot_polygons(list(all_cells.values()))

    return new_cell

def update_neighbors(all_cells: dict[int, VoronoiCell], cell1_id: int, cell2_id: int, new_cell_id: int):
    """
    Update the neighbors of two merged cells to reflect the new merged cell.

    Args:
        all_cells (dict[int, VoronoiCell]): Dictionary of all Voronoi cells with unique IDs as keys.
        cell1_id (int): ID of the first merged cell.
        cell2_id (int): ID of the second merged cell.
        new_cell_id (int): ID of the new merged cell.
    """
    # Update neighbor lists to reflect the new cell
    # print(f"Replacing neighbors {cell1_id} and {cell2_id} with {new_cell_id}.")
    for neighbor_id in all_cells[new_cell_id].neighbors:
        new_neighbors = all_cells[neighbor_id].neighbors.copy()
        for nid in all_cells[neighbor_id].neighbors:
            if nid == cell1_id or nid == cell2_id:
                new_neighbors.remove(nid)
                new_neighbors.add(new_cell_id)
        all_cells[neighbor_id].set_neighbors(new_neighbors)
        # print(f"new_neighbors of {neighbor_id}: {new_neighbors}")

def merge_small_pieces(cells: dict[int, VoronoiCell], min_area: float, max_index: int) -> tuple[list[VoronoiCell], int]:
    """
    Merge small Voronoi cells with their neighbors to ensure a minimum surface area.

    Args:
        cells (dict[int, VoronoiCell]): Dictionary of Voronoi cells with unique IDs as keys.
        min_area (float): Minimum allowed area for a cell.
        max_index (int): Current maximum index.

    Returns:
        Tuple: Updated list of cells and the new maximum index.
    """
    merged_cells: dict[int, VoronoiCell] = {int(cell.id): cell for cell in cells.values()}
    removed_ids = set()
    for cell_id, cell in cells.items():
        if not cell_id in merged_cells:
            continue
        if cell.polygon.area < min_area:
            # Get the smallest neighbor
            smallest_neighbor_id = min(cell.neighbors, key=lambda id: merged_cells[id].polygon.area)
            smallest_neighbor = merged_cells[smallest_neighbor_id]
            
            # Merge the two cells
            new_cell = merge_cells(cell, smallest_neighbor, max_index, merged_cells)
            max_index += 1
            merged_cells[new_cell.id] = new_cell
            update_neighbors(merged_cells, cell.id, smallest_neighbor.id, new_cell.id)

            # Remove the merged cells
            if not cell.id in removed_ids:
                removed_ids.add(cell.id)
                del merged_cells[cell.id]
            else:
                print(f"Cell {cell.id} already removed.")
            if not smallest_neighbor.id in removed_ids:
                removed_ids.add(smallest_neighbor.id)
                del merged_cells[smallest_neighbor.id]
            else:
                print(f"Cell {smallest_neighbor.id} already removed.")

    print(f"Removed {len(removed_ids)} small cells.")

    return merged_cells, max_index

def refine_voronoi(cells: dict[int, VoronoiCell], refinement_steps: int, width: float, height: float) -> list[Polygon]:
    """
    Refine Voronoi cells by generating random points within them and recomputing Voronoi.

    Args:
        cells (dict[int, VoronoiCell]): Dictionary of Voronoi cells with unique IDs as keys.
        refinement_steps (int): Number of refinement steps.
        width (float): Width of the puzzle.
        height (float): Height of the puzzle.

    Returns:
        list[Polygon]: Refined list of polygons after subdivision.
    """
    for _ in range(refinement_steps):
        new_points = []
        for cell in cells.values():
            if cell.polygon.area > 0:
                while True:
                    centroid = cell.polygon.centroid
                    random_offset = np.random.normal(scale=0.2, size=2)
                    new_point = Point(centroid.x + random_offset[0], centroid.y + random_offset[1])
                    if cell.polygon.contains(new_point):
                        new_points.append((new_point.x, new_point.y))
                        break
        if new_points:
            new_points += [(FAR_POINT_DISTANCE, 0),
                    (0, FAR_POINT_DISTANCE),
                    (-FAR_POINT_DISTANCE, 0),
                    (0, -FAR_POINT_DISTANCE)]
            vor = generate_voronoi(np.array(new_points))
            cells = extract_voronoi_cells(vor, width, height)
    
    return cells

# --- Plotting Functions ---


# --- Main Function ---

def generate_puzzle(
        grid_size: tuple[int, int],
        num_points: int,
        width: float,
        height: float, 
        refinement_steps: int = 0,
        min_area: float = None, 
        max_aspect_ratio: float = None,
        target_count: int = None) -> list[VoronoiCell]:
    """
    Generate a custom puzzle layout with Voronoi-based irregular pieces.

    Args:
        grid_size (tuple[int, int]): Dimensions of the grid in (rows, columns).
        num_points (int): Number of random points per grid cell.
        width (float): Width of the puzzle.
        height (float): Height of the puzzle.
        refinement_steps (int): Number of refinement steps to increase irregularity.
        min_area (float, optional): Minimum area of each piece.
        max_aspect_ratio (float, optional): Maximum aspect ratio for pieces.
        target_count (int, optional): Target number of pieces in the puzzle.

    Returns:
        list[Polygon]: List of polygons representing the final puzzle pieces.
    """
    points = generate_random_points(grid_size, num_points)
    points = scale_to_bounds(points, grid_size, width, height)
    vor = generate_voronoi(points)
    
    # Step 2: Clip the Voronoi cells to the puzzle bounding box
    cells: dict[int, VoronoiCell] = extract_voronoi_cells(vor, width, height)
    max_index = max(cells.keys()) # Get the maximum index of the cells



    # Step 3: Perform refinement steps if specified
    if refinement_steps > 0:
        # cells = refine_voronoi(cells, refinement_steps, width, height)
        for step in range(refinement_steps):
            cells = refine_voronoi(cells, refinement_steps=1, width=width, height=height)
            if DEBUG:
                debug_plot(cells, points)


    # Step 4: Optionally reduce to target piece count
    if target_count is not None and len(cells) > target_count:
        cells, max_index = reduce_to_target_count(cells, target_count, max_index)
        print(f"Reduced to {len(cells)} cells.")
    # Step 5: Optionally merge small pieces with a minimum area constraint
    if min_area is not None:
        cells, max_index = merge_small_pieces(cells, min_area, max_index)
        print(f"Merged small pieces. Now {len(cells)} cells.")

    return list(cells.values())

# Example usage:
if __name__ == "__main__":
    seed = np.random.randint(0, 1000)
    # seed = 205
    np.random.seed(seed)
    print(f"Random seed: {seed}")
    ##### settings for large puzzle:
    grid_size = (35, 25)
    num_points_per_cell = 1
    width, height = 70.0, 50.0
    refinement_steps = 20
    min_area = .5
    max_aspect_ratio = 2.0
    target_count = 500
    ##### test settings for small puzzle:
    # grid_size = (15, 10)
    # num_points_per_cell = 1
    # width, height = 32.0, 20.0
    # refinement_steps = 20
    # min_area = .5
    # max_aspect_ratio = 2.0
    # target_count = 100

    puzzle: list[VoronoiCell] = generate_puzzle(
        grid_size,
        num_points_per_cell,
        width,
        height,
        refinement_steps,
        min_area,
        max_aspect_ratio,
        target_count)
    # plot_polygons(puzzle, show_ids=False)
    # puzzle edges as rows ((x1, y1), (x2, y2))
    puzzle_edges = reduce_to_edges(puzzle)
    # plot_puzzle_edges(puzzle_edges)
    # plt.show()
    for edge in puzzle_edges:
        if on_puzzle_edge(edge[0, :], edge[1, :], (0, 0, width, height)):
            # plot straight edge
            plt.plot(
                edge[:, 0],
                edge[:, 1],
                color="#000",
                linewidth=0.5,
            )
            continue
        draw_connector(
            edge[0, :],
            edge[1, :],
            show_plot=False,
            min_scale=0.7,
            max_scale=1.1,
            color="#000",
            linewidth=0.5,
        )
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.show()
