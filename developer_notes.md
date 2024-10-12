As of 13.10.2024, we store the puzzle layout as a dict of `VoronoiCell` objects. These each include a unique ID, a set of indices of neighbouring cells and a `shapely.Polygon` object storing the geometry of the piece. The keys in this dict are the IDs.

When generating the Voronoi diagram, we always add four far away points in the four major directions. This ensures that all relevant voronoi cells are finite, so we can properly define polygons. These are then intersected with the bounding box of the puzzle to achieve a rectangular shape of the puzzle.

Currently, `reduce_to_target_count` merges the smallest piece (by area) with the smallest neighbor. `merge_small_pieces` merges pieces with area below a threshold with the smallest neighbour too. This can create pieces with very thin sections. To avoid this, we should instead find the neibor sharing the longest edge and merge with that one.

`refine_voronoi` currently only outputs convex pieces as only one point is generated in each cell before a new Voronoi diagram is calculated. This helps to obscure the initial grid structure used for generation, but doesn't produce very different shapes.

The approach using Voronoi diagrams may not be well-suited to generate rounded shapes. To achieve this, the edge connectors could include circular arcs of various angles.

The code was mostly generated with ChatGPT (GPT-4o) based on my own ideas. Several major modifications were required to get the code to work, which I implemented myself.
