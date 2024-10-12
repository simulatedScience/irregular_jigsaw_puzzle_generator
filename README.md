# Custom Irregular Puzzle Generator
Generate unique custom jigsaw puzzle layouts of any size with irregularly shaped pieces.

## Motivation
Regular puzzles with many pieces can be very difficult to solve, especially if they have high piece counts and large areas of flat colors. Irregular pieces can help in these situations and provide a new challenge for experienced puzzlers as some established techniques may not work as well.

## Algorithm idea

1. choose approximate height and width of the puzzle (in pieces)
2. generate grid with these dimensions
3. in each grid cell, choose a random point
4. scale grid to desired puzzle size
5. calculate the voronoi diagram from these points
6. (optional) For even more irregularly shaped pieces, generate a random number of random points within each voronoi cell. Calculate the new voronoi diagram from these points. Join any new cells where the new points are within the same old voronoi cell. Repeat this step as often as desired.
7. for all voronoi edges (between pieces), replace the edge with a randomized connection scaled to the proper size.
8. (optional) user may provide a target piece count. Choose a grid with slightly more pieces than that, then merge random pieces or the smallest ones until the target count is reached.
9. (optional) user may provide a minimum surface area for each piece. If a piece is too small, it is merged with a neighboring piece. Additionally, ensure the bounding box of each piece has an aspect ratio not too far from 1 (e.g. <3:1)


Currently, step 6 only generates one new point per cell. This doesn't do much to change the piece shapes, but makes the initial grid less recognizable. Steps 8 and 9 also help a lot to mask the grid structure as already irregular pieces are merged.

Steps 8 and 9 are currently the only way to get non-convex shapes.

**Step 7 is not implemented yet.**

## Optimizations
- from the voronoi diagram, calculate verlet lists, listing the neighbours of each cell. Then use these lists when merging with the smallest neighbouring pieces, updating the neighbours lists accordingly.

## Roadmap
- [x] Implement basic algorithm
- [ ] Implement step 7: place connectors on edges.  
  This can become very tedious as I likely wantto generate these interlocking connectors procedurally as well - I don't know how to do that yet.  
  Since we store polygons, we should easily be able to iterate over edges, calculate their length and place a scaled, randomly oriented connector if the length is above a certain threshold. It may be difficult to avoid placing two connectors per edge.
- [ ] in steps 8 and 9, merge the adjacent piece with the longest shared edge, not to the one with the smallest area (current implementation)  
  This modification requires finding the maximum length of shared edges between two polygons. Due to steps 6 (with multiple points) and 7 & 8, polygons are not necessarily convex, so adjacent polygons can share multiple edges.
- [ ] split code into multiple files

