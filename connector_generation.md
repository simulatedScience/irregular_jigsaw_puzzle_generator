# Procedurally generating interlocking connectors for jigsaw puzzle pieces

## concepts:
### idea 1: Voronoi diagram
1. Generate a random number of points along two parallel lines (possibly curved)
2. generate six additional points in a 2x3 grid between the two lines (this will form the connector)
3. generate a voronoi diagram from the points
4. merge cells into two larger polygons separated by the line.

### idea 2: splines
1. pick a fixed pattern of points that make a connector
2. apply random offsets to move each points
3. calculate a spline through the points
