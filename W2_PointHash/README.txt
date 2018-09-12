This is the first part of a two part workshop on computing the Nearest Neighbours on a bunch of points in parallel.

Space is divided into cells in a regular grid. The size / resolution of these grid cells should relate to the radius of the nearest neighbour search, but for the purposes of this exercise it is just hardcoded.

The first part is to determine the cell index of the grid cell in which a point lies. Implement this on both the CPU and GPU and compare performance for different numbers of points / resolutions of grid - what can you deduce?

More details are in the comment in main.cu.

Richard Southern, 23/12/2013
