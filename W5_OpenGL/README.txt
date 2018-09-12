In this workshop you will use OpenGL together with CUDA to draw a wobbly sine wave. All of the relevant code is in main.cu.

Currently the wobble is performed inefficiently on the CPU in the function updateGeometryCPU(double t), where t determines the angle offset of our wave (which gets us the wobble).

You're going to implement the updateGeometryGPU by implementing a kernel function, and mapping the OpenGL buffers to CUDA resources. There are many references in main.cu to the appropriate CUDA function calls to do this.

You'll also notice that the normals are currently not set. It was a real pain to do this, and I'll leave this as an advanced exercise should you want to compute the normals on the GPU. You'll get to see my (slightly buggy) version of this when I dish out the solution.

An interesting point is the overhead of switching betweeh CUDA and OpenGL. It would be difficult to quantify as it is Driver / OS / GPU dependent. I found that this guy has a small app to compute the overhead - he seems to get about 1.25ms on his system (which could be vital in a game).

Good luck!

Richard Southern, 13/02/2014
