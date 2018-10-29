This isn't really a workshop, but an embarrassingly parallel worked example.

It generates an image from a given signed distance field. Currently these are implemented in 3D but as the image is 2D the Z value is fixed at 0.5.

What this example demonstrates is the use of 2D block sizes for image processing and the use of __device__ functions in kernels.

Note that the writeImage stuff was lifted from another source, and as a result it's pretty ugly. I don't recommend using this code in your own projects.

Richard Southern, 29/11/2018
