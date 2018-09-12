In this workshop we will use CUDA to tesselate the patches of the Utah teapot in parallel. You will use the simple method for NURB's tesselation given here: http://www.idav.ucdavis.edu/education/CAGDNotes/Matrix-Cubic-Bezier-Patch/Matrix-Cubic-Bezier-Patch.html in order to develop a kernel which will sample patches from the teapot model. Each patch is associated with a particular block, while each thread is a point in a patch. In this way, the whole teapot gets sampled in parallel, which is pretty neat.

In this workshop you will make use of a number of nifty new concepts: in particular shared memory and block thread synchronisation.

In my implementation I made several performance compromises to improve readability, in particular the use of Eigen in the threads to give you routines for matrix multiplication.

Good luck!

Richard Southern, 08/01/2013
