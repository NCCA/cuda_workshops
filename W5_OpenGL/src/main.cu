// Windows junk: I don't think this will compile on Windows.
#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// OpenGL Graphics includes
#include <GL/glew.h>
#if defined (__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

// includes, cuda
#include <cuda_runtime.h>
#include <cutil_math.h>
#include <cuda_gl_interop.h>

// Utilities and timing functions
//#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
#include <timer.h>               // timing functions

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check
#include <helper_cuda_gl.h>      // helper functions for CUDA/GL interop
#include <vector_types.h>

// For thrust routines (e.g. stl-like operators and algorithms on vectors)
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

// Needed for output functions within the kernel
#include <stdio.h>

// Need this for the accurate clock
#include <sys/time.h>

// printf() is only supported
// for devices of compute capability 2.0 and higher
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
   #define printf(f, ...) ((void)(f, __VA_ARGS__),0)
#endif

// My own include to help with the loading in of GLSL shaders
#include <shader.h>

// My own lame trackball class
#include <trackball.h>

//*************** DEFINES, CONSTANTS and GLOBALS
#define REFRESH_DELAY     10 //ms

// The const values used within the application
const unsigned int window_width  = 512;
const unsigned int window_height = 512;

// This defines the resolution of the mesh in both x and y direction
const unsigned int res = 28;

// Our shader object
Shader shader;

// Interaction function prototypes
Trackball tb;

// Our vertex array index
GLuint vertexArrayIdx = 0;

// Vertex Buffer Object indices for GLSL
GLuint posIdx = 0; // The positional array buffer index - dynamic
GLuint normalsIdx = 0; // The normal array buffer index - dynamic
GLuint elementsIdx = 0; // The element array (triangles) buffer index - static (won't change)

// Buffer resources for both CUDA and OpenGL
void *d_pos_buffer = NULL; // This is the pointer to the positional data
void *d_normals_buffer = NULL; // This is the pointer to the vertex normal data

// The time our application started
double startTime = 0.0;


//*************** FUNCTION PROTOTYPES
// Fairly standard glut initialisation function which initialises the callbacks and sets up the basic window
bool initGL(int, char **);

// Creates the scene by starting with the shader and then making all the associated buffers
bool initScene(); // The parent function, which creates the shader and vertex attrib array
void initVertexBuffers(); // Creates the VBO's and registers with CUDA
void initTriangles(); // Creates some connectivity for our mesh
void initLightsAndMaterials(); // Sets up some display uniforms for our shader

// Updates the points and the surface normals on the GPU or CPU - t is the time
void updateGeometryGPU(double /*t*/);
void updateGeometryCPU(double /*t*/);

// The display function: the geometry is updated on the GPU, the shader is loaded and the geometry is drawn
void display();

// The timer event - posts the message that the scene should be redrawn
void timerEvent(int value);

// Cleanup the memory used by this program, called on exit
void cleanup();

//*************** KERNEL FUNCTION PROTOTYPES


/**
 * Host main routine
 */
int main(int argc, char **argv) {
    // Init our window
    initGL(argc, argv);

    // Pick the best device for our own evil ends.
    cudaGLSetGLDevice(gpuGetMaxGflopsDeviceId());

    // Initialise the scene - creates the shader and the CUDA/GL resources
    initScene();

    // start rendering mainloop
    glutMainLoop();
    atexit(cleanup);

    cudaDeviceReset();
    return EXIT_SUCCESS;
}

void keyboard(unsigned char key, int /*x*/, int /*y*/) {
    tb.keyboard(key,0,0);
}


void mouse(int button, int state, int x, int y) {
    tb.mouse(button, state, x, y);
}

void motion(int x, int y) {
    tb.motion(x,y);
}

/**
  * Initialise GL using the standard GLUT procedure.
  */
bool initGL(int argc, char **argv) {   
    // Standard stuff
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("Cuda GL Interop (VBO)");
    glutDisplayFunc(display);
    glutTimerFunc(REFRESH_DELAY, timerEvent,0);

    // Make a trackball and register interaction functions
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);

    // initialize necessary OpenGL extensions
    glewInit();

    if (! glewIsSupported("GL_VERSION_3_0 "))
    {
        fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush(stderr);
        return false;
    }

    // default initialization
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glDisable(GL_DEPTH_TEST);
    glEnableClientState(GL_VERTEX_ARRAY);
    //glEnableClientState(GL_TEXTURE_COORD_ARRAY);

    // viewport
    glViewport(0, 0, window_width, window_height);

    // Set the start time
    struct timeval tim;
    gettimeofday(&tim, NULL);
    startTime = tim.tv_sec+(tim.tv_usec * 1.0e-6);

    SDK_CHECK_ERROR_GL();
    return true;
}

/**
  * Initialise the scene. Don't fiddle with this.
  */
bool initScene() {
    // First load up and bind our shaders
    shader.init("shaders/phong.vs", "shaders/phong.fs");
    shader.bind();
    shader.printProperties();

    // We need to bind a vertex array before we start making array buffers
    glGenVertexArrays(1, &vertexArrayIdx);
    glBindVertexArray(vertexArrayIdx);

    glEnable(GL_DEPTH_TEST);
    // Initialise all the vertex buffers
    initVertexBuffers();

    // Initialise the element arrays for triangles
    initTriangles();

    // Initialise the lights and material uniforms
    initLightsAndMaterials();

    // Unbind the shader and vertex array
    glBindVertexArray(0);

    // Unbind our shader
    shader.unbind();

    // Check for GL errors
    SDK_CHECK_ERROR_GL();
    return true;
}

/**
  * Initialise the vertex buffers (you will need to register the cuda resource here)
  */
void initVertexBuffers() {
    // Create our GL buffers and bind them to CUDA
    glGenBuffers(1, &posIdx); // Generate the point buffer index
    glBindBuffer(GL_ARRAY_BUFFER, posIdx); // Bind it (all following operations apply)
    glBufferData(GL_ARRAY_BUFFER, res*res*sizeof(float3), 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0); // Unbind our buffers

    // register this buffer object with CUDA
    // http://developer.download.nvidia.com/compute/cuda/4_1/rel/toolkit/docs/online/group__CUDART__OPENGL_g43f69a041bdfa4f8b36aff99bf0171db.html

    // Now do the same for the normals
    glGenBuffers(1, &normalsIdx);
    glBindBuffer(GL_ARRAY_BUFFER, normalsIdx);
    glBufferData(GL_ARRAY_BUFFER, res*res*sizeof(float3), 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // register this buffer object with CUDA
    // http://developer.download.nvidia.com/compute/cuda/4_1/rel/toolkit/docs/online/group__CUDART__OPENGL_g43f69a041bdfa4f8b36aff99bf0171db.html
}

/**
  * Create the triangle indices (don't fiddle with this)
  */
void initTriangles() {
    // Define some connectivity information for our sheet.
    unsigned int num_tris = (res-1)*(res-1)*2;
    GLuint *tris = new GLuint[num_tris*3];
    int i, j, fidx = 0;
    for (i=0; i < res - 1; ++i) {
        for (j=0; j < res - 1; ++j) {
            tris[fidx*3+0] = i*res+j; tris[fidx*3+1] = i*res+j+1; tris[fidx*3+2] = (i+1)*res+j;
            fidx++;
            tris[fidx*3+0] = i*res+j+1; tris[fidx*3+1] = (i+1)*res+j+1; tris[fidx*3+2] = (i+1)*res+j;
            fidx++;
        }
    }

    // Create our buffer to contain element data (we will use indexed arrays to draw this shape)
    glGenBuffers(1, &elementsIdx);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementsIdx);

    // Note that in this call we copy the data to the GPU from the tris array
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 3*num_tris*sizeof(GLuint), tris, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    delete [] tris; // We can delete this array - it has been copied to the GPU
}

/**
  * Setup the shader materials and lights (don't fiddle with this)
  */
void initLightsAndMaterials() {
    // Bind our shader to the current context
    shader.bind();

    // Set some of the default lighting values on our shader (otherwise phong won't be sexy)
    GLfloat lightPos[] = {0.0f, 10.0f, 0.0f, 2.0f};
    GLfloat lightAmbient[] = {0.0f, 0.0f, 0.0f};
    GLfloat lightDiffuse[] = {1.0f, 1.0f, 1.0f};
    GLfloat lightSpecular[] = {1.0f, 1.0f, 1.0f};
    glUniform4fv(glGetUniformLocation(shader.id(), "u_Light.Position"), 1, lightPos);
    glUniform3fv(glGetUniformLocation(shader.id(), "u_Light.La"), 1, lightAmbient);
    glUniform3fv(glGetUniformLocation(shader.id(), "u_Light.Ld"), 1, lightDiffuse);
    glUniform3fv(glGetUniformLocation(shader.id(), "u_Light.Ls"), 1, lightSpecular);

    // Set some material (reflectivity) properties for our shader
    GLfloat matAmbient[] = {0.2f, 0.2f, 0.2f};
    GLfloat matDiffuse[] = {1.0f, 1.0f, 1.0f};
    GLfloat matSpecular[] = {1.0f, 1.0f, 1.0f};
    glUniform3fv(glGetUniformLocation(shader.id(), "u_Material.Ka"), 1, matAmbient);
    glUniform3fv(glGetUniformLocation(shader.id(), "u_Material.Kd"), 1, matDiffuse);
    glUniform3fv(glGetUniformLocation(shader.id(), "u_Material.Ks"), 1, matSpecular);
    glUniform1f(glGetUniformLocation(shader.id(), "u_Material.Shininess"), 4.0f);

    // Set up the projection matrix and put it on the shader (this is not built in on eigen :( )
    GLfloat zFar = 4.0f; GLfloat zNear = 0.5f; GLfloat fovY = 80.0f;
    GLint proj_id = glGetUniformLocation(shader.id(), "u_ProjectionMatrix");

    GLfloat *P = new GLfloat[16];
    tb.getProjectionMatrix(P, window_width, window_height, fovY, zNear, zFar);
    glUniformMatrix4fv(/*location*/ proj_id,
                       /*# of matrices*/ 1,
                       /*transpose?*/ GL_FALSE,
                       /*The matrix pointer*/ P);
    delete [] P;

    // Bind our shader to the current context
    shader.unbind();
}

/**
 * @brief updateGeometryCPU update the vertices (not the normals) on the CPU
 * @param etime
 */
void updateGeometryCPU(double etime) {
    float3 *h_pos_ptr = (float3*) malloc(sizeof(float3)*res*res);
    float3 *h_normals_ptr = (float3*) malloc(sizeof(float3)*res*res);

    unsigned int x,y;

    for (x=0; x < res; ++x) {
        for (y=0; y< res; ++y) {
            // calculate uv coordinates
            float u = float(x) / float(res);
            float v = float(y) / float(res);
            u = u*2.0f - 1.0f;
            v = v*2.0f - 1.0f;

            // calculate simple sine wave pattern
            float freq = 6.0f;
            float w = sin(u*freq + etime) * cos(v*freq + etime) * 0.25f;

            // write output vertex
            h_pos_ptr[x*res+y] = make_float3(u, w, v);

            // write out a dummy normal
            h_normals_ptr[x*res+y] = make_float3(0.0f, 1.0f, 0.0f);
        }
    }
    // Bind the buffer
    glBindBuffer(GL_ARRAY_BUFFER, posIdx);
    // Copy data to this buffer
    glBufferData(GL_ARRAY_BUFFER, res*res*sizeof(float3), h_pos_ptr, GL_DYNAMIC_DRAW);
    // Bind normals buffer
    glBindBuffer(GL_ARRAY_BUFFER, normalsIdx);
    // Copy data into the buffer
    glBufferData(GL_ARRAY_BUFFER, res*res*sizeof(float3), h_normals_ptr, GL_DYNAMIC_DRAW);
    // Unbind the buffers
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Delete our allocated memory
    delete [] h_pos_ptr;
    delete [] h_normals_ptr;
}

/**
  * Update the positions of the nodes of our sheet. 
  * This involves mapping the GL pointers, updating
  * the node positions, possibly computing surface normals, and
  * afterwards unmapping the pointers.
  */
void updateGeometryGPU(double t) {
    // Map the graphics resources for access by CUDA
    // http://developer.download.nvidia.com/compute/cuda/4_1/rel/toolkit/docs/online/group__CUDART__INTEROP_gb7064fb72e54d89d0666e192b45d35cc.html
    
    // Get the mapped pointer to use with CUDA
    // http://developer.download.nvidia.com/compute/cuda/4_1/rel/toolkit/docs/online/group__CUDART__INTEROP_ge7f893864a3d38a630e71a99f5a4e17f.html
    
    // Execute the kernel to update the position of the vertices

    // unmap buffer objects when you're done with them
    // http://horacio9573.no-ip.org/cuda/group__CUDART__INTEROP_gc4dcf300df27f8cf51a89f0287b07861.html
}

/**
  * Display the scene. The scene geometry is first updated and then the usual GLSL shader commands are
  * executed.
  */
void display()
{    
    struct timeval tim;
    gettimeofday(&tim, NULL);
    double now = tim.tv_sec+(tim.tv_usec * 1.0e-6);

    // run updateGeometry on either the CPU or the GPU to generate vertex positions 
    // (I think this cannot be done if the buffer is currently bound)
    updateGeometryCPU(now - startTime );

    // Clear the window
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Bind our shader to the current context
    shader.bind();
    // Bind our vertex array (which will have the attributes of the vertices, like normals and verts)
    glBindVertexArray(vertexArrayIdx);

    // Retrieve the attribute location from our currently bound shader, enable and
    // bind the vertex attrib pointer to our currently bound buffer
    glBindBuffer(GL_ARRAY_BUFFER, posIdx); // Bind it (all following operations apply)
    GLint vertAttribLoc = glGetAttribLocation(shader.id(), "a_VertexPosition");
    glEnableVertexAttribArray(vertAttribLoc);
    glVertexAttribPointer(vertAttribLoc, 3, GL_FLOAT, GL_FALSE, 0, 0);

    // Do the same for the normals
    glBindBuffer(GL_ARRAY_BUFFER, normalsIdx);
    GLint normAttribLoc = glGetAttribLocation(shader.id(), "a_VertexNormal");
    glEnableVertexAttribArray(normAttribLoc);
    glVertexAttribPointer(normAttribLoc, 3, GL_FLOAT, GL_TRUE, 0, 0);

    // Set up the modelview and normal matrices based on the mouse and keyboard input
    GLint mvLoc = glGetUniformLocation(shader.id(), "u_ModelViewMatrix");
    GLint nmLoc = glGetUniformLocation(shader.id(), "u_NormalMatrix");

    GLfloat *M = new GLfloat[16]; GLfloat *N = new GLfloat[9];
    tb.getModelViewMatrix(M);
    tb.getNormalMatrix(N);
    glUniformMatrix4fv(/*location*/mvLoc,
                       /*# of matrices*/1,
                       /*transpose?*/GL_FALSE,
                       /*The matrix pointer*/ M);

    // The normal matrix is the 3x3 (rotational part) of the inverse, transposed mv matrix
    glUniformMatrix3fv(nmLoc, 1, GL_TRUE, N);  

    delete [] M;
    delete [] N;

    // Draw our elements (the element buffer should still be enabled from the previous call to initScene())
    unsigned int num_tris = (res-1)*(res-1)*2;
    unsigned int num_points = res * res;
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementsIdx);
    glDrawElements(GL_TRIANGLES, num_tris*3, GL_UNSIGNED_INT, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    glDrawArrays(GL_POINTS, 0, num_points);

    // Unbind our shader
    glBindBuffer(GL_ARRAY_BUFFER,0);
    glBindVertexArray(0);
    shader.unbind();

    glutSwapBuffers();

    shader.unbind();
}

/**
  * The timer event triggers to force the redisplay message.
  */
void timerEvent(int value)
{
    glutPostRedisplay();
    glutTimerFunc(REFRESH_DELAY, timerEvent,0);
}

/**
  * Detach the cuda resource and then delete all memory allocated on the GPU by GLSL
  */
void cleanup() {
    if (posIdx) {
	// Make sure to unregister your resource before deleting the OpenGL
	// buffer, otherwise nasty things could happen!
	// http://developer.download.nvidia.com/compute/cuda/4_1/rel/toolkit/docs/online/group__CUDART__INTEROP_g1d45ac44d1affe17fb356e0b7a0b0560.html
        glDeleteBuffers(1, &posIdx);
        posIdx = 0;
    }
    if (normalsIdx) {
	// Make sure to unregister your resource before deleting the OpenGL
	// buffer, otherwise nasty things could happen!
	// http://developer.download.nvidia.com/compute/cuda/4_1/rel/toolkit/docs/online/group__CUDART__INTEROP_g1d45ac44d1affe17fb356e0b7a0b0560.html
        glDeleteBuffers(1,&normalsIdx);
        normalsIdx = 0;
    }
    if (elementsIdx) {
        glDeleteBuffers(1, &elementsIdx);
        elementsIdx = 0;
    }
}


