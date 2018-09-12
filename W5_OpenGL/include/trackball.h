#ifndef __TRACKBALL_H
#define __TRACKBALL_H

// Needed for matrices
//#include <eigen-nvcc/Dense>
//#include <eigen-nvcc/Geometry>


class Trackball {
public:
    Trackball();
    ~Trackball();
    void keyboard(unsigned char key, int /*x*/, int /*y*/);
    void mouse(int button, int state, int x, int y);
    void motion(int x, int y);

    /// Utility function to compute the projection matrix for GLSL
    void getProjectionMatrix(GLfloat*, GLuint w, GLuint h, float fovY, float zNear, float zFar);

    /// Return a pointer to the modelview matrix data
    void getModelViewMatrix(GLfloat*);

    /// Return a pointer to the normal matrix
    void getNormalMatrix(GLfloat*);

private:
    /// A bunch of parameters to keep track of the mouse position
    int mouse_old_x, mouse_old_y;
    int mouse_buttons;
    float rotate_x, rotate_y;
    float translate_z;
};

#endif //__TRACKBALL_H

