// OpenGL Graphics includes
#include <GL/glew.h>
#if defined (__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

#include <trackball.h>
#include <iostream>

// glm::vec3, glm::vec4, glm::ivec4, glm::mat4
#include <glm/glm.hpp>
// glm::translate, glm::rotate, glm::scale, glm::perspective
#include <glm/gtc/matrix_transform.hpp>
// glm::value_ptr
#include <glm/gtc/type_ptr.hpp>

#include <glm/gtc/matrix_access.hpp>


Trackball::Trackball() {
    mouse_buttons = 0;
    rotate_x = 0.0;
    rotate_y = 0.0;
    translate_z = -3.0;
}

Trackball::~Trackball() {
}

/**
  *
  */
void Trackball::keyboard(unsigned char key, int /*x*/, int /*y*/)
{
    switch (key)
    {
    case (27) :
        exit(EXIT_SUCCESS);
        break;
    }
}

/**
  *
  */
void Trackball::mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
    {
        mouse_buttons |= 1<<button;
    }
    else if (state == GLUT_UP)
    {
        mouse_buttons = 0;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}

/**
  *
  */
void Trackball::motion(int x, int y)
{
    float dx, dy;
    dx = (float)(x - mouse_old_x);
    dy = (float)(y - mouse_old_y);

    if (mouse_buttons & 1)
    {
        rotate_x += dy * 0.2f;
        rotate_y += dx * 0.2f;
    }
    else if (mouse_buttons & 4)
    {
        translate_z += dy * 0.01f;
    }
    mouse_old_x = x;
    mouse_old_y = y;
}

void Trackball::getProjectionMatrix(GLfloat *P, GLuint w, GLuint h, float fovY, float zNear, float zFar) {
    glm::mat4 proj;
    GLfloat r = GLfloat(w) / GLfloat(h);
    proj = glm::perspective(fovY, r, zNear, zFar);
    memcpy(P, glm::value_ptr(proj), 16*sizeof(GLfloat));
}

void Trackball::getModelViewMatrix(GLfloat *M) {
    glm::mat4 mv(1.0);
    mv = glm::translate(mv, glm::vec3(0.0f, 0.0f, translate_z));
    mv = glm::rotate(mv, rotate_x, glm::vec3(1.0f, 0.0f, 0.0f));
    mv = glm::rotate(mv, rotate_y, glm::vec3(0.0f, 1.0f, 0.0f));
    memcpy(M, glm::value_ptr(mv), 16*sizeof(GLfloat));
}

void Trackball::getNormalMatrix(GLfloat *N) {
    glm::mat4 mv = glm::mat4(1.0);
    mv = glm::rotate(mv, rotate_x, glm::vec3(1.0f, 0.0f, 0.0f));
    mv = glm::rotate(mv, rotate_y, glm::vec3(0.0f, 1.0f, 0.0f));
    memcpy(N, glm::value_ptr(glm::inverse(glm::mat3(mv))), 9 * sizeof(GLfloat));
}

