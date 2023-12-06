#ifndef CAMERA_CUH
#define CAMERA_CUH

#include <glm/glm.hpp>
#include "ch03/Ray.cuh"


class Camera
{
public:
    __device__
    Camera()
            : origin(0.0f, 0.0f, 0.0f),
              lowerLeft(-2.0f, -1.0f, -1.0f),
              horizontal(4.0f, 0.0f, 0.0f),
              vertical(0.0f, 2.0f, 0.0f)
    {}

    __device__
    Ray castRayAt(float u, float v) const
    {
        return {origin, lowerLeft + u * horizontal + v * vertical - origin};
    }

    glm::vec3 origin;
    glm::vec3 lowerLeft;
    glm::vec3 horizontal;
    glm::vec3 vertical;
};


#endif  // CAMERA_CUH
