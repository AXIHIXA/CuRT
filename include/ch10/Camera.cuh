#ifndef CAMERA_CUH
#define CAMERA_CUH

#include <glm/glm.hpp>
#include "ch03/Ray.cuh"


class Camera
{
public:
    __device__
    Camera(
            const glm::vec3 & lookfrom,
            const glm::vec3 & lookat,
            const glm::vec3 & vup,
            float vfov,
            float aspect
    )
    {
        // vfov is top to bottom in degrees
        float theta = vfov * M_PIf32 / 180.0f;
        float halfHeight = tanf(theta / 2.0f);
        float halfWidth = aspect * halfHeight;
        glm::vec3 w = glm::normalize(lookfrom - lookat);
        glm::vec3 u = glm::normalize(glm::cross(vup, w));
        glm::vec3 v = glm::cross(w, u);

        origin = lookfrom;
        lowerLeft = origin - halfWidth * u - halfHeight * v - w;
        horizontal = 2 * halfWidth * u;
        vertical = 2 * halfHeight * v;
    }

    __device__
    Ray castRayAt(float u, float v) const
    {
        return {origin, lowerLeft + u * horizontal + v * vertical - origin};
    }

private:
    glm::vec3 origin {};
    glm::vec3 lowerLeft {};
    glm::vec3 horizontal {};
    glm::vec3 vertical {};
};


#endif  // CAMERA_CUH
