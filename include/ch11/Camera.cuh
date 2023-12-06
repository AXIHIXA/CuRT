#ifndef CAMERA_CUH
#define CAMERA_CUH

#include <glm/glm.hpp>
#include "ch03/Ray.cuh"


__device__
glm::vec3 randomVectorInUnitSphere(curandState * localRandState);


class Camera
{
public:
    __device__
    Camera(
            const glm::vec3 & lookfrom,
            const glm::vec3 & lookat,
            const glm::vec3 & vup,
            float vfov,
            float aspect,
            float aperture,
            float focusDistance
    )
            : origin(lookfrom),
              lensRadius(aperture / 2.0f)
    {
        // vfov is top to bottom in degrees
        float theta = vfov * M_PIf32 / 180.0f;
        float halfHeight = tanf(theta / 2.0f);
        float halfWidth = aspect * halfHeight;
        w = glm::normalize(lookfrom - lookat);
        u = glm::normalize(glm::cross(vup, w));
        v = glm::cross(w, u);

        lowerLeft = origin - halfWidth * focusDistance * u - halfHeight * focusDistance * v - focusDistance * w;
        horizontal = 2.0f * halfWidth * focusDistance * u;
        vertical = 2.0f * halfHeight * focusDistance * v;
    }

    __device__
    Ray castRayAt(float s, float t, curandState * localRandState) const
    {
        glm::vec3 rd = lensRadius * randomVectorInUnitSphere(localRandState);
        glm::vec3 offset = u * rd.x + v * rd.y;
        return {origin + offset, lowerLeft + s * horizontal + t * vertical - origin - offset};
    }

private:
    glm::vec3 origin {};
    glm::vec3 lowerLeft {};
    glm::vec3 horizontal {};
    glm::vec3 vertical {};

    glm::vec3 u {};
    glm::vec3 v {};
    glm::vec3 w {};

    float lensRadius {0.0f};
};


#endif  // CAMERA_CUH
