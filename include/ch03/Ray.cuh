#ifndef RAY_CUH
#define RAY_CUH

#include <glm/glm.hpp>


class Ray
{
public:
    __device__
    Ray() : mOrigin(0.0f, 0.0f, 0.0f), mDirection(0.0f, 0.0f, -1.0f) {}

    __device__
    Ray(
            const glm::vec3 & origin,
            const glm::vec3 & direction
    )
            : mOrigin(origin),
              mDirection(glm::normalize(direction))
    {
    }

    __device__
    glm::vec3 o() const
    {
        return mOrigin;
    }

    __device__
    glm::vec3 d() const
    {
        return mDirection;
    }

    __device__
    glm::vec3 at(float t) const
    {
        return mOrigin + t * mDirection;
    }

private:
    glm::vec3 mOrigin;
    glm::vec3 mDirection;
};


#endif  // RAY_CUH
