#ifndef RAY_CUH
#define RAY_CUH

#include <glm/glm.hpp>


class Ray
{
public:
    __device__
    Ray() : o(0.0f, 0.0f, 0.0f), d(0.0f, 0.0f, -1.0f) {}

    __device__
    Ray(const glm::vec3 & o, const glm::vec3 & d) : o(o), d(glm::normalize(d)) {}

    __device__
    glm::vec3 origin() const
    {
        return o;
    }

    __device__
    glm::vec3 direction() const
    {
        return d;
    }

    __device__
    glm::vec3 pointAtParameter(float t) const
    {
        return o + t * d;
    }

private:
    glm::vec3 o;
    glm::vec3 d;
};


#endif  // RAY_CUH
