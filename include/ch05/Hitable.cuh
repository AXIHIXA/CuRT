#ifndef HITABLE_CUH
#define HITABLE_CUH

#include <glm/glm.hpp>


class Ray;


struct HitRecord
{
    __device__
    HitRecord() : t(0.0f), p(0.0f, 0.0f, 0.0f), normal(0.0f, 0.0f, 0.0f) {}

    float t;
    glm::vec3 p;
    glm::vec3 normal;
};


class Hitable
{
public:
    __device__
    virtual ~Hitable() noexcept = 0;

    __device__
    virtual bool hit(const Ray & r, float tMin, float tMax, HitRecord & rec) const = 0;
};


__device__
Hitable::~Hitable() noexcept = default;


#endif  // HITABLE_CUH
