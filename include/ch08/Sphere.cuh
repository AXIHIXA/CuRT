#ifndef SPHERE_CUH
#define SPHERE_CUH

#include <glm/glm.hpp>
#include <ch03/Ray.cuh>
#include "ch08/Hitable.cuh"


class Sphere : public Hitable
{
public:
    __device__
    Sphere(): center(0.0f, 0.0f, 0.0f), radius(1.0f), pMaterial(nullptr) {}

    __device__
    Sphere(glm::vec3 c, float r, Material * m) : center(c), radius(r), pMaterial(m) {}

    __device__
    ~Sphere() noexcept override = default;

    __device__
    bool hit(const Ray & r, float tMin, float tMax, HitRecord & rec) const override
    {
        glm::vec3 oc = r.o() - center;
        float a = glm::dot(r.d(), r.d());
        float b = glm::dot(oc, r.d());
        float c = glm::dot(oc, oc) - radius * radius;

        if (float discriminant = b * b - a * c; 0.0f < discriminant)
        {
            float t = (-b - sqrtf(discriminant)) / a;

            if (tMin < t and t < tMax)
            {
                // Assigning a whole struct breaks the code, e.g.:
                // rec = {t, r.pointAtParameter(rec.t), (rec.p - center) / radius, pMaterial};
                // Note that rec.normal relies on UPDATED rec.p! lol
                rec.t = t;
                rec.p = r.at(rec.t);
                rec.normal = (rec.p - center) / radius;
                rec.pMaterial = pMaterial;
                return true;
            }

            t = (-b + sqrtf(discriminant)) / a;

            if (tMin < t and t < tMax)
            {
                rec.t = t;
                rec.p = r.at(rec.t);
                rec.normal = (rec.p - center) / radius;
                rec.pMaterial = pMaterial;
                return true;
            }
        }

        return false;
    }

    Material * pMaterial;

private:
    glm::vec3 center;
    float radius;
};


#endif  // SPHERE_CUH
