#ifndef MATERIAL_CUH
#define MATERIAL_CUH

#include <glm/glm.hpp>
#include "ch03/Ray.cuh"
#include "ch08/Hitable.cuh"


__device__
glm::vec3 randomVectorInUnitSphere(curandState * localRandState)
{
    float r = curand_uniform(localRandState);
    float phi = curand_uniform(localRandState) * M_PIf32 * 2.0f;
    float theta = curand_uniform(localRandState) * M_PIf32;
    return {r * cosf(phi) * sinf(theta), r * sinf(phi) * sinf(theta), r * cosf(theta)};
}


class Material
{
public:
    __device__
    virtual ~Material() noexcept = 0;

    __device__
    virtual bool
    scatter(
            const Ray & rayIn,
            const HitRecord & rec,
            glm::vec3 & attenuation,
            Ray & rayScattered,
            curandState * localRandState
    ) const = 0;
};


__device__
Material::~Material() noexcept = default;


class Lambertian : public Material
{
public:
    __device__
    explicit Lambertian(const glm::vec3 & a) : albedo(a) {}

    __device__
    ~Lambertian() noexcept override = default;

    __device__
    bool scatter(
            const Ray & rayIn,
            const HitRecord & rec,
            glm::vec3 & attenuation,
            Ray & rayScattered,
            curandState * localRandState
    ) const override
    {
        glm::vec3 target = rec.p + rec.normal + randomVectorInUnitSphere(localRandState);
        rayScattered = Ray(rec.p, target - rec.p);
        attenuation = albedo;
        return true;
    }

private:
    glm::vec3 albedo;
};


class Metal : public Material
{
public:
    __device__
    Metal(const glm::vec3 & a, float f) : albedo(a), fuzz(f < 1.0f ? f : 1.0f) {}

    __device__
    ~Metal() noexcept override = default;

    __device__
    bool scatter(
            const Ray & rayIn,
            const HitRecord & rec,
            glm::vec3 & attenuation,
            Ray & rayScattered,
            curandState * localRandState
    ) const override
    {
        glm::vec3 reflected = glm::reflect(rayIn.direction(), rec.normal);
        rayScattered = Ray(rec.p, reflected + fuzz * randomVectorInUnitSphere(localRandState));
        attenuation = albedo;
        return 0.0f < glm::dot(rayScattered.direction(), rec.normal);
    }

private:
    glm::vec3 albedo;
    float fuzz;
};


__device__
float schlick(float cosine, float refIdx)
{
    float r0 = (1.0f - refIdx) / (1.0f + refIdx);
    r0 *= r0;
    return r0 + (1.0f - r0) * powf((1.0f - cosine), 5.0f);
}


__device__
bool refract(const glm::vec3 & v, const glm::vec3 & n, float niOverNt, glm::vec3 & refracted)
{
    glm::vec3 uv = glm::normalize(v);
    float dt = dot(uv, n);

    if (float discriminant = 1.0f - niOverNt * niOverNt * (1 - dt * dt); 0.0f < discriminant)
    {
        refracted = niOverNt * (uv - n * dt) - n * sqrtf(discriminant);
        return true;
    }
    else
    {
        return false;
    }
}


class Dielectric : public Material
{
public:
    __device__
    explicit Dielectric(float ri) : refIdx(ri) {}

    __device__
    bool scatter(
            const Ray & rayIn,
            const HitRecord & rec,
            glm::vec3 & attenuation,
            Ray & rayScattered,
            curandState * localRandState
    ) const override
    {
        attenuation = glm::vec3(1.0f, 1.0f, 1.0f);

        glm::vec3 outwardNormal;
        float niOverNt;
        float cosine;

        if (0.0f < glm::dot(rayIn.direction(), rec.normal))
        {
            outwardNormal = -rec.normal;
            niOverNt = refIdx;
            cosine = glm::dot(rayIn.direction(), rec.normal) / sqrtf(glm::dot(rayIn.direction(), rayIn.direction()));
            cosine = sqrtf(1.0f - refIdx * refIdx * (1.0f - cosine * cosine));
        }
        else
        {
            outwardNormal = rec.normal;
            niOverNt = 1.0f / refIdx;
            cosine = -glm::dot(rayIn.direction(), rec.normal) / sqrtf(glm::dot(rayIn.direction(), rayIn.direction()));
        }

        glm::vec3 refracted;

        float reflectProbability =
                refract(rayIn.direction(), outwardNormal, niOverNt, refracted) ?
                schlick(cosine, refIdx) :
                1.0f;

        rayScattered =
                curand_uniform(localRandState) < reflectProbability ?
                Ray(rec.p, reflect(rayIn.direction(), rec.normal)) :
                Ray(rec.p, refracted);

        return true;
    }

private:
    float refIdx;
};


#endif  // MATERIAL_CUH
