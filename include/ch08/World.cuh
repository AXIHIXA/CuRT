#ifndef WORLD_CUH
#define WORLD_CUH

#include "ch08/Hitable.cuh"


class World : public Hitable
{
public:
    __device__
    World() : list(nullptr), listSize(0) {}

    __device__
    World(Hitable ** __restrict__ l, int n) : list(l), listSize(n) {}

    __device__
    ~World() noexcept override = default;

    __device__
    bool hit(const Ray & r, float tMin, float tMax, HitRecord & rec) const override
    {
        HitRecord tmp;

        bool flag = false;
        float t = tMax;

        for (int i = 0; i < listSize; i++)
        {
            if (list[i]->hit(r, tMin, t, tmp))
            {
                flag = true;
                t = tmp.t;
                rec = tmp;
            }
        }

        return flag;
    }

private:
    Hitable ** list;
    int listSize;
};


#endif  // WORLD_CUH
