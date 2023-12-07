#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <opencv2/opencv.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "ch03/Ray.cuh"
#include "ch05/Hitable.cuh"
#include "ch05/Sphere.cuh"
#include "ch05/World.cuh"
#include "util/CudaUtil.h"


constexpr dim3 kBlockDim {32U, 32U, 1U};

constexpr int kWidth = 1200;

constexpr int kHeight = 600;

constexpr float kFloatMax = std::numeric_limits<float>::max();


__device__
glm::vec3 color(const Ray & r, Hitable ** __restrict__ world)
{
    HitRecord rec;

    if ((*world)->hit(r, 0.0f, kFloatMax, rec))
    {
        return 0.5f * glm::vec3(rec.normal.x + 1.0f, rec.normal.y + 1.0f, rec.normal.z + 1.0f);
    }
    else
    {
        float t = 0.5f * (r.d().y + 1.0f);
        return (1.0f - t) * glm::vec3(1.0f, 1.0f, 1.0f) + t * glm::vec3(0.5f, 0.7f, 1.0f);
    }
}


__global__
void createWorld(Hitable ** __restrict__ list, Hitable ** __restrict__ world)
{
    list[0] = new Sphere({0.0f, 0.0f, -1.0f}, 0.5f);
    list[1] = new Sphere({0.0f, -100.5f, -1.0f}, 100.0f);
    *world = new World(list, 2);
}


__global__
void freeWorld(Hitable ** __restrict__ list, Hitable ** __restrict__ world)
{
    delete list[0];
    delete list[1];
    delete *world;
}


__global__
void render(
        glm::vec3 * __restrict__ fb,
        int width,
        int height,
        glm::vec3 lowerLeft,
        glm::vec3 horizontal,
        glm::vec3 vertical,
        glm::vec3 origin,
        Hitable ** __restrict__ world
)
{
    auto x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    auto y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);

    if (x < width and y < height)
    {
        float u = static_cast<float>(x) / static_cast<float>(width);
        float v = static_cast<float>(y) / static_cast<float>(height);

        // Ray from global-space origin through viewport pixel
        Ray r(origin, lowerLeft + u * horizontal + v * vertical);
        fb[y * width + x] = color(r, world);
    }
}


int main(int argc, char * argv[])
{
    // Carefully manage memory on the GPU.
    // Sanitize memory leaks with
    // $ compute-sanitizer --leak-check=full ./cmake-build-release/05_normal

    // World setup.
    // Note that we are making use of polymorphism, so all structures should be pointers.
    // Also note that these structures are __deivice__ and thus could be initialized on device only.
    // Thus: We allocate pointers on host, and new (allocate & initilize) structures on device.
    thrust::device_vector<Hitable *> dHitablePtrPool(3);
    Hitable ** dList = dHitablePtrPool.data().get();
    Hitable ** dWorld = dList + 2;
    createWorld<<<1U, 1U>>>(dList, dWorld);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Render
    dim3 mGridDim = {
            (kWidth + kBlockDim.x - 1U) / kBlockDim.x,
            (kHeight + kBlockDim.y - 1U) / kBlockDim.y,
            1U
    };

    thrust::device_vector<glm::vec3> dFb(kWidth * kHeight, {1.0f, 1.0f, 1.0f});

    render<<<mGridDim, kBlockDim>>>(
            dFb.data().get(),
            kWidth,
            kHeight,
            glm::vec3(-2.0f, -1.0f, -1.0f),
            glm::vec3(4.0f, 0.0f, 0.0f),
            glm::vec3(0.0f, 2.0f, 0.0f),
            glm::vec3(0.0f, 0.0f, 0.0f),
            dWorld
    );
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Free resources
    freeWorld<<<1U, 1U>>>(dList, dWorld);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Display rendered image
    thrust::host_vector<glm::vec3> hFb = dFb;
    cv::Mat img(kHeight, kWidth, CV_32FC3, hFb.data());
    cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
    cv::flip(img, img, 0);
    cv::imshow("Chap 05 Normal", img);
    cv::waitKey();

    return EXIT_SUCCESS;
}
