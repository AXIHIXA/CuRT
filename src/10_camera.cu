#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <glm/glm.hpp>
#include <opencv2/opencv.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "ch03/Ray.cuh"
#include "ch08/Hitable.cuh"
#include "ch08/Material.cuh"
#include "ch08/Sphere.cuh"
#include "ch08/World.cuh"
#include "ch10/Camera.cuh"
#include "util/CudaUtil.h"


constexpr dim3 kBlockDim {32U, 32U, 1U};

constexpr int kWidth = 1200;

constexpr int kHeight = 600;

constexpr float kFloatMax = std::numeric_limits<float>::max();

constexpr float kHitEps = 1e-3f;

constexpr int kMaxRecursionLevel = 50;

// SSAA (Super-Sample Anti Aliasing):
//     Simply render at a higher resolution and downsample back.
//     Fragment shading is done for multiple times for each pixel.
// MSAA (Multi-Sample Anti Aliasing):
//     For each pixel, do fragment shading only once at its center,
//     but do hitting tests (depth/stencil tests) for multiple times
//     at randomly-sampled points inside this pixel,
//     and dilute the fragment color with background color
//     by number of hits over number of tests.
//     Requires more tuning on the World class
//     (ignoring one specific object for background color).
constexpr int kAntiAliasingFactor = 100;


__device__
glm::vec3 color(
        const Ray & r,
        Hitable ** __restrict__ world,
        curandState * __restrict__ localRandState
)
{
    Ray currentRay = r;
    glm::vec3 currentAttenuation = {1.0f, 1.0f, 1.0f};

    for (int i = 0; i != kMaxRecursionLevel; ++i)
    {
        HitRecord rec;

        // Ignore hits that are very close to the calculated intersection point
        // to wipe off intersections points rounded-off to the interior of the sphere.
        if ((*world)->hit(currentRay, kHitEps, kFloatMax, rec))
        {
            Ray rayScattered;
            glm::vec3 attenuation;

            if (rec.pMaterial->scatter(currentRay, rec, attenuation, rayScattered, localRandState))
            {
                currentAttenuation *= attenuation;
                currentRay = rayScattered;
            }
            else
            {
                return {0.0f, 0.0f, 0.0f};
            }
        }
        else
        {
            float t = 0.5f * (currentRay.d().y + 1.0f);
            glm::vec3 c = (1.0f - t) * glm::vec3(1.0f, 1.0f, 1.0f) + t * glm::vec3(0.5f, 0.7f, 1.0f);
            return currentAttenuation * c;
        }
    }

    // exceeded recursion
    return {0.0f, 0.0f, 0.0f};
}


__global__
void createWorld(
        Hitable ** __restrict__ list,
        Hitable ** __restrict__ world,
        Camera ** __restrict__ camera,
        int width,
        int height
)
{
    list[0] = new Sphere({0.0f, 0.0f, -1.0f}, 0.5f, new Lambertian({0.1f, 0.2f, 0.5f}));
    list[1] = new Sphere({0.0f, -100.5f, -1.0f}, 100.0f, new Lambertian({0.8f, 0.8f, 0.0f}));
    list[2] = new Sphere({1.0f, 0.0f, -1.0f}, 0.5f, new Metal({0.8f, 0.6f, 0.2f}, 0.0f));
    list[3] = new Sphere({-1.0f, 0.0f, -1.0f}, 0.5f, new Dielectric(1.5f));
    list[4] = new Sphere({-1.0f, 0.0f, -1.0f}, -0.45f, new Dielectric(1.5f));
    *world = new World(list, 5);
    *camera = new Camera(
            {-2.0f, 2.0f, 1.0f},
            {0.0f, 0.0f, -1.0f},
            {0.0f, 1.0f, 0.0f},
            20.0f,
            static_cast<float>(width) / static_cast<float>(height)
    );
}


__global__
void freeWorld(
        Hitable ** __restrict__ list,
        Hitable ** __restrict__ world,
        Camera ** camera
)
{
    for (int i = 0; i != 5; ++i)
    {
        delete reinterpret_cast<Sphere *>(list[i])->pMaterial;
        delete list[i];
    }

    delete *world;
    delete *camera;
}


__global__
void initRandState(
        curandState * __restrict__ randState,
        int width,
        int height
)
{
    auto x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    auto y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);

    if (x < width and y < height)
    {
        // Each thread gets same seed, a different sequence number, no offset.
        curand_init(1984, y * width + x, 0, randState + y * width + x);
    }
}


__global__
void render(
        glm::vec3 * __restrict__ fb,
        int width,
        int height,
        int antiAliasingFactor,
        Camera ** __restrict__ camera,
        Hitable ** __restrict__ world,
        curandState * __restrict__ randState
)
{
    auto x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    auto y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);

    if (x < width and y < height)
    {
        int pixelIdx = y * width + x;

        // Generator state can be stored in global memory between kernel launches,
        // used in local memory for fast generation,
        // and then stored back into global memory.
        // See https://docs.nvidia.com/cuda/curand/device-api-overview.html#performance-notes
        curandState localRandState = randState[pixelIdx];

        glm::vec3 tmpColor {0.0f, 0.0f, 0.0f};

        for (int s = 0; s != antiAliasingFactor; ++s)
        {
            float u = (static_cast<float>(x) + curand_uniform(&localRandState)) / static_cast<float>(width);
            float v = (static_cast<float>(y) + curand_uniform(&localRandState)) / static_cast<float>(height);
            tmpColor += color((*camera)->castRayAt(u, v), world, &localRandState);
        }

        tmpColor /= static_cast<float>(antiAliasingFactor);

        // Gamma correction for images
        tmpColor.x = sqrtf(tmpColor.x);
        tmpColor.y = sqrtf(tmpColor.y);
        tmpColor.z = sqrtf(tmpColor.z);

        fb[pixelIdx] = tmpColor;
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
    thrust::device_vector<void *> dPtrPool(7);
    Hitable ** dList = reinterpret_cast<Hitable **>(dPtrPool.data().get());
    Hitable ** dWorld = dList + 5;
    Camera ** dCamera = reinterpret_cast<Camera **>((dPtrPool.data() + 6).get());
    createWorld<<<1U, 1U>>>(
            dList,
            dWorld,
            dCamera,
            kWidth,
            kHeight
    );
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Render
    dim3 mGridDim = {
            (kWidth + kBlockDim.x - 1U) / kBlockDim.x,
            (kHeight + kBlockDim.y - 1U) / kBlockDim.y,
            1U
    };

    thrust::device_vector<glm::vec3> dFb(kWidth * kHeight, {1.0f, 1.0f, 1.0f});
    thrust::device_vector<curandState> dRandState(kWidth * kHeight * 3);

    // These two kernels are on the same stream (null stream).
    // Sequentiall execution is guaranteed by CUDA runtime.
    initRandState<<<mGridDim, kBlockDim>>>(
            dRandState.data().get(),
            kWidth,
            kHeight
    );
    CUDA_CHECK_LAST_ERROR();

    render<<<mGridDim, kBlockDim>>>(
            dFb.data().get(),
            kWidth,
            kHeight,
            kAntiAliasingFactor,
            dCamera,
            dWorld,
            dRandState.data().get()
    );
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Free resources
    freeWorld<<<1U, 1U>>>(
            dList,
            dWorld,
            dCamera
    );
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Display rendered image
    thrust::host_vector<glm::vec3> hFb = dFb;
    cv::Mat img(kHeight, kWidth, CV_32FC3, hFb.data());
    cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
    cv::flip(img, img, 0);
    cv::imshow("Chap 10 Camera", img);
    cv::waitKey();

    return EXIT_SUCCESS;
}
