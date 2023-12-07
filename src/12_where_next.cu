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
#include "ch11/Camera.cuh"
#include "util/CudaUtil.h"


constexpr dim3 kBlockDim {32U, 32U, 1U};

constexpr int kWidth = 1200;

constexpr int kHeight = 800;

constexpr float kFloatMax = std::numeric_limits<float>::max();

constexpr float kHitEps = 1e-3f;

constexpr int kMaxRecursionLevel = 50;

constexpr int kNumSpheres = 22 * 22 + 1 + 3;

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
        int height,
        curandState * __restrict__ randState
)
{
    curandState localRandState = *randState;

    list[0] = new Sphere({0.0f, -1000.0f, -1.0f}, 1000.0f, new Lambertian({0.5f, 0.5f, 0.5f}));

    int i = 0;

    for (int a = -11; a < 11; ++a)
    {
        for (int b = -11; b < 11; ++b)
        {
            float randomMaterial = curand_uniform(&localRandState);

            glm::vec3 center = {
                    static_cast<float>(a) + curand_uniform(&localRandState),
                    0.2f,
                    static_cast<float>(b) + curand_uniform(&localRandState)
            };

            if (randomMaterial < 0.8f)
            {
                list[++i] = new Sphere(
                        center,
                        0.2f,
                        new Lambertian(
                                {
                                        curand_uniform(&localRandState) * curand_uniform(&localRandState),
                                        curand_uniform(&localRandState) * curand_uniform(&localRandState),
                                        curand_uniform(&localRandState) * curand_uniform(&localRandState)
                                })
                );
            }
            else if (randomMaterial < 0.95f)
            {
                list[++i] = new Sphere(
                        center,
                        0.2f,
                        new Metal(
                                {
                                        0.5f * (1.0f + curand_uniform(&localRandState)),
                                        0.5f * (1.0f + curand_uniform(&localRandState)),
                                        0.5f * (1.0f + curand_uniform(&localRandState))
                                },
                                0.5f * curand_uniform(&localRandState)
                        )

                );
            }
            else
            {
                list[++i] = new Sphere(center, 0.2f, new Dielectric(1.5f));
            }
        }
    }

    list[++i] = new Sphere({0.0f, 1.0f, 0.0f}, 1.0f, new Dielectric(1.5f));
    list[++i] = new Sphere({-4.0f, 1.0f, 0.0f}, 1.0f, new Lambertian({0.4f, 0.2f, 0.1f}));
    list[++i] = new Sphere({4.0f, 1.0f, 0.0f}, 1.0f, new Metal({0.7f, 0.6f, 0.5f}, 0.0f));

    *randState = localRandState;

    *world = new World(list, kNumSpheres);

    glm::vec3 lookfrom = {13.0f, 2.0f, 3.0f};
    glm::vec3 lookat = {0.0f, 0.0f, 0.0f};
    float focusDistance = 10.0f;
    float aperture = 0.1f;
    *camera = new Camera(
            lookfrom,
            lookat,
            {0.0f, 1.0f, 0.0f},
            30.0f,
            static_cast<float>(width) / static_cast<float>(height),
            aperture,
            focusDistance
    );
}


__global__
void freeWorld(
        Hitable ** __restrict__ list,
        Hitable ** __restrict__ world,
        Camera ** camera
)
{
    for (int i = 0; i != kNumSpheres; ++i)
    {
        delete reinterpret_cast<Sphere *>(list[i])->pMaterial;
        delete list[i];
    }

    delete *world;
    delete *camera;
}


__global__
void initOneRandState(
        curandState * __restrict__ randState
)
{
    curand_init(1984, 0, 0, randState);
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
        // Old version is NOT optimal, see
        // https://docs.nvidia.com/cuda/curand/device-api-overview.html#performance-notes
        // Each thread gets same seed, a different sequence number, no offset.
        // curand_init(1984, y * width + x, 0, randState + y * width + x);

        // Each thread gets a different seed, the same sequence, no offset.
        // Says the documentation above:
        // Initialization of the random generator state generally requires
        // more registers and local memory than random number generation.
        // It may be beneficial to separate calls to curand_init() and curand()
        // into separate kernels for maximum performance.
        // State setup can be an expensive operation.
        // One way to speed up the setup is to
        // use different seeds for each thread and a constant sequence number of 0.
        // This can be especially helpful if many generators need to be created.

        int pixelIdx = y * width + x;
        curand_init(1984 + pixelIdx, 0, 0, randState + pixelIdx);
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
            tmpColor += color((*camera)->castRayAt(u, v, &localRandState), world, &localRandState);
        }

        randState[pixelIdx] = localRandState;

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
    thrust::device_vector<void *> dPtrPool(kNumSpheres + 2);
    Hitable ** dList = reinterpret_cast<Hitable **>(dPtrPool.data().get());
    Hitable ** dWorld = dList + kNumSpheres;
    Camera ** dCamera = reinterpret_cast<Camera **>((dPtrPool.data() + kNumSpheres + 1).get());

    {
        thrust::device_vector<curandState> worldCreationRandState(1);
        initOneRandState<<<1U, 1U>>>(worldCreationRandState.data().get());
        CUDA_CHECK_LAST_ERROR();

        createWorld<<<1U, 1U>>>(
                dList,
                dWorld,
                dCamera,
                kWidth,
                kHeight,
                worldCreationRandState.data().get()
        );
        CUDA_CHECK_LAST_ERROR();
        CUDA_CHECK(cudaDeviceSynchronize());
    }

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
    cv::imshow("Chap 12 Where Next", img);
    cv::waitKey();

    return EXIT_SUCCESS;
}
