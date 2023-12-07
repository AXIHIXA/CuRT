#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <opencv2/opencv.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "ch03/Ray.cuh"
#include "util/CudaUtil.h"


constexpr dim3 kBlockDim {32U, 32U, 1U};

constexpr int kWidth = 1200;

constexpr int kHeight = 600;


__device__
glm::vec3 color(const Ray & r)
{
    float t = 0.5f * (r.d().y + 1.0f);
    return (1.0f - t) * glm::vec3(1.0f) + t * glm::vec3(0.5f, 0.7f, 1.0f);
}


__global__
void render(
        glm::vec3 * __restrict__ fb,
        int width,
        int height,
        glm::vec3 lowerLeft,
        glm::vec3 horizontal,
        glm::vec3 vertical,
        glm::vec3 origin
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
        fb[y * width + x] = color(r);
    }
}


int main(int argc, char * argv[])
{
    thrust::device_vector<glm::vec3> dFb(kWidth * kHeight, {1.0f, 1.0f, 1.0f});

    dim3 mGridDim = {
            (kWidth + kBlockDim.x - 1U) / kBlockDim.x,
            (kHeight + kBlockDim.y - 1U) / kBlockDim.y,
            1U
    };

    render<<<mGridDim, kBlockDim>>>(
            dFb.data().get(),
            kWidth,
            kHeight,
            glm::vec3(-2.0f, -1.0f, -1.0f),
            glm::vec3(4.0f, 0.0f, 0.0f),
            glm::vec3(0.0f, 2.0f, 0.0f),
            glm::vec3(0.0f, 0.0f, 0.0f)
    );
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());

    thrust::host_vector<glm::vec3> hFb = dFb;
    cv::Mat img(kHeight, kWidth, CV_32FC3, hFb.data());
    cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
    cv::flip(img, img, 0);
    cv::imshow("Chap 03 Ray", img);
    cv::waitKey();

    return EXIT_SUCCESS;
}
