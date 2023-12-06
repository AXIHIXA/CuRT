#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <opencv2/opencv.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "util/CudaUtil.h"


constexpr dim3 kBlockDim {32U, 32U, 1U};

constexpr int kWidth = 1200;

constexpr int kHeight = 600;


__global__
void render(glm::vec3 * __restrict__ fb, int width, int height)
{
    auto x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    auto y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);

    if (x < width and y < height)
    {
        fb[y * width + x] = {
                static_cast<float>(x) / static_cast<float>(width),
                static_cast<float>(y) / static_cast<float>(height),
                0.2f
        };
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

    render<<<mGridDim, kBlockDim>>>(dFb.data().get(), kWidth, kHeight);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());

    thrust::host_vector<glm::vec3> hFb = dFb;
    cv::Mat img(kHeight, kWidth, CV_32FC3, hFb.data());
    cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
    cv::flip(img, img, 0);
    cv::imshow("Chap 01 Output", img);
    cv::waitKey();

    return EXIT_SUCCESS;
}
