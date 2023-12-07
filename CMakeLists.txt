cmake_minimum_required(VERSION 3.23)
project(CuRT CUDA CXX)

# Ensure correct CUDA architecture
# NVIDIA Geforce RTX 2080 Ti has Compute Capability 7.5
# https://developer.nvidia.com/cuda-gpus
# https://stackoverflow.com/questions/67794606/cmake-cuda-architecture-flags
set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_VERBOSE_MAKEFILE ON)

find_package(glm REQUIRED)
find_package(OpenCV REQUIRED)

set(UTIL
        include/util/CudaUtil.h
        include/util/TimerGuard.h
)

set(ALL_INCLUDE_DIRS
        ${OpenCV_INCLUDE_DIRS}
        ${GLM_INCLUDE_DIRS}
        include
)

set(ALL_LIBRARIES
        ${OpenCV_LIBS}
        cudart
        curand
)

set(ALL_COMPILE_DEFS
        -DGLM_ENFORCE_CUDA
)

set(ALL_COMPILE_OPTS
        $<$<CONFIG:DEBUG>:-O0>
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:DEBUG>>:-G>
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)

set(01_OUTPUT 01_output)
add_executable(${01_OUTPUT}
        ${UTIL}
        src/01_output.cu
)
target_compile_definitions(${01_OUTPUT} PUBLIC ${ALL_COMPILE_DEFS})
target_compile_options(${01_OUTPUT} PUBLIC ${ALL_COMPILE_OPTS})
target_include_directories(${01_OUTPUT} PUBLIC ${ALL_INCLUDE_DIRS})
target_link_libraries(${01_OUTPUT} ${ALL_LIBRARIES})

set(03_RAY 03_ray)
add_executable(${03_RAY}
        ${UTIL}
        include/ch03/Ray.cuh
        src/03_ray.cu
)
target_compile_definitions(${03_RAY} PUBLIC ${ALL_COMPILE_DEFS})
target_compile_options(${03_RAY} PUBLIC ${ALL_COMPILE_OPTS})
target_include_directories(${03_RAY} PUBLIC ${ALL_INCLUDE_DIRS})
target_link_libraries(${03_RAY} ${ALL_LIBRARIES})

set(04_SPHERE 04_sphere)
add_executable(${04_SPHERE}
        ${UTIL}
        include/ch03/Ray.cuh
        src/04_sphere.cu
)
target_compile_definitions(${04_SPHERE} PUBLIC ${ALL_COMPILE_DEFS})
target_compile_options(${04_SPHERE} PUBLIC ${ALL_COMPILE_OPTS})
target_include_directories(${04_SPHERE} PUBLIC ${ALL_INCLUDE_DIRS})
target_link_libraries(${04_SPHERE} ${ALL_LIBRARIES})

set(05_NORMAL 05_normal)
add_executable(${05_NORMAL}
        ${UTIL}
        include/ch03/Ray.cuh
        include/ch05/Hitable.cuh
        include/ch05/Sphere.cuh
        include/ch05/World.cuh
        src/05_normal.cu
)
target_compile_definitions(${05_NORMAL} PUBLIC ${ALL_COMPILE_DEFS})
target_compile_options(${05_NORMAL} PUBLIC ${ALL_COMPILE_OPTS})
target_include_directories(${05_NORMAL} PUBLIC ${ALL_INCLUDE_DIRS})
target_link_libraries(${05_NORMAL} ${ALL_LIBRARIES})

set(06_ANTIALIASING 06_antialiasing)
add_executable(${06_ANTIALIASING}
        ${UTIL}
        include/ch03/Ray.cuh
        include/ch05/Hitable.cuh
        include/ch05/Sphere.cuh
        include/ch05/World.cuh
        include/ch06/Camera.cuh
        src/06_antialiasing.cu
)
target_compile_definitions(${06_ANTIALIASING} PUBLIC ${ALL_COMPILE_DEFS})
target_compile_options(${06_ANTIALIASING} PUBLIC ${ALL_COMPILE_OPTS})
target_include_directories(${06_ANTIALIASING} PUBLIC ${ALL_INCLUDE_DIRS})
target_link_libraries(${06_ANTIALIASING} ${ALL_LIBRARIES})

set(07_DIFFUSE 07_diffuse)
add_executable(${07_DIFFUSE}
        ${UTIL}
        include/ch03/Ray.cuh
        include/ch05/Hitable.cuh
        include/ch05/Sphere.cuh
        include/ch05/World.cuh
        include/ch06/Camera.cuh
        src/07_diffuse.cu
)
target_compile_definitions(${07_DIFFUSE} PUBLIC ${ALL_COMPILE_DEFS})
target_compile_options(${07_DIFFUSE} PUBLIC ${ALL_COMPILE_OPTS})
target_include_directories(${07_DIFFUSE} PUBLIC ${ALL_INCLUDE_DIRS})
target_link_libraries(${07_DIFFUSE} ${ALL_LIBRARIES})

set(08_METAL 08_metal)
add_executable(${08_METAL}
        ${UTIL}
        include/ch03/Ray.cuh
        include/ch06/Camera.cuh
        include/ch08/Hitable.cuh
        include/ch08/Material.cuh
        include/ch08/Sphere.cuh
        include/ch08/World.cuh
        src/08_metal.cu
)
target_compile_definitions(${08_METAL} PUBLIC ${ALL_COMPILE_DEFS})
target_compile_options(${08_METAL} PUBLIC ${ALL_COMPILE_OPTS})
target_include_directories(${08_METAL} PUBLIC ${ALL_INCLUDE_DIRS})
target_link_libraries(${08_METAL} ${ALL_LIBRARIES})

set(09_DIELECTRIC 09_dielectric)
add_executable(${09_DIELECTRIC}
        ${UTIL}
        include/ch03/Ray.cuh
        include/ch06/Camera.cuh
        include/ch08/Hitable.cuh
        include/ch08/Material.cuh
        include/ch08/Sphere.cuh
        include/ch08/World.cuh
        src/09_dielectric.cu
)
target_compile_definitions(${09_DIELECTRIC} PUBLIC ${ALL_COMPILE_DEFS})
target_compile_options(${09_DIELECTRIC} PUBLIC ${ALL_COMPILE_OPTS})
target_include_directories(${09_DIELECTRIC} PUBLIC ${ALL_INCLUDE_DIRS})
target_link_libraries(${09_DIELECTRIC} ${ALL_LIBRARIES})

set(10_CAMERA 10_camera)
add_executable(${10_CAMERA}
        ${UTIL}
        include/ch03/Ray.cuh
        include/ch08/Hitable.cuh
        include/ch08/Material.cuh
        include/ch08/Sphere.cuh
        include/ch08/World.cuh
        include/ch10/Camera.cuh
        src/10_camera.cu
)
target_compile_definitions(${10_CAMERA} PUBLIC ${ALL_COMPILE_DEFS})
target_compile_options(${10_CAMERA} PUBLIC ${ALL_COMPILE_OPTS})
target_include_directories(${10_CAMERA} PUBLIC ${ALL_INCLUDE_DIRS})
target_link_libraries(${10_CAMERA} ${ALL_LIBRARIES})

set(11_FOCUS_BLUR 11_focus_blur)
add_executable(${11_FOCUS_BLUR}
        ${UTIL}
        include/ch03/Ray.cuh
        include/ch08/Hitable.cuh
        include/ch08/Material.cuh
        include/ch08/Sphere.cuh
        include/ch08/World.cuh
        include/ch11/Camera.cuh
        src/11_focus_blur.cu
)
target_compile_definitions(${11_FOCUS_BLUR} PUBLIC ${ALL_COMPILE_DEFS})
target_compile_options(${11_FOCUS_BLUR} PUBLIC ${ALL_COMPILE_OPTS})
target_include_directories(${11_FOCUS_BLUR} PUBLIC ${ALL_INCLUDE_DIRS})
target_link_libraries(${11_FOCUS_BLUR} ${ALL_LIBRARIES})

set(12_WHERE_NEXT 12_where_next)
add_executable(${12_WHERE_NEXT}
        ${UTIL}
        include/ch03/Ray.cuh
        include/ch08/Hitable.cuh
        include/ch08/Material.cuh
        include/ch08/Sphere.cuh
        include/ch08/World.cuh
        include/ch11/Camera.cuh
        src/12_where_next.cu
)
target_compile_definitions(${12_WHERE_NEXT} PUBLIC ${ALL_COMPILE_DEFS})
target_compile_options(${12_WHERE_NEXT} PUBLIC ${ALL_COMPILE_OPTS})
target_include_directories(${12_WHERE_NEXT} PUBLIC ${ALL_INCLUDE_DIRS})
target_link_libraries(${12_WHERE_NEXT} ${ALL_LIBRARIES})

#set(21_MOTION_BLUR 21_motion_blur)
#add_executable(${21_MOTION_BLUR}
#        ${UTIL}
#        include/ch21/Ray.cuh
#        include/ch08/Hitable.cuh
#        include/ch08/Material.cuh
#        include/ch08/Sphere.cuh
#        include/ch08/World.cuh
#        include/ch11/Camera.cuh
#        src/21_motion_blur.cu
#)
#target_compile_definitions(${21_MOTION_BLUR} PUBLIC ${ALL_COMPILE_DEFS})
#target_compile_options(${21_MOTION_BLUR} PUBLIC ${ALL_COMPILE_OPTS})
#target_include_directories(${21_MOTION_BLUR} PUBLIC ${ALL_INCLUDE_DIRS})
#target_link_libraries(${21_MOTION_BLUR} ${ALL_LIBRARIES})









