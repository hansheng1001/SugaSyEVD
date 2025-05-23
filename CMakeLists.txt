cmake_minimum_required(VERSION 3.12)


# set(CMAKE_C_COMPILER "/opt/dtk/cuda/bin/nvcc")
# set(CMAKE_CXX_COMPILER "/opt/dtk/cuda/bin/nvcc")


# set(CMAKE_BUILD_TYPE "Debug")
# set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O3 -Wall -g -G -lineinfo -ggdb -DDEBUG")
# set(CMAKE_C_FLAGS_DEBUG "$ENV{CFLAGS} -O3 -Wall -g -G -lineinfo -ggdb -DDEBUG")

set(CMAKE_BUILD_TYPE "Release")

# set(CUDA_NVCC_FLAGS_DEBUG "${CUDA_NVCC_FLAGS_DEBUG} -g -G -lineinfo")

# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -g -G -lineinfo")

# 可用于使用ncu可以将ptx和汇编进行对应
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O0")

# set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -g  -G")
# set(CMAKE_C_FLAGS_DEBUG "$ENV{CFLAGS} -g -G")

# 使用下面这种方法才能够在编译中加入-G选项
# add_compile_options("-G")
# add_compile_options(-fuse-ld=/opt/dtk/cuda/bin/nvcc)
add_compile_options(--gpu-max-threads-per-block=1024)
add_compile_options(-O3)

# set(CMAKE_LINKER "/opt/dtk/cuda/bin/nvcc")
# # 指定链接器标志
# set(CMAKE_CXX_LINKER_FLAGS "${CMAKE_CXX_LINKER_FLAGS} -fuse-ld=/opt/dtk/cuda/bin/nvcc")

# set(CMAKE_EXE_LINKER_FLAGS_INIT "-fuse-ld=ldd")
# set(CMAKE_MODULE_LINKER_FLAGS_INIT "-fuse-ld=ldd")
# set(CMAKE_SHARED_LINKER_FLAGS_INIT "-fuse-ld=ldd")

# add_link_options("-fuse-ld=/opt/dtk/cuda/bin/nvcc")

set(CMAKE_VERBOSE_MAKEFILE ON)

# set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb -DDEBUG -Xcompiler -rdynamic -lineinfo")
# set(CMAKE_C_FLAGS_DEBUG "$ENV{CFLAGS} -O0 -Wall -g -ggdb -DDEBUG -Xcompiler -rdynamic -lineinfo")
# set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb -DDEBUG")
# set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
message("${CMAKE_CXX_FLAGS_DEBUG}")

# 配置Clangd进行跳转
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_USE_RESPONSE_FILE_FOR_OBJECTS OFF)
set(CMAKE_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
set(CMAKE_USE_RESPONSE_FILE_FOR_LIBRARIES OFF)
set(CMAKE_USE_RESPONSE_FILE_FOR_LINK_LIBRARIES OFF)
set(CMAKE_C_USE_RESPONSE_FILE_FOR_OBJECTS 0)
set(CMAKE_C_USE_RESPONSE_FILE_FOR_LIBRARIES 0)
set(CMAKE_CXX_USE_RESPONSE_FILE_FOR_OBJECTS 0)
set(CMAKE_CXX_USE_RESPONSE_FILE_FOR_INCLUDES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS 0)


set(CMAKE_VERBOSE_MAKEFILE ON)


#H100 90
#2880Ti 75
#4090 89
# A100 80

# set(CMAKE_CUDA_ARCHITECTURES 90)
# set(CMAKE_CUDA_ARCHITECTURES native)

# 在2.ndsl和1.ndsl中找不到编译器,需要进行设置.在87上要注释掉
# set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.6/bin/nvcc)
project(myqrcode LANGUAGES CXX CUDA)


# set(CMAKE_LINKER $ENV{CUDA_PATH}/bin/nvcc)
# set(CMAKE_CXX_LINK_EXECUTABLE "<CMAKE_LINKER> <FLAGS> <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")
set(CMAKE_CUDA_HOST_LINK_LAUNCHER $ENV{CUDA_PATH}/bin/nvcc)

# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -fgpu-rdc")

# 设置可以打印出核函数使用寄存器数和大小数的函数
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v -Xcompiler -rdynamic -lineinfo")

add_subdirectory(src)
# add_subdirectory(debug)
# add_subdirectory(benchmark)

# add_subdirectory(test)