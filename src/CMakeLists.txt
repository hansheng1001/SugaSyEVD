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
# 编译出utils,编译为动态库
set(UTILS_SOURCES_DIR ${CMAKE_SOURCE_DIR}/util)
file(GLOB_RECURSE UTILS_SOURCES "${UTILS_SOURCES_DIR}/*.cpp" "${UTILS_SOURCES_DIR}/*.cu")
add_library(my_utils_shared_lib SHARED  ${UTILS_SOURCES})

# target_include_directories(my_utils_shared_lib PRIVATE ${CMAKE_SOURCE_DIR}/util/inc)
include_directories(${CMAKE_SOURCE_DIR}/util/inc)

target_link_libraries(my_utils_shared_lib cublas curand)

# 编译出api, 这个依赖uitls动态库, 也需要编译为动态库
set(API_SOURCES_DIR ${CMAKE_SOURCE_DIR}/api)
file(GLOB_RECURSE API_SOURCES "${API_SOURCES_DIR}/*.cpp" "${API_SOURCES_DIR}/*.cu")
message("${API_SOURCES}")
add_library(my_api_shared_lib SHARED ${API_SOURCES})

target_include_directories(my_api_shared_lib PRIVATE ${CMAKE_SOURCE_DIR}/util/inc)
target_include_directories(my_api_shared_lib PRIVATE ${CMAKE_SOURCE_DIR}/api/inc)
target_include_directories(my_api_shared_lib PRIVATE ${CMAKE_SOURCE_DIR}/inc)
target_link_libraries(my_api_shared_lib PUBLIC my_utils_shared_lib)


# 编译出fortran,编译为动态库
# set(UTILS_SOURCES_DIR ${CMAKE_SOURCE_DIR}/FortranC)
# file(GLOB_RECURSE UTILS_SOURCES "${UTILS_SOURCES_DIR}/*.cpp" "${UTILS_SOURCES_DIR}/*.cu")
# add_library(my_FortranC_shared_lib SHARED  ${UTILS_SOURCES})

# target_include_directories(my_FortranC_shared_lib PRIVATE ${CMAKE_SOURCE_DIR}/FortranC/inc)


add_subdirectory(EVD)
# add_subdirectory(panelQR)
# add_subdirectory(SVD)
# add_subdirectory(sy2sb)
# add_subdirectory(sy2sb_backTrans)
# add_subdirectory(sy2st)

# add_subdirectory(BC_backTrans)
