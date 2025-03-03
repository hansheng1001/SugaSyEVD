
# include_directories(
#         ../QR/include
#         /home/szhang/cutlass/include/
#         /home/szhang/cutlass/tools/util/include
# )

file(GLOB CODE_SOURCES "*.cu")



foreach(SRC ${CODE_SOURCES})
    get_filename_component(NAME ${SRC} NAME_WE)
    message(${NAME})
    add_executable(${NAME} "${NAME}.cu")
    set_property(TARGET ${NAME}
            PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    
    target_include_directories(${NAME} PRIVATE ${CMAKE_SOURCE_DIR}/util/inc)
    target_include_directories(${NAME} PRIVATE ${CMAKE_SOURCE_DIR}/api/inc)
    target_include_directories(${NAME} PRIVATE ${CMAKE_SOURCE_DIR}/inc)

    target_link_libraries(${NAME} my_api_shared_lib)
    target_link_libraries(${NAME} curand cusolver cublas)

    # 判断 ${NAME} 是否等于 my_EVD_useLAPACKE_CompQ
    # message(STATUS "The value of NAME is: ${NAME}")
    if(${NAME} STREQUAL "my_EVD_useLAPACKE_CompQ")
        # message(STATUS "come here")
        # 添加私有头文件路径
        # target_include_directories(${NAME} PRIVATE "/home/wanghs/hpc/magma-github/include")
        # target_include_directories(${NAME} PRIVATE "/home/wanghs/hpc/magma-icl/magma-2.8.0/include")
        # target_include_directories(${NAME} PRIVATE "/home/wanghs/hpc/magma-2.7.2/include")

        # 添加库文件路径
        # link_directories("/home/wanghs/hpc/magma-2.8.0/lib")
        # target_link_directories(${NAME} PRIVATE "/home/wanghs/hpc/magma-github/build/lib")
        # target_link_directories(${NAME} PRIVATE "/home/wanghs/hpc/magma-icl/magma-2.8.0/lib")
        # target_link_directories(${NAME} PRIVATE "/home/wanghs/hpc/magma-2.7.2/lib")

        # 链接库文件
        # target_link_libraries(${NAME} magma mkl_gf_ilp64 mkl_gnu_thread mkl_core)
        # target_link_libraries(${NAME} magma)
        # target_link_libraries(${NAME} lapacke)
        #  -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl

        # target_include_directories(${NAME} PRIVATE "/opt/intel/oneapi/mkl/latest/include")
        # target_link_libraries(${NAME} mkl_gf_ilp64 mkl_gnu_thread mkl_core gomp)
        
        # get_target_property(LINK_DIRECTORIES ${NAME} INTERFACE_LINK_DIRECTORIES)
        # message(STATUS "Link directories for target ${NAME}: ${LINK_DIRECTORIES}")

    endif()
    
    # if(${NAME} STREQUAL "my_EVD_useLAPACKE")

    #     target_link_libraries(${NAME} lapacke)
        
    #     # get_target_property(LINK_DIRECTORIES ${NAME} INTERFACE_LINK_DIRECTORIES)
    #     # message(STATUS "Link directories for target ${NAME}: ${LINK_DIRECTORIES}")

    # endif()
endforeach()