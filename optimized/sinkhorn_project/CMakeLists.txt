cmake_minimum_required(VERSION 3.16)

find_package(ASC REQUIRED)
project(sinkhorn_ops LANGUAGES ASC CXX)
set(CMAKE_CXX_STANDARD 17)

set(ACL_INCLUDE_DIR "$ENV{ASCEND_HOME_PATH}/aarch64-linux/include")
set(ASC_INCLUDE_DIR "$ENV{ASCEND_HOME_PATH}/aarch64-linux/asc/include")
set(ACL_LIB_DIR "$ENV{ASCEND_HOME_PATH}/lib64")
set(ASC_LIB_DIR "$ENV{ASCEND_HOME_PATH}/aarch64-linux/lib64")

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
)
find_package(Torch REQUIRED HINTS ${TORCH_CMAKE_PREFIX_PATH})

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import os, torch_npu; print(os.path.dirname(torch_npu.__file__))"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE TORCH_NPU_PATH
)
set(TORCH_NPU_INCLUDE_DIRS ${TORCH_NPU_PATH}/include)
set(TORCH_NPU_LIBRARIES ${TORCH_NPU_PATH}/lib)

# 这里换成了我们 sinkhorn 的文件
add_library(sinkhorn_ops SHARED
    op_kernel/sinkhorn_kernel.asc
    op_extension/sinkhorn_torch.cpp
)

target_include_directories(sinkhorn_ops PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/op_extension
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_NPU_INCLUDE_DIRS}
    ${ACL_INCLUDE_DIR}
    ${ASC_INCLUDE_DIR}
    ${Python3_INCLUDE_DIRS}
)

target_link_libraries(sinkhorn_ops PRIVATE
    torch_npu tiling_api platform unified_dlog dl m graph_base ascendcl ascendc_runtime
)

target_link_directories(sinkhorn_ops PRIVATE
    ${ACL_LIB_DIR} ${ASC_LIB_DIR} ${TORCH_NPU_LIBRARIES}
)

target_compile_options(sinkhorn_ops PRIVATE
    ${TORCH_CXX_FLAGS}
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
)

# The huawei bisheng toolchain exports C++-mangled kernel stubs; the contest
# server toolchain exports C linkage.  Gate the declaration in sinkhorn_torch.cpp.
option(SINKHORN_KERNEL_CXX_LINKAGE "Declare kernel entry point with C++ linkage" OFF)
if(SINKHORN_KERNEL_CXX_LINKAGE)
    target_compile_definitions(sinkhorn_ops PRIVATE SINKHORN_KERNEL_CXX_LINKAGE)
endif()

message(STATUS "TORCH_LIBRARY module 'libsinkhorn_ops.so' will be built")
