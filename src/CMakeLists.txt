# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without modification, are permitted
# provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright notice, this list of
#       conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright notice, this list of
#       conditions and the following disclaimer in the documentation and/or other materials
#       provided with the distribution.
#     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
#       to endorse or promote products derived from this software without specific prior written
#       permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

###############################################################################
# Source files
###############################################################################

set(SOURCE_FILES
	common.cu
	common_device.cu
	cpp_api.cu
	cutlass_mlp.cu
	cutlass_resnet.cu
	encoding.cu
	loss.cu
	network.cu
	object.cu
	optimizer.cu
	reduce_sum.cu
)
set(INCLUDE_FILES
    ../include/tiny-cuda-nn/common.h
    ../include/tiny-cuda-nn/common_device.h
    ../include/tiny-cuda-nn/config.h
    ../include/tiny-cuda-nn/cpp_api.h
    ../include/tiny-cuda-nn/cuda_graph.h
    ../include/tiny-cuda-nn/cutlass_matmul.h
    ../include/tiny-cuda-nn/encoding.h
    ../include/tiny-cuda-nn/gpu_matrix.h
    ../include/tiny-cuda-nn/gpu_memory.h
    ../include/tiny-cuda-nn/gpu_memory_json.h
    ../include/tiny-cuda-nn/loss.h
    ../include/tiny-cuda-nn/matrix_layout.h
    ../include/tiny-cuda-nn/multi_stream.h
    ../include/tiny-cuda-nn/network.h
    ../include/tiny-cuda-nn/network_with_input_encoding.h
    ../include/tiny-cuda-nn/object.h
    ../include/tiny-cuda-nn/optimizer.h
    ../include/tiny-cuda-nn/random.h
    ../include/tiny-cuda-nn/reduce_sum.h
    ../include/tiny-cuda-nn/trainer.h
)
set(INCLUDE_FILES_ENCODINGS
    ../include/tiny-cuda-nn/encodings/composite.h
    ../include/tiny-cuda-nn/encodings/frequency.h
    ../include/tiny-cuda-nn/encodings/grid.h
    ../include/tiny-cuda-nn/encodings/identity.h
    ../include/tiny-cuda-nn/encodings/nrc.h
    ../include/tiny-cuda-nn/encodings/oneblob.h
    ../include/tiny-cuda-nn/encodings/spherical_harmonics.h
    ../include/tiny-cuda-nn/encodings/triangle_wave.h
)
set(INCLUDE_FILES_LOSSES
    ../include/tiny-cuda-nn/losses/cross_entropy.h
    ../include/tiny-cuda-nn/losses/l1.h
    ../include/tiny-cuda-nn/losses/l2.h
    ../include/tiny-cuda-nn/losses/mape.h
    ../include/tiny-cuda-nn/losses/relative_l1.h
    ../include/tiny-cuda-nn/losses/relative_l2.h
    ../include/tiny-cuda-nn/losses/relative_l2_luminance.h
    ../include/tiny-cuda-nn/losses/smape.h
    ../include/tiny-cuda-nn/losses/variance_is.h
)
set(INCLUDE_FILES_NETWORKS
    ../include/tiny-cuda-nn/networks/cutlass_mlp.h
    ../include/tiny-cuda-nn/networks/cutlass_resnet.h
    ../include/tiny-cuda-nn/networks/fully_fused_mlp.h
)
set(INCLUDE_FILES_OPTIMIZERS
    ../include/tiny-cuda-nn/optimizers/adam.h
    ../include/tiny-cuda-nn/optimizers/average.h
    ../include/tiny-cuda-nn/optimizers/batched.h
    ../include/tiny-cuda-nn/optimizers/ema.h
    ../include/tiny-cuda-nn/optimizers/exponential_decay.h
    ../include/tiny-cuda-nn/optimizers/lookahead.h
    ../include/tiny-cuda-nn/optimizers/novograd.h
    ../include/tiny-cuda-nn/optimizers/sgd.h
    ../include/tiny-cuda-nn/optimizers/shampoo.h
)

if (MIN_GPU_ARCH GREATER_EQUAL 70)
	list(APPEND SOURCE_FILES fully_fused_mlp.cu)
endif()

###############################################################################
# Linker / library
###############################################################################

add_library(tiny-cuda-nn STATIC 
    ${SOURCE_FILES} ${INCLUDE_FILES}
    ${INCLUDE_FILES_ENCODINGS}
    ${INCLUDE_FILES_LOSSES}
    ${INCLUDE_FILES_NETWORKS}
    ${INCLUDE_FILES_OPTIMIZERS})
target_link_libraries(tiny-cuda-nn PUBLIC ${CUDA_LIBRARIES} cuda cublas)
target_compile_options(tiny-cuda-nn PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)
target_include_directories(tiny-cuda-nn PUBLIC ${CMAKE_ROOT_DIR}/include)
target_include_directories(tiny-cuda-nn PUBLIC ${CMAKE_ROOT_DIR}/dependencies)

source_group("Header Files" FILES ${INCLUDE_FILES})
source_group("Header Files\\Encodings" FILES ${INCLUDE_FILES_ENCODINGS})
source_group("Header Files\\Losses" FILES ${INCLUDE_FILES_LOSSES})
source_group("Header Files\\Networks" FILES ${INCLUDE_FILES_NETWORKS})
source_group("Header Files\\Optimizers" FILES ${INCLUDE_FILES_OPTIMIZERS})

