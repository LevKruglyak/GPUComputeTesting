//
//  ParallelReduction.metal
//  GPUComputeTesting
//
//  Created by Lev Kruglyak on 6/25/21.
//

#include <metal_stdlib>
using namespace metal;

#define THREAD_GROUP_SIZE 1024

/*
 Reduction using atomic operations
 */
kernel void reduction1(const device     uint &count             [[ buffer(0) ]],
                       const device     int *input              [[ buffer(1) ]],
                       device           atomic_int &output      [[ buffer(2) ]],
                                        uint threadLocalIndex  [[ thread_position_in_threadgroup ]],
                                        uint threadgroupSize    [[ threads_per_threadgroup ]]) {
                                
    uint index = threadLocalIndex;

    while (index < count) {
        atomic_fetch_add_explicit(&output, input[index], memory_order_relaxed);
        index += threadgroupSize;
    }
}

/*
 Simple parallel reduction
 */
kernel void reduction2(const device     uint &count             [[ buffer(0) ]],
                       const device     int *input              [[ buffer(1) ]],
                       device           int &output             [[ buffer(2) ]],
                                        uint threadLocalIndex   [[ thread_position_in_threadgroup ]],
                                        uint threadgroupSize    [[ threads_per_threadgroup ]]) {

    threadgroup int cache[THREAD_GROUP_SIZE];

    // Make sure allocated cache is cleared
    cache[threadLocalIndex] = 0;
    
    uint index = threadLocalIndex;

    while (index < count) {
        cache[threadLocalIndex] += input[index];
        index += threadgroupSize;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint cutoff = threadgroupSize / 2;

    while (cutoff != 0) {
        if (threadLocalIndex < cutoff) {
            cache[threadLocalIndex] += cache[threadLocalIndex + cutoff];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        cutoff /= 2;
    }

    if (threadLocalIndex == 0) {
        output = cache[0];
    }
}

/*
 Optimized parallel reduction
 */
kernel void reduction3(const device     uint &count             [[ buffer(0) ]],
                       const device     int *input              [[ buffer(1) ]],
                       device           int &output             [[ buffer(2) ]],
                       device           atomic_int *cache       [[ buffer(3) ]],
                                        uint threadGlobalIndex  [[ thread_position_in_grid ]],
                                        uint threadgroupSize    [[ threads_per_threadgroup ]]) {
    
    atomic_store_explicit(&cache[threadGlobalIndex], input[threadGlobalIndex], memory_order_relaxed);

    threadgroup_barrier(mem_flags::mem_device);
    
    for (uint cutoff = count / 2; cutoff > 0; cutoff >>= 1) {
        if (threadGlobalIndex < cutoff) {
            atomic_fetch_add_explicit(&cache[threadGlobalIndex], atomic_load_explicit(&cache[threadGlobalIndex + cutoff], memory_order_relaxed), memory_order_relaxed);
        }

        threadgroup_barrier(mem_flags::mem_device);
    }

    if (threadGlobalIndex == 0) {
        output = atomic_load_explicit(&cache[0], memory_order_relaxed);
    }
}
