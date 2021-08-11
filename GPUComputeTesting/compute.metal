//
//  compute.metal
//  GPUComputeTesting
//
//  Created by Lev Kruglyak on 6/18/21.
//

#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

kernel void atomicCounter(device atomic_int &counter [[ buffer(0) ]]) {
    
    atomic_fetch_add_explicit(&counter, 1, memory_order_relaxed);
}
