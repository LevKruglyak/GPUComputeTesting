//
//  main.swift
//  GPUComputeTesting
//
//  Created by Lev Kruglyak on 6/18/21.
//

import Foundation
import Metal

Metal.initialize()

Metal.capture {
    runParallelReduction()
}

//let counterBuffer = Buffer<Int32>(data: 0, label: "Counter", .storageModeShared);
//
//let kernel = KernelFunction(kernelName: "atomicCounter",
//                            debugLabel: "Atomic Counter",
//                            threadsPerThreadgroup: (1024, 1, 1),
//                            threadgroupsPerGrid: (1, 1, 1))
//
//kernel.addBuffer(counterBuffer)
//
//Metal.capture({
//
//    Metal.run("Main", {
//        kernel.dispatch()
//    })
//
//    print(counterBuffer.contents[0])
//
//})
