//
//  ParallelReduction.swift
//  GPUComputeTesting
//
//  Created by Lev Kruglyak on 6/25/21.
//

import Foundation

let sizeOfArray: Int = 1024 * 128
let compareToCPU: Bool = false;

func runParallelReduction() {
    Metal.setDebug()
    Metal.capture {
        
        let watch = Stopwatch()
        
        // Generate array
        var correctSum: Int32 = 0
        var inputArray = ContiguousArray<Int32>(repeating: 0, count: sizeOfArray)
        (0..<sizeOfArray).forEach({
            inputArray[$0] = 1;//Int32.random(in: 0...1)
            correctSum += inputArray[$0]
        })
        
        if (compareToCPU) {
            // Measure time to add on CPU
            watch.start()
            
            var cpuSum: Int32 = 0
            for index in 0..<sizeOfArray {
                cpuSum += inputArray[index]
            }
            
            watch.stop()
            
            print("CPU Reduction: \(watch.getMs()) ms")
        }
        
        // Generate buffers
        let countBuffer = Buffer<UInt32>(data: UInt32(sizeOfArray), label: "Count", .storageModeManaged)
        let cacheBuffer = Buffer<Int32>(count: sizeOfArray, label: "Cache", .storageModeManaged)
        let inputBuffer = Buffer<Int32>(data: inputArray, label: "Input", .storageModeManaged)
        let outputBuffer = Buffer<Int32>(data: 0, label: "Output", .storageModeShared);
            
        // Kernel 1
        var kernelIndex = 1
        
        var kernelInput = KernelInput()
        
        kernelInput.addBuffer(countBuffer)
        kernelInput.addBuffer(inputBuffer)
        kernelInput.addBuffer(outputBuffer)
        
        // Reset buffers
        outputBuffer.contents[0] = 0
        
        var reductionKernel = Kernel(kernelName: "reduction\(kernelIndex)",
                                     debugLabel: "Reduction \(kernelIndex)",
                                     threadsPerThreadgroup: (1024, 1, 1),
                                     threadgroupsPerGrid: (1, 1, 1),
                                     input: kernelInput)

        watch.start()
        
        Metal.run("Reduction \(kernelIndex)", {
            reductionKernel.dispatch()
        })
        
        watch.stop()
        
        if (correctSum != outputBuffer.contents[0]) {
            print("Reduction \(kernelIndex) failed, expected \(correctSum) recieved \(outputBuffer.contents[0])")
        }
        
        print("Reduction \(kernelIndex): \(watch.getMs()) ms")
        
        // Kernel 2
        kernelIndex = 2
        
        kernelInput = KernelInput()
        
        kernelInput.addBuffer(countBuffer)
        kernelInput.addBuffer(inputBuffer)
        kernelInput.addBuffer(outputBuffer)
        
        // Reset buffers
        outputBuffer.contents[0] = 0
        
        reductionKernel = Kernel(kernelName: "reduction\(kernelIndex)",
                                     debugLabel: "Reduction \(kernelIndex)",
                                     threadsPerThreadgroup: (1024, 1, 1),
                                     threadgroupsPerGrid: (1, 1, 1),
                                     input: kernelInput)

        watch.start()
        
        Metal.run("Reduction \(kernelIndex)", {
            reductionKernel.dispatch()
        })
        
        watch.stop()
        
        if (correctSum != outputBuffer.contents[0]) {
            print("Reduction \(kernelIndex) failed, expected \(correctSum) recieved \(outputBuffer.contents[0])")
        }
        
        print("Reduction \(kernelIndex): \(watch.getMs()) ms")
        
        // Kernel 3
        kernelIndex = 3
        
        kernelInput = KernelInput()
        
        kernelInput.addBuffer(countBuffer)
        kernelInput.addBuffer(inputBuffer)
        kernelInput.addBuffer(outputBuffer)
        kernelInput.addBuffer(cacheBuffer)
        
        // Reset buffers
        outputBuffer.contents[0] = 0
        
        reductionKernel = Kernel(kernelName: "reduction\(kernelIndex)",
                                     debugLabel: "Reduction \(kernelIndex)",
                                     threadsPerThreadgroup: (1024, 1, 1),
                                     threadgroupsPerGrid: (sizeOfArray / 1024, 1, 1),
                                     input: kernelInput)

        watch.start()
        
        Metal.run("Reduction \(kernelIndex)", {
            reductionKernel.dispatch()
        })
        
        watch.stop()
        
        if (correctSum != outputBuffer.contents[0]) {
            print("Reduction \(kernelIndex) failed, expected \(correctSum) recieved \(outputBuffer.contents[0])")
        }
        
        print("Reduction \(kernelIndex): \(watch.getMs()) ms")
    }
}
