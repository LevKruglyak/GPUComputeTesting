//
//  MetalUtils.swift
//  GPUComputeTesting
//
//  Created by Lev Kruglyak on 6/24/21.
//

import Cocoa
import MetalKit

class Stopwatch {
    private var startTime = DispatchTime.now()
    private var stopTime = DispatchTime.now()
    
    func start() {
        startTime = DispatchTime.now()
    }
    
    func stop() {
        stopTime = DispatchTime.now()
    }
    
    func getMs() -> Double {
        let nanoTime = stopTime.uptimeNanoseconds - startTime.uptimeNanoseconds
        return Double(nanoTime) / 1_000_000
    }
}

class Metal {

    fileprivate static var device: MTLDevice! = nil
    fileprivate static var queue: MTLCommandQueue! = nil
    fileprivate static var library: MTLLibrary! = nil
    fileprivate static var buffer: MTLCommandBuffer! = nil
    
    fileprivate static var debug: Bool = false;
    
    class func initialize() {
        /// Create the reference to the GPU
        self.device = MTLCreateSystemDefaultDevice()
        self.queue = device.makeCommandQueue()
        self.library = device.makeDefaultLibrary()
    }
    
    class func setDebug() {
        self.debug = true;
    }
    
    class func run(_ label: String, _ action: () -> Void) {
        buffer = queue.makeCommandBuffer()
        buffer.label = label
        
        action()
        
        buffer.commit()
        buffer.waitUntilCompleted()
        buffer = nil
    }
    
    class func capture(_ action: () -> Void) {
        if (!debug) {
            action()
        } else {
            let captureManager = MTLCaptureManager.shared()
            let captureDescriptor = MTLCaptureDescriptor()
            
            captureDescriptor.captureObject = device
            
            do {
                try captureManager.startCapture(with: captureDescriptor)
            
                action()
            
                captureManager.stopCapture()
            } catch {
                print("Error capturing GPU frame \(error)")
            }
        }
    }
}

class KernelInput {
    var buffers: [MTLBuffer]
    
    init() {
        self.buffers = []
    }
    
    init(_ buffers: [MTLBuffer]) {
        self.buffers = buffers
    }
    
    func addBuffer<T>(_ buffer: Buffer<T>) {
        buffers.append(buffer.MTLBuffer);
    }
}

/*
 Simple wrapper for a compute function
 */
class Kernel {
    
    private var computePipelineState: MTLComputePipelineState!
    
    private let kernelName: String
    private let debugLabel: String
    
    private var input: KernelInput
    
    private let threadsPerThreadgroup: MTLSize
    private let threadgroupsPerGrid: MTLSize
    
    init(kernelName: String, debugLabel: String, threadsPerThreadgroup: (Int, Int, Int), threadgroupsPerGrid: (Int, Int, Int), input: KernelInput = KernelInput()) {
        self.kernelName = kernelName
        self.debugLabel = debugLabel
        
        self.threadsPerThreadgroup = MTLSizeMake(threadsPerThreadgroup.0, threadsPerThreadgroup.1, threadsPerThreadgroup.2);
        self.threadgroupsPerGrid = MTLSizeMake(threadgroupsPerGrid.0, threadgroupsPerGrid.1, threadgroupsPerGrid.2);
        
        self.computePipelineState = nil;
        
        do {
            if let compute = Metal.library.makeFunction(name: kernelName) {
                self.computePipelineState = try Metal.device.makeComputePipelineState(function: compute)
            }
        } catch {
            print("Failed to create compute pipeline state")
        }

        self.input = input
    }
    
    func addBuffer<T>(_ buffer: Buffer<T>) {
        input.addBuffer(buffer)
    }
    
    func dispatch() {
        if (Metal.buffer != nil) {
            // Create compute encoder
            if let computeEncoder = Metal.buffer.makeComputeCommandEncoder() {
                computeEncoder.setComputePipelineState(self.computePipelineState)
                computeEncoder.label = self.debugLabel
                
                // Set buffers
                for index in 0..<input.buffers.count {
                    computeEncoder.setBuffer(input.buffers[index], offset: 0, index: index)
                }
                
                // Dispatch the threadgroups
                computeEncoder.dispatchThreadgroups(self.threadgroupsPerGrid, threadsPerThreadgroup: self.threadsPerThreadgroup)
                computeEncoder.endEncoding()
                
            } else {
                print("Failed to create compute encoder")
            }
        } else {
            print("Missing command buffer")
        }
    }
}

/*
 Simple wrapper for MTLBuffer object
 */
class Buffer<T> {
    
    fileprivate let MTLBuffer: MTLBuffer!
    
    public var contents: UnsafeMutablePointer<T>! = nil
    public let count: Int
    
    /*
     Default constructor
     */
    init(_ MTLBuffer: MTLBuffer) {
        self.MTLBuffer = MTLBuffer
        self.count = MTLBuffer.length / MemoryLayout<T>.size
        
        fillContents()
    }
    
    /*
     Initialize with single value
     */
    init(data: T, label: String, _ resourceOptions: MTLResourceOptions) {
        self.count = 1
        
        var bufferData = data;
        
        self.MTLBuffer = Metal.device.makeBuffer(bytes: &bufferData, length: MemoryLayout<T>.size, options: resourceOptions)
        self.MTLBuffer.label = label;
        
        fillContents()
    }
    
    /*
     Initialize with array full of values
     */
    init(data: ContiguousArray<T>, label: String, _ resourceOptions: MTLResourceOptions) {
        self.count = data.count
        
        self.MTLBuffer = data.withUnsafeBytes { (bufferPointer) -> MTLBuffer? in
            guard let baseAddress = bufferPointer.baseAddress else { return nil }
            return Metal.device.makeBuffer(bytes: baseAddress, length: bufferPointer.count, options: resourceOptions)
        }
        self.MTLBuffer.label = label
        
        fillContents()
    }
    
    /*
     Initialize empty with a given count
     */
    init(count: Int, label: String, _ resourceOptions: MTLResourceOptions) {
        self.count = count
        
        self.MTLBuffer = Metal.device.makeBuffer(length: MemoryLayout<T>.size * count, options: resourceOptions)
        self.MTLBuffer.label = label
        
        fillContents()
    }
    
    func sync() {
        if (self.MTLBuffer.resourceOptions == .storageModeManaged) {
            self.MTLBuffer.didModifyRange(0..<self.MTLBuffer.length)
        }
    }
    
    private func fillContents() {
        //print(self.MTLBuffer.resourceOptions)
        if (self.MTLBuffer.resourceOptions != .storageModePrivate) {
            contents = UnsafeMutableRawPointer(self.MTLBuffer.contents()).bindMemory(to:T.self, capacity: count)
        }
    }
}
