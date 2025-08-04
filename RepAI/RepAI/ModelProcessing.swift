import Foundation
import Accelerate // For vDSP_conv, vDSP_vsub, etc.
import CoreML     // For MLMultiArray

// MARK: - Helper Functions

func mlMultiArrayToArray(_ multiArray: MLMultiArray) -> [Float] {
    let length = multiArray.count
    var result = [Float](repeating: 0.0, count: length)

    if multiArray.dataType == .float32 {
        let ptr = multiArray.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<length {
            result[i] = ptr[i]
        }
    } else if multiArray.dataType == .float16 {
        let ptr = multiArray.dataPointer.assumingMemoryBound(to: Float16.self)
        for i in 0..<length {
            result[i] = Float(ptr[i]) // Convert Float16 to Float32
        }
    } else {
        print("Warning: mlMultiArrayToArray received unsupported data type: \(multiArray.dataType)")
    }
    return result
}

func arrayToMLMultiArray(_ array: [Float], shape: [NSNumber]) -> MLMultiArray? {
    do {
        let multiArray = try MLMultiArray(shape: shape, dataType: .float32)
        let ptr = multiArray.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<array.count {
            ptr[i] = array[i]
        }
        return multiArray
    } catch {
        print("Error converting array to MLMultiArray: \(error.localizedDescription)")
        return nil
    }
}

func diff(_ array: [Float]) -> [Float] {
    guard array.count > 1 else { return [] }
    var result = [Float](repeating: 0.0, count: array.count - 1)
    for i in 0..<array.count - 1 {
        result[i] = array[i+1] - array[i]
    }
    return result
}

func findIndicesStart(in array: [Float]) -> [Int] {
    var indices: [Int] = []
    for (index, value) in array.enumerated() {
        if value > 0 {
            indices.append(index)
        }
    }
    return indices
}

func findIndicesEnd(in array: [Float]) -> [Int] {
    var indices: [Int] = []
    for (index, value) in array.enumerated() {
        if value < 0 {
            indices.append(index)
        }
    }
    return indices
}

// MARK: - get_time_in_reps_swift Function

func get_time_in_reps_swift(predsMultiArray: MLMultiArray, stride: Int, winsize: Int) -> MLMultiArray? {
    let timestamp = Int(Date().timeIntervalSince1970)
    let predictionExportManager = PredictionExportManager()
    // predsMultiArray comes from ConvNet's output, which is (T, L) now (since N=1)
    
    guard predsMultiArray.shape.count == 2 else {
        print("Error: predsMultiArray expected 2 dimensions (T, L). Got \(predsMultiArray.shape.count).")
        return nil
    }
    
    let current_T_dim = predsMultiArray.shape[0].intValue // This is T (should be SEQUENCE_LENGTH)
    let L_dim = predsMultiArray.shape[1].intValue // This is L (should be WINDOW_LENGTH)

    // These constants (SEQUENCE_LENGTH, WINDOW_LENGTH, FIXED_N_BATCH_SIZE, KERNEL_SIZE)
    // are expected to be defined as global constants in SharedSessionManager.swift
    // and will be accessible here.
    guard current_T_dim == SEQUENCE_LENGTH else {
        print("Error: predsMultiArray T dimension mismatch. Expected \(SEQUENCE_LENGTH), got \(current_T_dim).")
        return nil
    }
    guard L_dim == WINDOW_LENGTH else {
        print("Error: predsMultiArray L dimension mismatch. Expected \(WINDOW_LENGTH), got \(L_dim).")
        return nil
    }
    
    predictionExportManager.exportFloats(data: predsMultiArray, filename: "predsMultiArray_\(timestamp).txt")

    let predsFlatArray = mlMultiArrayToArray(predsMultiArray)
    
    var consolidatedi: [Float] = []
    for t_idx in 0..<current_T_dim { // Loop over T time steps
        let timeStart = t_idx * L_dim
        let predi = Array(predsFlatArray[timeStart..<(timeStart + L_dim)])
        let upper = (t_idx < current_T_dim - 1) ? stride : L_dim
        consolidatedi.append(contentsOf: predi[0..<upper])
    }
    
    do {
        let outputArray = try MLMultiArray(consolidatedi)
        predictionExportManager.exportFloats(data: outputArray, filename: "consolidatediNoConv_\(timestamp).txt")
    } catch {
        print("Broke")
    }

    let kernel: [Float] = [Float](repeating: 1.0 / Float(KERNEL_SIZE), count: 100) // Use KERNEL_SIZE constant
    var consolidatediRounded: [Float] = [Float](repeating: 0.0, count: consolidatedi.count)
    
    let startingZeros: [Float] = [Float](repeating: 0.0, count: 50)
    consolidatedi = startingZeros + consolidatedi + startingZeros
    consolidatediRounded = vDSP.convolve(consolidatedi, withKernel: kernel)
    for i: Int in 0..<consolidatediRounded.count {
        if consolidatediRounded[i] == 0.5 {
            print("Found equal value")
            consolidatediRounded[i] = 0
        }
    }
    print("Conv performed")
    
    consolidatediRounded = consolidatediRounded.map { $0.rounded() }
    
    do {
        let outputArray = try MLMultiArray(consolidatediRounded)
        predictionExportManager.exportFloats(data: outputArray, filename: "consolidatediRounded_\(timestamp).txt")
    } catch {
        print("Broke")
    }

    var time_in_repi = [Float](repeating: 0.0, count: consolidatediRounded.count)
    var end_repsi: [Int] = []

    if consolidatediRounded.reduce(0, +) > 0 {
        let diffArray = diff(consolidatediRounded)
        do {
            let outputArray = try MLMultiArray(diffArray)
            predictionExportManager.exportFloats(data: outputArray, filename: "diffArray_\(timestamp).txt")
        } catch {
            print("Broke")
        }
        var starts = findIndicesStart(in: diffArray)
        var ends = findIndicesEnd(in: diffArray)
        
        do {
            try print("Test", MLMultiArray(starts))
            try print("Test", MLMultiArray(starts).shape)
        } catch {
            print("Nice")
        }
        
        do {
            let outputArray = try MLMultiArray(starts)
            predictionExportManager.exportInts(data: outputArray, filename: "startsInit_\(timestamp).txt")
        } catch {
            print("Broke")
        }
        do {
            let outputArray = try MLMultiArray(ends)
            predictionExportManager.exportInts(data: outputArray, filename: "endsInit_\(timestamp).txt")
        } catch {
            print("Broke")
        }

        if starts.isEmpty || ends.isEmpty {
            if starts.isEmpty && !ends.isEmpty {
                starts = [0]
            } else if !starts.isEmpty && ends.isEmpty {
                ends = [consolidatediRounded.count]
            }
        } else {
            if starts[0] > ends[0] {
                starts.insert(0, at: 0)
            }
            if ends.last! < starts.last! {
                ends.append(consolidatediRounded.count)
            }
        }
        
        for i in 0..<min(starts.count, ends.count) {
            let start = starts[i]
            let end = ends[i]
            end_repsi.append((start + end) / 2)
        }
        
        do {
            let outputArray = try MLMultiArray(starts)
            predictionExportManager.exportInts(data: outputArray, filename: "starts_\(timestamp).txt")
        } catch {
            print("Broke")
        }
        do {
            let outputArray = try MLMultiArray(ends)
            predictionExportManager.exportInts(data: outputArray, filename: "ends_\(timestamp).txt")
        } catch {
            print("Broke")
        }
        
        // Use `stride` and `winsize` parameters directly
        let full_winsize_val = Float(stride * (current_T_dim - 1) + winsize)

        for i in 1..<end_repsi.count {
            let startIndex = end_repsi[i-1]
            let endIndex = end_repsi[i]
            let value = Float(end_repsi[i] - end_repsi[i-1]) / full_winsize_val
            for j in startIndex..<endIndex {
                if j < time_in_repi.count {
                    time_in_repi[j] = value
                }
            }
        }
    }
    
    do {
        let outputArray = try MLMultiArray(time_in_repi)
        predictionExportManager.exportFloats(data: outputArray, filename: "time_in_repi_\(timestamp).txt")
    } catch {
        print("Broke")
    }
    
    // Rewindow and stack
    var finalTimeInRep: [Float] = []
    
    // Unfold: iterate over T windows (since N=1)
    
    for t_window_idx in 0..<current_T_dim {
        let start_idx = t_window_idx * stride
        // Ensure not to go out of bounds for the last window
        //let _ = min(start_idx + winsize, time_in_repi.count)
        let window = Array(time_in_repi[start_idx..<(start_idx + winsize)])
        finalTimeInRep.append(contentsOf: window)
    }
    
    print(finalTimeInRep.count)
    
    // The final output shape for time_in_rep is (N=1, T, winsize)
    let outputShape: [NSNumber] = [
        NSNumber(value: FIXED_N_BATCH_SIZE), // N (fixed at 1, from SharedSessionManager)
        NSNumber(value: current_T_dim),      // T
        NSNumber(value: winsize)             // winsize (from parameter, which is WINDOW_LENGTH)
    ]
    
    return arrayToMLMultiArray(finalTimeInRep, shape: outputShape)
}
