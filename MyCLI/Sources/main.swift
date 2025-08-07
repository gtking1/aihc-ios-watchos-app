import Accelerate
import CoreML

let convNetInputShape: [NSNumber] = [
    NSNumber(value: 32),
    NSNumber(value: 256),
    NSNumber(value: 6)
]

if #available(macOS 15, *) {
    let convNetModelInputX: MLMultiArray = try MLMultiArray([Float](repeating: 1.0, count: 49152))

    var consolidatedi: [Float] = [Float](repeating: 0.0, count: 2240)
    var X_smooth: [Float] = [Float](repeating: 0.0, count: 2240)

    var shapedConvNetModelInputX: MLShapedArray<Float> = MLShapedArray<Float>(convNetModelInputX)

    shapedConvNetModelInputX = shapedConvNetModelInputX.reshaped(to: [32, 256, 6]).transposed(permutation: [0, 2, 1])

    // print(shapedConvNetModelInputX)

    let finalConvNetModelInputX: MLMultiArray = MLMultiArray(shapedConvNetModelInputX)

    let startingZeros2: [Float] = [Float](repeating: 0.0, count: 7)
    let endingZeros2: [Float] = [Float](repeating: 0.0, count: 8)
    let kernel2: [Float] = [Float](repeating: 1.0 / Float(15), count: 15)
    
    for i: Int in 0..<32 {
        for j: Int in 0..<6 {
            var X_smooth: [Float] = [Float](repeating: 0.0, count: 256)
            var X_smoothRounded: [Float] = [Float](repeating: 0.0, count: 256)
            for k: Int in 0..<256 {
                let accessShape: [NSNumber] = [
                    NSNumber(value: i),
                    NSNumber(value: j),
                    NSNumber(value: k)
                ]
                X_smooth[k] = finalConvNetModelInputX[accessShape].floatValue
                // print(finalConvNetModelInputX[accessShape].floatValue)
            }
            X_smooth = startingZeros2 + X_smooth + endingZeros2
            X_smoothRounded = vDSP.convolve(X_smooth, withKernel: kernel2)
            for k: Int in 0..<256 {
                let accessShape: [NSNumber] = [
                    NSNumber(value: i),
                    NSNumber(value: j),
                    NSNumber(value: k)
                ]
                finalConvNetModelInputX[accessShape] = NSNumber(value: X_smoothRounded[k])
                print(finalConvNetModelInputX[accessShape].floatValue)
            }
        }
    }


    consolidatedi.replaceSubrange(470..<530, with: repeatElement(1, count: 60))
    consolidatedi.replaceSubrange(1470..<1530, with: repeatElement(1, count: 60))
    X_smooth.replaceSubrange(470..<530, with: repeatElement(1, count: 60))
    X_smooth.replaceSubrange(1470..<1530, with: repeatElement(1, count: 60))

    let KERNEL_SIZE: Int = 100

    // let kernel: [Float] = [Float](repeating: 1.0 / Float(KERNEL_SIZE), count: KERNEL_SIZE) // Use KERNEL_SIZE constant
    let kernel: [Float] = [Float](repeating: 1.0 / Float(KERNEL_SIZE), count: 100) // Use KERNEL_SIZE constant
    //let kernel2: [Float] = [Float](repeating: 1.0 / Float(15), count: 15)
    var consolidatediRounded: [Float] = [Float](repeating: 0.0, count: consolidatedi.count)
    var X_smoothRounded: [Float] = [Float](repeating: 0.0, count: consolidatedi.count)

    // let padSize: Int = KERNEL_SIZE / 2
    // var paddedConsolidatedi: [Float] = [Float](repeating: 0.0, count: consolidatedi.count + 2 * padSize)
    // for i: Int in 0..<consolidatedi.count {
    //     paddedConsolidatedi[i + padSize] = consolidatedi[i]
    // }

    // vDSP_conv(paddedConsolidatedi, 1, kernel.reversed(), 1, &convolvedData, 1, vDSP_Length(consolidatedi.count), vDSP_Length(kernel.count))

    let startingZeros: [Float] = [Float](repeating: 0.0, count: 50)
    consolidatedi = startingZeros + consolidatedi + startingZeros
    consolidatediRounded = vDSP.convolve(consolidatedi, withKernel: kernel)
    //let startingZeros2: [Float] = [Float](repeating: 0.0, count: 7)
    //let endingZeros2: [Float] = [Float](repeating: 0.0, count: 8)
    X_smooth = startingZeros2 + X_smooth + endingZeros2
    X_smoothRounded = vDSP.convolve(X_smooth, withKernel: kernel2)
    // for i: Int in 0..<consolidatediRounded.count {
    //     if consolidatediRounded[i] == 0.5 {
    //         print("Found equal value")
    //         consolidatediRounded[i] = 0
    //     }
    // }
    // for i: Int in 0..<consolidatediRounded.count {
    //     if X_smooth[i] == 0.5 {
    //         print("Found equal value")
    //         X_smooth[i] = 0
    //     }
    // }

    // consolidatediRounded = consolidatediRounded.map { $0.rounded() }


    let outFilename: NSString = "test_file.txt"

    // Begin file manager segment
    // Check for file presence and create it if it does not exist
    let filemgr: FileManager = FileManager.default
    let path: URL? = filemgr.urls(for: FileManager.SearchPathDirectory.documentDirectory, in: FileManager.SearchPathDomainMask.userDomainMask).last?.appendingPathComponent(outFilename as String)
    if !filemgr.fileExists(atPath: (path?.absoluteString)!) {
    filemgr.createFile(atPath: String(outFilename),  contents:Data(" ".utf8), attributes: nil)
    }
    // End file manager Segment

    // Open outFilename for writing – this does not create a file
    let fileHandle: FileHandle? = FileHandle(forWritingAtPath: outFilename as String)

    if(fileHandle == nil)
    {
    print("Open of outFilename forWritingAtPath: failed.  \nCheck whether the file already exists.  \nIt should already exist.\n");
    exit(0)
    }

    // var consolidatediBefore: [Float] = [Float](repeating: 0.0, count: 2240)

    // consolidatediBefore.replaceSubrange(470..<530, with: repeatElement(1, count: 60))
    // consolidatediBefore.replaceSubrange(1470..<1530, with: repeatElement(1, count: 60))

    for i: Int in 0..<49152
    {
        var s0: String = String(finalConvNetModelInputX[i].floatValue)
        print(finalConvNetModelInputX.count)
        if i != (finalConvNetModelInputX.count - 1) {
            s0.append(" ")
        }
        fileHandle!.write(s0.data(using: .utf8)!)
    }
}