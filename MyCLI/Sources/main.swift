import Accelerate

var consolidatedi: [Float] = [Float](repeating: 0.0, count: 2240)

consolidatedi.replaceSubrange(470..<530, with: repeatElement(1, count: 60))
consolidatedi.replaceSubrange(1470..<1530, with: repeatElement(1, count: 60))

let KERNEL_SIZE: Int = 100

// let kernel: [Float] = [Float](repeating: 1.0 / Float(KERNEL_SIZE), count: KERNEL_SIZE) // Use KERNEL_SIZE constant
let kernel: [Float] = [Float](repeating: 1.0 / Float(KERNEL_SIZE), count: 100) // Use KERNEL_SIZE constant
var consolidatediRounded: [Float] = [Float](repeating: 0.0, count: consolidatedi.count)

// let padSize: Int = KERNEL_SIZE / 2
// var paddedConsolidatedi: [Float] = [Float](repeating: 0.0, count: consolidatedi.count + 2 * padSize)
// for i: Int in 0..<consolidatedi.count {
//     paddedConsolidatedi[i + padSize] = consolidatedi[i]
// }

// vDSP_conv(paddedConsolidatedi, 1, kernel.reversed(), 1, &convolvedData, 1, vDSP_Length(consolidatedi.count), vDSP_Length(kernel.count))

if #available(macOS 10.15, *) {
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
}

consolidatediRounded = consolidatediRounded.map { $0.rounded() }


var outFilename: NSString = "test_file.txt"

// Begin file manager segment
// Check for file presence and create it if it does not exist
let filemgr: FileManager = FileManager.default
let path: URL? = filemgr.urls(for: FileManager.SearchPathDirectory.documentDirectory, in:     FileManager.SearchPathDomainMask.userDomainMask).last?.appendingPathComponent(outFilename as String)
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

var consolidatediBefore: [Float] = [Float](repeating: 0.0, count: 2240)

consolidatediBefore.replaceSubrange(470..<530, with: repeatElement(1, count: 60))
consolidatediBefore.replaceSubrange(1470..<1530, with: repeatElement(1, count: 60))

for i: Int in 0..<consolidatediRounded.count
{
    var s0: String = String(consolidatediRounded[i])
    if i != consolidatediRounded.count - 1 {
        s0.append(" ")
    }
    fileHandle!.write(s0.data(using: .utf8)!)
}