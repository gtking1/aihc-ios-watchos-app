import Foundation
import WatchConnectivity
import CoreML
import Accelerate

// MARK: - Global Constants for Model Dimensions
let FIXED_N_BATCH_SIZE = 1 // N (fixed for Core ML input)
let SEQUENCE_LENGTH = 32   // T (number of windows your LSTM expects)
let CHANNELS = 6           // C (e.g., 3 accel + 3 gyro)
let WINDOW_LENGTH = 256    // L (length of each window/frame)

// Constants for get_time_in_reps_swift based on your previous info:
let LSTM_STRIDE = 64
let KERNEL_SIZE = 100 // This constant seems unused in the provided code, but kept for context.
let WINS_SIZE = WINDOW_LENGTH

// NEW CONSTANT: Inference trigger window count
let INFERENCE_TRIGGER_WINDOWS = 8


// MARK: - PredictionExportManager (New Class for Data Export)
class PredictionExportManager {
    private let fileManager = FileManager.default
    private let documentsDirectory: URL

    init() {
        // Get the URL for the app's Documents directory
        documentsDirectory = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first!
        print("PredictionExportManager initialized. Documents directory: \(documentsDirectory.path)")
    }

    /// Converts an MLMultiArray to a flat Swift Array of Floats.
    private func mlMultiArrayToArray(multiArray: MLMultiArray) -> [Float] {
        let count = multiArray.count
        let ptr = multiArray.dataPointer.assumingMemoryBound(to: Float.self)
        var array = [Float](repeating: 0.0, count: count)
        for i in 0..<count {
            array[i] = ptr[i]
        }
        return array
    }
    
    private func mlMultiArrayToArrayInts(multiArray: MLMultiArray) -> [Int32] {
        let count = multiArray.count
        print("Count", count)
        let ptr = multiArray.dataPointer.assumingMemoryBound(to: Int32.self)
        print(ptr)
        var array = [Int32](repeating: 0, count: count)
        print(array)
        for i in 0..<count {
            array[i] = ptr[i]
            print(array[i])
            print(ptr[i])
        }
        print(array)
        return array
    }

    /// Saves a flat array of floats to a text file in the Documents directory.
    func exportFloats(data: MLMultiArray, filename: String) {
        let flatArray = mlMultiArrayToArray(multiArray: data)
        let dataString = flatArray.map { String($0) }.joined(separator: " ")
        let fileURL = documentsDirectory.appendingPathComponent(filename)

        do {
            try dataString.write(to: fileURL, atomically: true, encoding: .utf8)
            print("Successfully exported \(filename) to \(fileURL.lastPathComponent)")
        } catch {
            print("Error exporting \(filename): \(error.localizedDescription)")
        }
    }
    
    func exportInts(data: MLMultiArray, filename: String) {
        let flatArray = mlMultiArrayToArrayInts(multiArray: data)
        let dataString = flatArray.map { String($0) }.joined(separator: " ")
        let fileURL = documentsDirectory.appendingPathComponent(filename)

        do {
            try dataString.write(to: fileURL, atomically: true, encoding: .utf8)
            print("Successfully exported \(filename) to \(fileURL.lastPathComponent)")
        } catch {
            print("Error exporting \(filename): \(error.localizedDescription)")
        }
    }

    /// Clears all exported files from the documents directory.
    func clearExportedFiles() {
        do {
            let fileURLs = try fileManager.contentsOfDirectory(at: documentsDirectory, includingPropertiesForKeys: nil)
            for fileURL in fileURLs {
                if fileURL.pathExtension == "txt" { // Only remove our exported text files
                    try fileManager.removeItem(at: fileURL)
                }
            }
            print("Cleared all exported .txt files from documents directory.")
        } catch {
            print("Error clearing exported files: \(error.localizedDescription)")
        }
    }
}

// MARK: - SharedSessionManager Class

class SharedSessionManager: NSObject, ObservableObject {
    static let shared = SharedSessionManager() // Singleton instance

    // MARK: - Existing Published Properties
    @Published var lastPrediction: Double = 0.0
    @Published var hasReceivedPrediction: Bool = false
    @Published var lastReceivedMessage: String = "No message" // Keep for general message tracking
    @Published var connectionStatus: String = "Not Activated"
    @Published var lastPredictionTimestamp: Date? = nil

    // Separate isSessionActive for each platform's internal state
    #if os(watchOS)
    @Published var isSessionActive: Bool = false // Watch's internal session active state
    #elseif os(iOS)
    @Published var isSessionActive: Bool = false // iPhone's internal session active state
    #endif

    // MARK: - NEW Published Properties for Watch Status (iPhone side will update these)
    @Published var watchCollectionStatus: String = "Watch Idle"
    @Published var watchRawBufferSampleCount: Int = 0
    @Published var watchReadyForWindow: Bool = false
    @Published var watchWindowsInBatchCount: Int = 0

    // MARK: - Core ML Related Properties (iPhone Only)
    #if os(iOS)
    private var convNetModel: convnet_model? // Your ConvNet model class
    private var lstmCoreModel: lstm_core_model? // Your LSTM model class
    internal var sensorDataManager: SensorDataManager!
    private var predictionExportManager: PredictionExportManager? // NEW: Instance for exporting data

    // iPhone's own buffer count
    @Published var currentBufferCount: Int = 0
    #endif

    // MARK: - Initialization
    private override init() {
        super.init()
        if WCSession.isSupported() {
            WCSession.default.delegate = self
            WCSession.default.activate()
            print("WCSession activated.")
        } else {
            print("WCSession is not supported on this device.")
        }

        #if os(iOS)
        // Initialize PredictionExportManager first
        self.predictionExportManager = PredictionExportManager()

        // Load Core ML models and initialize SensorDataManager on iPhone
        do {
            self.convNetModel = try convnet_model(configuration: MLModelConfiguration())
            self.lstmCoreModel = try lstm_core_model(configuration: MLModelConfiguration())

            // Initialize SensorDataManager and pass callbacks
            self.sensorDataManager = SensorDataManager(
                channels: CHANNELS,
                windowLength: WINDOW_LENGTH,
                sequenceLength: SEQUENCE_LENGTH,
                inferenceTriggerWindowCount: INFERENCE_TRIGGER_WINDOWS,
                onBufferCountUpdate: { [weak self] count in
                    DispatchQueue.main.async {
                        self?.currentBufferCount = count
                    }
                }
            )
            print("Core ML models (ConvNet, LSTM) loaded successfully on iPhone.")
            
            print("Array test")
            
            //var shapeArray : [NSNumber] = [41952]
            
            //let floats : [Float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
            
            //let myArray = try MLMultiArray(floats)
//            let myArray = try MLMultiArray(shape: [49152], dataType: .float32)
//            
//            var shapedArray = MLShapedArray<Float>(myArray)
//            
//            print(shapedArray.shape)
//            
//            shapedArray = shapedArray.reshaped(to: [32, 256, 6]).transposed(permutation: [0, 2, 1])
//            
//            print(shapedArray.shape)
//            
//            let finalArray = MLMultiArray(shapedArray)
            
            
        } catch {
            print("Error loading Core ML models on iPhone: \(error)")
        }
        #endif

        // Ensure connection status is updated immediately after init
        DispatchQueue.main.async {
            self.updateConnectionStatus()
        }
    }

    // MARK: - Public Methods
    func updateConnectionStatus() {
        if WCSession.default.isReachable {
            connectionStatus = "Reachable"
        } else if WCSession.default.activationState == .activated {
            connectionStatus = "Activated"
        } else {
            connectionStatus = "Not Activated"
        }
    }

    func sendToCompanion(message: [String: Any], replyHandler: (([String: Any]) -> Void)? = nil, errorHandler: ((Error) -> Void)? = nil) {
        guard WCSession.default.isReachable else {
            DispatchQueue.main.async { // Ensure UI update on main thread
                self.connectionStatus = "Not Reachable"
                print("WCSession not reachable. Message not sent: \(message.keys.first ?? "Unknown")")
                errorHandler?(NSError(domain: "WatchConnectivityError", code: 0, userInfo: [NSLocalizedDescriptionKey: "Companion not reachable"]))
            }
            return
        }

        WCSession.default.sendMessage(message, replyHandler: { reply in
            DispatchQueue.main.async { // Ensure reply handler execution on main thread
                print("Received reply from companion: \(reply)")
                replyHandler?(reply)
            }
        }) { error in
            DispatchQueue.main.async { // Ensure error handler execution on main thread
                print("Error sending message: \(error.localizedDescription)")
                self.connectionStatus = "Error: \(error.localizedDescription)" // Update connection status on error
                errorHandler?(error)
            }
        }
    }

    #if os(iOS)
    // MARK: - iPhone Session State Management
    /// Resets all iPhone-side session state properties and signals SensorDataManager to stop.
    private func resetiPhoneSessionState() {
        DispatchQueue.main.async {
            self.isSessionActive = false // iPhone's internal session status
            self.lastPrediction = 0.0
            self.hasReceivedPrediction = false
            self.lastPredictionTimestamp = nil
            self.currentBufferCount = 0 // Reset iPhone's buffer count display
            
            // Reset Watch's reported status
            self.watchCollectionStatus = "Watch Idle"
            self.watchRawBufferSampleCount = 0
            self.watchWindowsInBatchCount = 0
            self.watchReadyForWindow = false
            self.lastReceivedMessage = "iPhone Session Reset" // Indicate internal reset
        }
        // Signal the SensorDataManager to stop and clear its buffer
        self.sensorDataManager.stopProcessing()
        // Optionally clear exported files upon session reset
        //self.predictionExportManager?.clearExportedFiles()
        print("iPhone: All iPhone session states reset. SensorDataManager stopped. Exported files cleared.")
    }

    /// Activates iPhone-side session state and signals SensorDataManager to start.
    private func activateiPhoneSessionState() {
        DispatchQueue.main.async {
            self.isSessionActive = true // iPhone's internal session status
            // Reset prediction display to indicate a new session starting
            self.lastPrediction = 0.0
            self.hasReceivedPrediction = false
            self.lastPredictionTimestamp = nil
        }
        // Signal the SensorDataManager to start processing
        self.sensorDataManager.startProcessing()
        print("iPhone: iPhone session states activated. SensorDataManager started.")
    }
    #endif // #if os(iOS)

    // MARK: - Core ML Inference (iPhone Only)
    #if os(iOS)
    private func performInference(with convNetBatchInput: MLMultiArray) -> Double {
        guard let convNetModel = convNetModel,
              let lstmCoreModel = lstmCoreModel,
              let predictionExportManager = predictionExportManager else {
            print("Core ML models (ConvNet or LSTM) not loaded for inference.")
            return 0.0
        }

        do {
            // Generate a unique timestamp for this inference run to name files
            let timestamp = Int(Date().timeIntervalSince1970)
            
            // 1. Prepare ConvNet Input
            guard convNetBatchInput.shape.count == 4,
                  convNetBatchInput.shape[0].intValue == FIXED_N_BATCH_SIZE,
                  convNetBatchInput.shape[1].intValue == SEQUENCE_LENGTH, // T
                  convNetBatchInput.shape[2].intValue == CHANNELS,
                  convNetBatchInput.shape[3].intValue == WINDOW_LENGTH else {
                print("Error: convNetBatchInput from SensorDataManager has unexpected shape for ConvNet: \(convNetBatchInput.shape)")
                return 0.0
            }

            // Create a new MLMultiArray with shape [T, C, L] for ConvNet's 'x' input
            let convNetInputShape: [NSNumber] = [
                NSNumber(value: SEQUENCE_LENGTH),
                NSNumber(value: WINDOW_LENGTH),
                NSNumber(value: CHANNELS)
            ]
            
            let lstmtInputShape: [NSNumber] = [
                NSNumber(value: SEQUENCE_LENGTH),
                NSNumber(value: WINDOW_LENGTH),
                NSNumber(value: CHANNELS)
            ]
            
            let convNetModelInputX = try MLMultiArray(shape: convNetInputShape, dataType: .float32)

            // Copy data from convNetBatchInput (1xTxCxL) to convNetModelInputX (TxCxL)
            let convNetBatchInputPtr = convNetBatchInput.dataPointer.assumingMemoryBound(to: Float.self)
            let convNetModelInputXPtr = convNetModelInputX.dataPointer.assumingMemoryBound(to: Float.self)
            
            let elementsPerSequence = SEQUENCE_LENGTH * CHANNELS * WINDOW_LENGTH
            memcpy(convNetModelInputXPtr, convNetBatchInputPtr, elementsPerSequence * MemoryLayout<Float>.size)
            
            //            var shapedArray = MLShapedArray<Float>(myArray)
            //
            //            print(shapedArray.shape)
            //
            //            shapedArray = shapedArray.reshaped(to: [32, 256, 6]).transposed(permutation: [0, 2, 1])
            //
            //            print(shapedArray.shape)
            //
            //            let finalArray = MLMultiArray(shapedArray)
            
            var shapedConvNetModelInputX = MLShapedArray<Float>(convNetModelInputX)
            
            shapedConvNetModelInputX = shapedConvNetModelInputX.transposed(permutation: [0, 2, 1])
            
            let finalConvNetModelInputX = MLMultiArray(shapedConvNetModelInputX)
            
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
            
            var lstmUnsqueeze = MLShapedArray<Float>(finalConvNetModelInputX).reshaped(to: [1, 32, 6, 256])
            let finalLstmModelInputX = MLMultiArray(lstmUnsqueeze)

            // EXPORT: Input to the Encoder (ConvNet)
            predictionExportManager.exportFloats(data: finalConvNetModelInputX, filename: "encoder_input_\(timestamp).txt")
            predictionExportManager.exportFloats(data: finalLstmModelInputX, filename: "encoder_input_\(timestamp).txt")


            // 2. ConvNet Inference
            var tic = CFAbsoluteTimeGetCurrent()
            let convNetPredictionOutput = try convNetModel.prediction(x: finalConvNetModelInputX)
            var toc = CFAbsoluteTimeGetCurrent()
            let duration1 = Int32(1000 * (toc - tic))
            let round1Output = convNetPredictionOutput.round_1

            print("ConvNet output (round_1) shape: \(round1Output.shape)")

            // EXPORT: Output of the Encoder (ConvNet)
            predictionExportManager.exportFloats(data: round1Output, filename: "encoder_output_\(timestamp).txt")

            // EXPORT: Input to get_time_in_reps_swift (same as encoder output)
            predictionExportManager.exportFloats(data: round1Output, filename: "get_time_in_reps_input_\(timestamp).txt")


            // 3. Process with get_time_in_reps_swift
            tic = CFAbsoluteTimeGetCurrent()
            guard let timeInRepMultiArray = get_time_in_reps_swift(predsMultiArray: round1Output, stride: LSTM_STRIDE, winsize: WINS_SIZE) else {
                print("get_time_in_reps_swift returned nil.")
                return 0.0
            }
            toc = CFAbsoluteTimeGetCurrent()
            let duration2 = Int32(1000 * (toc - tic))

            print("TimeInRep output shape: \(timeInRepMultiArray.shape)")

            // EXPORT: Output of get_time_in_reps_swift
            predictionExportManager.exportFloats(data: timeInRepMultiArray, filename: "get_time_in_reps_output_\(timestamp).txt")


            // 4. LSTMNet Inference
            let lstmModelInput = lstm_core_modelInput(
                x: finalLstmModelInputX,
                time_in_rep: timeInRepMultiArray
            )
            
            // EXPORT: Input 'x' to the LSTM
            predictionExportManager.exportFloats(data: finalLstmModelInputX, filename: "lstm_input_x_\(timestamp).txt")
            // EXPORT: Input 'time_in_rep' to the LSTM
            predictionExportManager.exportFloats(data: timeInRepMultiArray, filename: "lstm_input_time_in_rep_\(timestamp).txt")


            tic = CFAbsoluteTimeGetCurrent()
            let lstmPredictionOutput = try lstmCoreModel.prediction(input: lstmModelInput)
            toc = CFAbsoluteTimeGetCurrent()
            let duration3 = Int32(1000 * (toc - tic))
            
            print("duration1: \(duration1) duration2: \(duration2) duration3: \(duration3)")
            
            let finalPredictionSeries = lstmPredictionOutput.var_357

            print("LSTM output (var_357) shape: \(finalPredictionSeries.shape)")

            // EXPORT: Output of the LSTM
            predictionExportManager.exportFloats(data: finalPredictionSeries, filename: "lstm_output_\(timestamp).txt")


            // Extract the final prediction (last value of the sequence).
            guard finalPredictionSeries.shape.count == 3,
                  finalPredictionSeries.shape[0].intValue == FIXED_N_BATCH_SIZE,
                  finalPredictionSeries.shape[1].intValue == SEQUENCE_LENGTH,
                  finalPredictionSeries.shape[2].intValue == 1 else {
                print("Error: LSTM output 'var_357' has unexpected shape: \(finalPredictionSeries.shape)")
                return 0.0
            }

            let lastPredictionIndex = [
                0, // N (batch size)
                SEQUENCE_LENGTH - 1, // T (last time step)
                0  // Value (single output)
            ] as [NSNumber]

            let rawPrediction = finalPredictionSeries[lastPredictionIndex].floatValue

            // --- Apply Sigmoid and Rounding ---
            let sigmoidValue = 1.0 / (1.0 + exp(-Double(rawPrediction)))
            let finalBinaryPrediction = round(sigmoidValue)

            print("Raw prediction from LSTM: \(rawPrediction), Sigmoid value: \(sigmoidValue), Final binary prediction: \(finalBinaryPrediction)")
            
            DispatchQueue.main.async { // Ensure UI updates on main thread
                self.lastPredictionTimestamp = Date() // Get current date/time
            }

            return finalBinaryPrediction
        } catch {
            print("Error during two-stage inference (ConvNet -> get_time_in_reps -> LSTM): \(error)")
            return 0.0
        }
    }
    #endif
}

// MARK: - WCSessionDelegate Extension

extension SharedSessionManager: WCSessionDelegate {
    func session(_ session: WCSession, activationDidCompleteWith activationState: WCSessionActivationState, error: Error?) {
        DispatchQueue.main.async {
            self.updateConnectionStatus()
            if let error = error {
                print("WCSessionDelegate: Session activation failed. State: \(activationState.rawValue), Error: \(error.localizedDescription)")
            } else {
                print("WCSessionDelegate: Session activation successful. State: \(activationState.rawValue)")
            }
            // Update the isSessionActive @Published property based on the actual activation state
            #if os(watchOS)
            self.isSessionActive = (activationState == .activated) // For Watch, it's just about WCSession
            #elseif os(iOS)
            // For iPhone, isSessionActive is managed by messages from Watch, not just WCSession activation
            #endif
        }
    }

    #if os(iOS)
    func sessionDidBecomeInactive(_ session: WCSession) {
        DispatchQueue.main.async {
            self.updateConnectionStatus()
            print("iPhone: WCSession became inactive. Setting internal session to inactive and attempting reactivate.")
            self.resetiPhoneSessionState() // Crucial: Reset state if WCSession goes inactive
            WCSession.default.activate() // Attempt to reactivate to maintain connection
        }
    }

    func sessionDidDeactivate(_ session: WCSession) {
        DispatchQueue.main.async {
            self.updateConnectionStatus()
            print("iPhone: WCSession deactivated. Setting internal session to inactive and attempting reactivate.")
            self.resetiPhoneSessionState() // Crucial: Reset state if WCSession deactivates
            WCSession.default.activate() // Attempt to reactivate to maintain connection
        }
    }

    func sessionReachabilityDidChange(_ session: WCSession) {
        DispatchQueue.main.async {
            self.updateConnectionStatus()
            print("iPhone: Session reachability changed: \(session.isReachable ? "Reachable" : "Not Reachable")")
            // If the companion becomes unreachable while the iPhone session is active, stop processing
            if !session.isReachable && self.isSessionActive {
                print("iPhone: Companion became unreachable while session active. Stopping iPhone session.")
                self.resetiPhoneSessionState()
            }
        }
    }
    #endif // #if os(iOS)

    func session(_ session: WCSession, didReceiveMessage message: [String : Any], replyHandler: @escaping ([String : Any]) -> Void) {
        DispatchQueue.main.async {
            // Update lastReceivedMessage for any incoming message (general display)
            self.lastReceivedMessage = "Received: \(message.keys.joined(separator: ", "))"
            //print("Received message on device: \(message)")

            #if os(iOS)
            // --- NEW: Handle Watch Status Updates ---
            if let watchStatusUpdate = message["watchStatusUpdate"] as? Bool, watchStatusUpdate == true {
                if let status = message["collectionStatus"] as? String {
                    self.watchCollectionStatus = status
                }
                if let rawCount = message["rawBufferCount"] as? Int {
                    self.watchRawBufferSampleCount = rawCount
                }
                if let ready = message["readyForWindow"] as? Bool {
                    self.watchReadyForWindow = ready
                }
                if let batchCount = message["windowsInBatch"] as? Int {
                    self.watchWindowsInBatchCount = batchCount
                }
                replyHandler(["status_ack": true]) // Acknowledge receipt of status update
                return // IMPORTANT: Return here to prevent further processing if it's just a status update
            }
            // --- END Watch Status Updates ---

            // Handle session reset message from Watch
            if let reset = message["resetSession"] as? Bool, reset == true {
                print("iPhone: Received resetSession message from Watch. Performing full session reset.")
                self.resetiPhoneSessionState() // First, clear everything
                self.activateiPhoneSessionState() // Then, activate for a new session
                replyHandler(["message": "iPhone session reset and ready."])
                return
            }

            // --- CRUCIAL FIX: Handle session stop message from Watch ---
            if let stop = message["stopSession"] as? Bool, stop == true {
                print("iPhone: Received stopSession message from Watch. Stopping iPhone session.")
                self.resetiPhoneSessionState() // Reset all state, including setting isSessionActive to false
                replyHandler(["message": "iPhone session stopped."])
                return
            }

            // Only process sensor data if the iPhone session is marked as active
            if self.isSessionActive {
                if let incomingWindows = message["sensorData"] as? [[Float]] {
                    print("iPhone: Received \(incomingWindows.count) sensor data windows.")
                    self.sensorDataManager.addWindowsForInferenceTrigger(incomingWindows: incomingWindows) { [weak self] fullSequenceReadyForInference in
                        guard let self = self else { return }

                        if let convNetInput = fullSequenceReadyForInference {
                            print("iPhone: Full sequence (T=\(SEQUENCE_LENGTH)) ready and inference trigger met (\(self.sensorDataManager.accumulatedNewWindowsCount) new windows). Processing sensor data via pipeline...")
                            let prediction = self.performInference(with: convNetInput)

                            print("iPhone: Sending prediction \(String(format: "%.0f", prediction)) back to Watch.")
                            // Send reply back to Watch with prediction
                            replyHandler(["prediction": prediction, "message": "Prediction processed on iPhone"])

                            // Update iPhone's UI state with the prediction
                            self.lastPrediction = prediction
                            self.hasReceivedPrediction = true
                            self.lastPredictionTimestamp = Date() // Update timestamp here
                        } else {
                            print("iPhone: Received sensor data, accumulating for inference. Buffer count: \(self.sensorDataManager.buffer.count), New windows: \(self.sensorDataManager.accumulatedNewWindowsCount)/\(INFERENCE_TRIGGER_WINDOWS).")
                            replyHandler(["message": "Data received, accumulating for inference."])
                        }
                    }
                } else {
                    print("iPhone: Received unknown message format from Watch (expected sensorData).")
                    replyHandler(["error": "Unknown message format"])
                }
            } else {
                print("iPhone: Received data but session is inactive. Discarding and replying.")
                replyHandler(["message": "Session inactive, data discarded."])
            }
            #elseif os(watchOS)
                // For Watch, any unhandled message from iPhone (besides prediction, which is handled in SensorDataCollector)
                // could be logged or ignored.
                print("Watch: Received an unhandled direct message from iPhone: \(message)")
                // For a general unhandled message, just reply to close the communication channel
                replyHandler(["message": "Watch received unhandled message."])
            #endif
        }
    }
}


// MARK: - SensorDataManager (iPhone Only)
#if os(iOS)
class SensorDataManager {
    internal var buffer: [[Float]] // Stores incoming windows
    private let channels: Int
    private let windowLength: Int
    private let sequenceLength: Int
    private let inferenceTriggerWindowCount: Int // How many new windows before triggering inference

    private var onBufferCountUpdate: ((Int) -> Void)?
    
    // Tracks if the SensorDataManager is actively processing/buffering data
    private var isActive: Bool = false

    // Counter for newly added windows since last inference trigger
    internal private(set) var accumulatedNewWindowsCount: Int = 0

    init(channels: Int, windowLength: Int, sequenceLength: Int, inferenceTriggerWindowCount: Int, onBufferCountUpdate: ((Int) -> Void)? = nil) {
        self.channels = channels
        self.windowLength = windowLength
        self.sequenceLength = sequenceLength
        self.inferenceTriggerWindowCount = inferenceTriggerWindowCount
        self.buffer = []
        self.onBufferCountUpdate = onBufferCountUpdate
        print("SensorDataManager initialized.")
    }
    
    // MARK: Lifecycle Methods
    func startProcessing() {
        if !isActive {
            print("SensorDataManager: Starting processing.")
            isActive = true
            clearBuffer() // Clear buffer upon starting a new session
        }
    }
    
    func stopProcessing() {
        if isActive {
            print("SensorDataManager: Stopping processing.")
            isActive = false
            clearBuffer() // Clear buffer upon stopping the session
        }
    }

    // This method is now internal and used by addWindowsForInferenceTrigger
    private func addWindowsToBuffer(_ newWindows: [[Float]]) {
        // Only add if active
        guard isActive else {
            print("SensorDataManager: Not active, discarding incoming windows.")
            return
        }
        
        buffer.append(contentsOf: newWindows)
        
        // Ensure buffer doesn't exceed the required sequenceLength
        // We always keep the *latest* `sequenceLength` windows
        if buffer.count > sequenceLength {
            buffer.removeFirst(buffer.count - sequenceLength)
        }
        print("SensorDataManager: Buffer size: \(buffer.count) windows (max \(sequenceLength)).")
        onBufferCountUpdate?(buffer.count)
    }

    // Public method to add windows and conditionally trigger inference
    func addWindowsForInferenceTrigger(incomingWindows: [[Float]], completion: (MLMultiArray?) -> Void) {
        addWindowsToBuffer(incomingWindows) // Add to the main buffer (handles isActive check internally)

        // Increment the count of new windows since the last trigger
        // Only increment if we actually added them (i.e., if isActive was true)
        if isActive {
            accumulatedNewWindowsCount += incomingWindows.count
        }

        // Trigger inference ONLY if:
        // 1. We have accumulated enough *new* windows to meet the trigger count, AND
        // 2. The total buffer now contains the full SEQUENCE_LENGTH (T) windows.
        if isActive && accumulatedNewWindowsCount >= inferenceTriggerWindowCount && buffer.count == sequenceLength {
            print("SensorDataManager: Triggering inference. Full sequence of \(sequenceLength) windows available, and \(accumulatedNewWindowsCount) new windows accumulated (>= \(inferenceTriggerWindowCount)).")
            let multiArray = getCurrentSequenceForInference()
            accumulatedNewWindowsCount = 0 // Reset counter after triggering inference
            completion(multiArray)
        } else {
            completion(nil) // Not enough new windows or buffer not full yet, return nil
        }
    }

    func clearBuffer() {
        buffer.removeAll()
        accumulatedNewWindowsCount = 0 // Reset counter on clear
        print("SensorDataManager: Buffer cleared.")
        onBufferCountUpdate?(buffer.count)
    }

    func getCurrentSequenceForInference() -> MLMultiArray? {
        // This method should only be called when `buffer.count` is exactly `sequenceLength`
        // due to the condition in `addWindowsForInferenceTrigger`.
        guard buffer.count == sequenceLength else {
            print("SensorDataManager: Error - getCurrentSequenceForInference called when buffer.count (\(buffer.count)) != sequenceLength (\(sequenceLength)).")
            return nil
        }
        
        let currentSequence = buffer // The buffer already holds exactly `sequenceLength` windows

        let shape: [NSNumber] = [
            NSNumber(value: FIXED_N_BATCH_SIZE),
            NSNumber(value: sequenceLength),
            NSNumber(value: channels),
            NSNumber(value: windowLength)
        ]

        do {
            let multiArray = try MLMultiArray(shape: shape, dataType: .float32)

            for t_idx in 0..<sequenceLength {
                let currentWindow = currentSequence[t_idx]

                for c_idx in 0..<channels {
                    for l_idx in 0..<windowLength {
                        let bufferFlatIndex = c_idx * windowLength + l_idx
                        let mlMultiArrayIndex = [
                            0, // Batch dimension (N)
                            t_idx, // Time dimension (T)
                            c_idx, // Channel dimension (C)
                            l_idx  // Window length dimension (L)
                        ] as [NSNumber]
                        multiArray[mlMultiArrayIndex] = NSNumber(value: currentWindow[bufferFlatIndex])
                    }
                }
            }

            print("SensorDataManager: Successfully created MLMultiArray for inference (shape \(multiArray.shape)).")
            return multiArray
        } catch {
            print("SensorDataManager: Error creating MLMultiArray: \(error.localizedDescription)")
            return nil
        }
    }
}
#endif
