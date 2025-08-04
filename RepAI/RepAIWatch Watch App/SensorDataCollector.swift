// SensorDataCollector.swift (for watchOS target)

import Foundation
import CoreMotion
import WatchConnectivity
import WatchKit

// MARK: - Constants
// (Omitting WINDOW_LENGTH, CHANNELS, WATCH_WINDOWS_TO_SEND_BATCH as per user request,
// as these are defined elsewhere in your project)

let SLIDING_STRIDE = 64 // Number of samples to slide the window by

let DEVICE_MOTION_UPDATE_INTERVAL = 0.01 // 100 Hz

// Normalization constants
let ACCEL_G_MIN: Float = -2.0
let ACCEL_G_MAX: Float = 2.0
let GYRO_DEG_MIN: Float = -250.0
let GYRO_DEG_MAX: Float = 250.0
let RADIANS_TO_DEGREES_FACTOR = Float(180.0 / Double.pi) // For converting gyro from rad/s to deg/s

// Delay for starting data collection
let START_DELAY_SECONDS: Double = 5.0 // 5 seconds delay

// Utility function for clamping
private func clamp(_ value: Float, min: Float, max: Float) -> Float {
    return fmaxf(min, fminf(value, max))
}


class SensorDataCollector: NSObject, ObservableObject {
    private let motionManager = CMMotionManager()
    
    // Buffers for raw, individual axis data
    private var rawAccelBufferX: [Double] = []
    private var rawAccelBufferY: [Double] = []
    private var rawAccelBufferZ: [Double] = []
    private var rawGyroBufferX: [Double] = []
    private var rawGyroBufferY: [Double] = []
    private var rawGyroBufferZ: [Double] = []

    @Published var currentReadIndex: Int = 0

    @Published var isCollecting = false
    // Removed direct sessionManager access here, it's now passed via closure
    // private let sessionManager = SharedSessionManager.shared

    // Buffer for accumulating complete transformed windows before sending
    private var windowAccumulationBuffer: [[Float]] = []

    // Property to hold the DispatchWorkItem for cancelling the delayed start
    private var collectionStartWorkItem: DispatchWorkItem?

    // NEW: Closures to inject dependencies for sending data and handling prediction replies
    private var sendDataToiPhoneClosure: (([[Float]], (([String: Any]) -> Void)?, ((Error) -> Void)?) -> Void)?
    private var onPredictionReceivedClosure: ((Double) -> Void)?


    @Published var collectionStatus: String = "Idle"
    @Published var currentRawBufferSampleCount: Int = 0
    @Published var windowsInBatchCount: Int = 0


    // NEW: Modified initializer to accept injected dependencies
    init(sendData: @escaping (([[Float]], (([String: Any]) -> Void)?, ((Error) -> Void)?) -> Void),
         onPredictionReceived: @escaping ((Double) -> Void)) {
        super.init() // Call super.init() first for NSObject
        self.sendDataToiPhoneClosure = sendData
        self.onPredictionReceivedClosure = onPredictionReceived
    }

    // You might still need a default init if you instantiate this class without parameters elsewhere.
    // If not, you can remove this. For now, keeping it but it might not be used.
    override init() {
        super.init()
        // Default behavior if closures are not provided (e.g., for testing or if not used with SwiftUI)
        // In a real app, you'd likely make the custom init() mandatory or provide a mock for testing.
        print("Warning: SensorDataCollector initialized without sendData or onPredictionReceived closures. Functionality may be limited.")
    }


    func startCollecting() {
        // NOTE: WCSession.default.isReachable check is now primarily handled by the caller (WatchContentView)
        // before calling dataCollector.startCollecting(), as it sends the reset message first.
        // However, keeping a guard here for robustness if called directly.
        guard WCSession.default.isReachable else {
            print("Watch: WCSession not reachable. Cannot start collection.")
            collectionStatus = "Not Reachable"
            return
        }
        
        guard !isCollecting else {
            print("Watch: Already collecting.")
            return
        }
        
        WKInterfaceDevice.current().play(.start) // A short, firm vibration and sound
        print("Watch: Haptic feedback played for button press (Type: .start).")

        // Clear all buffers on start (do this immediately)
        rawAccelBufferX.removeAll()
        rawAccelBufferY.removeAll()
        rawAccelBufferZ.removeAll()
        rawGyroBufferX.removeAll()
        rawGyroBufferY.removeAll()
        rawGyroBufferZ.removeAll()
        
        currentReadIndex = 0 // Reset read index
        currentRawBufferSampleCount = 0 // Reset raw sample count
        windowAccumulationBuffer.removeAll()
        windowsInBatchCount = 0

        isCollecting = true // Set to true immediately so the stop button can cancel the countdown
        collectionStatus = "Starting in \(Int(START_DELAY_SECONDS)) seconds..."
        print("Watch: Collection scheduled to start in \(START_DELAY_SECONDS) seconds.")

        // Create a DispatchWorkItem that will execute after the delay
        collectionStartWorkItem = DispatchWorkItem { [weak self] in
            guard let self = self else { return }

            // Crucial: Check if collection was stopped during the countdown
            // If isCollecting is false, it means stopCollecting() was called during the delay
            guard self.isCollecting else {
                print("Watch: Collection was stopped during countdown. Aborting start.")
                self.collectionStatus = "Idle" // Reset status for UI
                WKInterfaceDevice.current().play(.stop)
                return
            }
            
            WKInterfaceDevice.current().play(.directionUp) // A different, confirming vibration and sound
            print("Watch: Haptic feedback played for data collection start (Type: .directionUp).")

            // --- Core Change: Use CMDeviceMotion exclusively ---
            if self.motionManager.isDeviceMotionAvailable {
                self.motionManager.deviceMotionUpdateInterval = DEVICE_MOTION_UPDATE_INTERVAL
                self.motionManager.startDeviceMotionUpdates(to: .main) { [weak self] (data, error) in
                    guard let self = self, let motion = data else { return }
                    
                    // Extract User Acceleration (acceleration without gravity)
                    let userAcceleration = motion.userAcceleration
                    self.rawAccelBufferX.append(userAcceleration.x)
                    self.rawAccelBufferY.append(userAcceleration.y)
                    self.rawAccelBufferZ.append(userAcceleration.z)
                    
                    // Extract Rotation Rate (gyroscope data)
                    let rotationRate = motion.rotationRate
                    self.rawGyroBufferX.append(rotationRate.x)
                    self.rawGyroBufferY.append(rotationRate.y)
                    self.rawGyroBufferZ.append(rotationRate.z)

                    self.currentRawBufferSampleCount = self.rawAccelBufferX.count // Update based on collected samples
                    self.processSensorData()
                }
                self.collectionStatus = "Collecting (Device Motion)..."
                print("Watch: Device Motion data collection started.")
            } else {
                print("Watch: Device Motion not available. Cannot collect sensor data.")
                self.collectionStatus = "Device Motion Not Available"
                self.isCollecting = false // Turn off collecting if it can't start
                return
            }
        }

        // Schedule the work item to execute after the specified delay on the main queue
        DispatchQueue.main.asyncAfter(deadline: .now() + START_DELAY_SECONDS, execute: collectionStartWorkItem!)
    }

    func stopCollecting() {
        guard isCollecting else { return }

        // If the start is still pending (during the countdown), cancel it
        collectionStartWorkItem?.cancel()
        collectionStartWorkItem = nil // Clear the reference to the work item

        // Stop device motion updates if they have already started
        motionManager.stopDeviceMotionUpdates()
        
        isCollecting = false
        collectionStatus = "Idle"
        print("Watch: Sensor data collection stopped.")
        
        // Clear any unsent windows and raw data immediately on stop
        rawAccelBufferX.removeAll()
        rawAccelBufferY.removeAll()
        rawAccelBufferZ.removeAll()
        rawGyroBufferX.removeAll()
        rawGyroBufferY.removeAll()
        rawGyroBufferZ.removeAll()
        currentReadIndex = 0 // Reset read index
        currentRawBufferSampleCount = 0 // Reset raw sample count
        windowAccumulationBuffer.removeAll()
        windowsInBatchCount = 0
        WKInterfaceDevice.current().play(.start)
    }

    private func processSensorData() {
        // Loop to extract all possible windows with the current stride
        while (currentReadIndex + WINDOW_LENGTH) <= rawAccelBufferX.count {
            // Ensure all 6 raw buffers have enough samples for one window from currentReadIndex
            guard (currentReadIndex + WINDOW_LENGTH) <= rawAccelBufferY.count,
                  (currentReadIndex + WINDOW_LENGTH) <= rawAccelBufferZ.count,
                  (currentReadIndex + WINDOW_LENGTH) <= rawGyroBufferX.count,
                  (currentReadIndex + WINDOW_LENGTH) <= rawGyroBufferY.count,
                  (currentReadIndex + WINDOW_LENGTH) <= rawGyroBufferZ.count else {
                break // Not enough data across all buffers for a full window at currentReadIndex
            }

            // --- Data Extraction and Transformation for the current window ---
            var transformedWindow: [Float] = Array(repeating: 0.0, count: CHANNELS * WINDOW_LENGTH)

            for i in 0..<WINDOW_LENGTH {
                // Get data from our buffers. These values are now coming from CMDeviceMotion.
                // Accel values are userAcceleration (in g's)
                let watchAccelX = Float(rawAccelBufferX[currentReadIndex + i])
                let watchAccelY = Float(rawAccelBufferY[currentReadIndex + i])
                let watchAccelZ = Float(rawAccelBufferZ[currentReadIndex + i])
                // Gyro values are rotationRate (in radians/sec)
                let watchGyroX = Float(rawGyroBufferX[currentReadIndex + i])
                let watchGyroY = Float(rawGyroBufferY[currentReadIndex + i])
                let watchGyroZ = Float(rawGyroBufferZ[currentReadIndex + i])

                // Convert gyroscope from radians/sec to degrees/sec
                let watchGyroXDeg = watchGyroX * RADIANS_TO_DEGREES_FACTOR
                let watchGyroYDeg = watchGyroY * RADIANS_TO_DEGREES_FACTOR // Corrected typo here
                let watchGyroZDeg = watchGyroZ * RADIANS_TO_DEGREES_FACTOR

                // Apply your custom transformation: [y, -x, -z] for both accel and gyro
                // Apply clamping for normalization
                let imuAccelX = clamp(watchAccelY, min: ACCEL_G_MIN, max: ACCEL_G_MAX)
                let imuAccelY = clamp(-watchAccelX, min: ACCEL_G_MIN, max: ACCEL_G_MAX)
                let imuAccelZ = clamp(-watchAccelZ, min: ACCEL_G_MIN, max: ACCEL_G_MAX)

                let imuGyroX = clamp(watchGyroYDeg, min: GYRO_DEG_MIN, max: GYRO_DEG_MAX)
                let imuGyroY = clamp(-watchGyroXDeg, min: GYRO_DEG_MIN, max: GYRO_DEG_MAX)
                let imuGyroZ = clamp(-watchGyroZDeg, min: GYRO_DEG_MIN, max: GYRO_DEG_MAX)

                // Store in the `transformedWindow` in the correct channel order
                // Channel order: [Acc_X, Acc_Y, Acc_Z, Gyr_X, Gyr_Y, Gyr_Z]
                transformedWindow[i * CHANNELS + 0] = imuAccelX
                transformedWindow[i * CHANNELS + 1] = imuAccelY
                transformedWindow[i * CHANNELS + 2] = imuAccelZ
                transformedWindow[i * CHANNELS + 3] = imuGyroX
                transformedWindow[i * CHANNELS + 4] = imuGyroY
                transformedWindow[i * CHANNELS + 5] = imuGyroZ
            }

            // Add the complete transformed window to the accumulation buffer
            windowAccumulationBuffer.append(transformedWindow)
            windowsInBatchCount = windowAccumulationBuffer.count // Update published property

            print("Watch: Collected one transformed (overlapping) window. Accumulated: \(windowsInBatchCount) / \(WATCH_WINDOWS_TO_SEND_BATCH)")

            // Advance the read index for the next window by the sliding stride
            currentReadIndex += SLIDING_STRIDE

            // Trim the raw buffers if currentReadIndex moves past a significant amount of old data
            if currentReadIndex >= SLIDING_STRIDE {
                rawAccelBufferX.removeFirst(SLIDING_STRIDE)
                rawAccelBufferY.removeFirst(SLIDING_STRIDE)
                rawAccelBufferZ.removeFirst(SLIDING_STRIDE)
                rawGyroBufferX.removeFirst(SLIDING_STRIDE)
                rawGyroBufferY.removeFirst(SLIDING_STRIDE)
                rawGyroBufferZ.removeFirst(SLIDING_STRIDE)
                currentReadIndex -= SLIDING_STRIDE // Adjust index relative to the new start of buffer
            }

            // Check if enough windows are accumulated to send
            if windowAccumulationBuffer.count >= WATCH_WINDOWS_TO_SEND_BATCH {
                print("Watch: Sending \(windowAccumulationBuffer.count) accumulated (overlapping) windows to iPhone...")
                
                // --- MODIFIED: Use the injected sendDataToiPhoneClosure ---
                sendDataToiPhoneClosure?(windowAccumulationBuffer, { [weak self] reply in
                    guard let self = self else { return }
                    // THIS IS WHERE THE PREDICTION IS RECEIVED ON THE WATCH SIDE
                    // AND PASSED UP TO WatchContentView via the onPredictionReceivedClosure
                    
                    // Only process the reply if the Watch's session is still active
                    // (This check is now redundant if the sessionManager.isSessionActive is used in the closure)
                    // if self.sessionManager.isSessionActive { // Removed this check as it's now handled by the closure logic
                        if let prediction = reply["prediction"] as? Double {
                            self.onPredictionReceivedClosure?(prediction) // Call the callback for ContentView
                            print("Watch: Received prediction from iPhone: \(prediction)")
                        } else if let message = reply["message"] as? String {
                            // Update status if iPhone sends a message without a prediction
                            DispatchQueue.main.async {
                                self.collectionStatus = "iPhone: \(message)"
                            }
                            print("Watch: Reply message from iPhone: \(message)")
                        } else {
                            // Handle cases where reply might be empty or unexpected
                            DispatchQueue.main.async {
                                self.collectionStatus = "iPhone: Unkown Reply"
                            }
                            print("Watch: Received unknown reply format from iPhone.")
                        }
                    // } else { // Removed this check
                    //     print("Watch: Received late reply after session stopped. Discarding.")
                    // }
                }) { error in
                    print("Watch: Error sending data: \(error.localizedDescription)")
                    // Automatically stop the session if the companion becomes unreachable
                    if (error as NSError).domain == "WatchConnectivityError" && (error as NSError).code == 0 {
                        print("Watch: Companion not reachable, stopping collection automatically.")
                        self.stopCollecting()
                        // The sessionManager.isSessionActive update is now handled by SharedSessionManager directly
                        // self.sessionManager.isSessionActive = false // Removed direct access
                    }
                    DispatchQueue.main.async {
                        self.collectionStatus = "Error Sending: \(error.localizedDescription)"
                    }
                }

                // Clear the buffer after sending a batch
                windowAccumulationBuffer.removeAll()
                windowsInBatchCount = 0
            }
        }
    }
}
