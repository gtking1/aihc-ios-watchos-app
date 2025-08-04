// WatchContentView.swift

import SwiftUI
import Combine
import WatchConnectivity
import WatchKit

let WATCH_WINDOWS_TO_SEND_BATCH = 8

struct WatchContentView: View {
    @EnvironmentObject var sessionManager: SharedSessionManager

    // SensorDataCollector should be initialized with dependencies it needs,
    // especially how to send data and how to handle replies.
    @StateObject var dataCollector: SensorDataCollector // Manages sensor data and sending

    // This @State is primarily for local UI control (e.g., button text)
    // and triggers calls to dataCollector.start/stop.
    // It also reflects the dataCollector's collecting state.
    @State private var isSessionActiveUI: Bool = false // Use a distinct name for UI state

    init() {
        _dataCollector = StateObject(wrappedValue: SensorDataCollector(
            // Pass the sendMessage function and a prediction callback
            sendData: { windows, replyHandler, errorHandler in
                SharedSessionManager.shared.sendToCompanion(
                    message: ["sensorData": windows],
                    replyHandler: replyHandler,
                    errorHandler: errorHandler
                )
            },
            onPredictionReceived: { prediction in
                // This closure is called when SensorDataCollector receives a prediction reply
                DispatchQueue.main.async {
                    SharedSessionManager.shared.lastPrediction = prediction
                    if prediction == 1.0 {
                        WKInterfaceDevice.current().play(.directionDown)
                    }
                    SharedSessionManager.shared.hasReceivedPrediction = true
                    SharedSessionManager.shared.lastPredictionTimestamp = Date()
                    print("WatchContentView: Prediction received callback executed. Prediction: \(prediction)")
                }
            }
        ))
    }

    var body: some View {
        VStack {
            // WCSession Connection Status (from SharedSessionManager)
            Text("\(sessionManager.connectionStatus)")
                .font(.caption2)
                .padding(.bottom, 2)

            // Watch's Internal Sensor Collection Status (from SensorDataCollector)
            Text("\(dataCollector.collectionStatus)")
                .font(.caption2)
                .fontWeight(.bold)
                .foregroundColor(dataCollector.isCollecting ? .green : .red)
                .padding(.bottom, 2)

            // UPDATED: Display total raw samples in buffer
            Text("Samples: \(dataCollector.currentRawBufferSampleCount)")
                .font(.caption2)

            // NEW: Indicates if enough raw data is present to form another window
            Text("Ready for Window: \((dataCollector.currentRawBufferSampleCount - dataCollector.currentReadIndex) >= WINDOW_LENGTH ? "Yes" : "No")")
                .font(.caption2)

            // Windows accumulated in the batch before sending
            Text("Batch Windows: \(dataCollector.windowsInBatchCount) / \(WATCH_WINDOWS_TO_SEND_BATCH)")
                .font(.caption2)
                .padding(.bottom, 2)

            Text("Last Msg: \(sessionManager.lastReceivedMessage)")
                .font(.caption2)

            // Display Prediction
            if sessionManager.hasReceivedPrediction {
                Text("Prediction: \(sessionManager.lastPrediction, specifier: "%.0f")")
                    .font(.headline)
                    .padding(.top, 5)
                if let timestamp = sessionManager.lastPredictionTimestamp {
                    Text("(\(timestamp, formatter: itemFormatter))")
                        .font(.caption2)
                        .foregroundColor(.gray)
                }
            } else {
                Text("No prediction yet.")
                    .font(.caption)
                    .padding(.top, 5)
            }

            Spacer()

            Button(dataCollector.isCollecting ? "Stop Session" : "Start Session") {
                if dataCollector.isCollecting {
                    stopSession()
                } else {
                    startSession()
                }
            }
            .font(.caption)
            .buttonStyle(.borderedProminent)
            .tint(dataCollector.isCollecting ? .red : .green)
        }
        .padding()
        // IMPORTANT: Ensure collection and session stop when view disappears
        .onDisappear {
            print("Watch: WatchContentView disappeared. Ensuring session is stopped.")
            stopSession()
        }
        .onChange(of: dataCollector.isCollecting) { _, newValue in
            // Keep the UI button state in sync with the data collector's actual state
            isSessionActiveUI = newValue
        }
    }

    private func startSession() {
        print("Watch: Starting new session. isSessionActiveUI = true")
        sessionManager.lastPrediction = 0.0 // Reset previous prediction on new session
        sessionManager.hasReceivedPrediction = false // Reset prediction status

        // 1. Send a reset message to the iPhone
        sessionManager.sendToCompanion(message: ["resetSession": true]) { reply in
            print("Watch (Reply for Reset): Received reply: \(reply)")
            // Start actual sensor collection ONLY AFTER iPhone confirms reset
            // Or if you want to be robust, check for a specific success message from iPhone
            if (reply["message"] as? String)?.contains("reset and ready") ?? false {
                self.dataCollector.startCollecting()
                // The dataCollector's @Published isCollecting will update the UI via onChange
            } else {
                print("Watch: iPhone did not confirm reset successfully. Not starting data collection.")
                self.stopSession() // Treat as a failure to start
            }
        } errorHandler: { error in
            print("Watch: Error sending reset message: \(error.localizedDescription)")
            self.stopSession() // Stop session if reset fails to send
        }
    }

    private func stopSession() {
        print("Watch: Stop Session called.")

        // Stop sensor data collection first
        dataCollector.stopCollecting()
        // The dataCollector's @Published isCollecting will update the UI via onChange

        // Immediately clear the prediction display on the Watch
        sessionManager.hasReceivedPrediction = false
        sessionManager.lastPrediction = 0.0
        sessionManager.lastPredictionTimestamp = nil

        // Send a message to the iPhone to indicate the session has stopped
        sessionManager.sendToCompanion(message: ["stopSession": true]) { reply in
            print("Watch (Stop Reply): Received reply for stop: \(reply)")
        } errorHandler: { error in
            print("Watch (Stop Error): Error sending stop message: \(error.localizedDescription)")
        }
        print("Watch: Session stopped.")
    }

    private let itemFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateStyle = .none
        formatter.timeStyle = .medium
        return formatter
    }()
}
