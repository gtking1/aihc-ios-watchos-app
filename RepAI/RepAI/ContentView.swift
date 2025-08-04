import SwiftUI

struct ContentView: View {
    @EnvironmentObject var sessionManager: SharedSessionManager
    // No additional @State variables are typically needed here as SharedSessionManager manages state
    
    private static var dateFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateStyle = .none // No date
        formatter.timeStyle = .medium // e.g., 3:00:35 PM
        return formatter
    }()

    var body: some View {
        VStack {
            Image(systemName: "iphone.gen3")
                .imageScale(.large)
                .foregroundStyle(.tint)
            Text("Rep Smarter, Not Harder")
                .font(.title3)

            Spacer()

            // WCSession Connection Status
            Text("Connection: \(sessionManager.connectionStatus)")
                .font(.caption)
                .padding(.bottom, 2)
            
            // --- NEW: iPhone's Internal Session Status ---
            Text(sessionManager.isSessionActive ? "Session: IN PROGRESS" : "Session: STOPPED")
                .font(.caption)
                .fontWeight(.bold)
                .foregroundColor(sessionManager.isSessionActive ? .green : .red)
                .padding(.bottom, 2)
            // --- END NEW ---

            Text("Last Message from Watch: \(sessionManager.lastReceivedMessage)")
                .font(.caption)

            if sessionManager.hasReceivedPrediction {
                Text("Prediction: \(sessionManager.lastPrediction, specifier: "%.0f")")
                    .font(.headline)
                    .padding(.top, 5)
            } else {
                Text("No prediction yet.")
                    .font(.caption)
                    .padding(.top, 5)
            }
            
            if let timestamp = sessionManager.lastPredictionTimestamp {
                                Text("Last Updated: \(timestamp, formatter: Self.dateFormatter)")
                                    .font(.caption)
                                    .foregroundColor(.gray)
                            } else {
                                Text("Last Updated: N/A")
                                    .font(.caption)
                                    .foregroundColor(.gray)
                            }
            
            Spacer()
        }
        .padding()
        // No .onDisappear or timer management needed here for iPhone's ContentView
        // as the session is managed by SharedSessionManager based on Watch's messages.
    }
}
