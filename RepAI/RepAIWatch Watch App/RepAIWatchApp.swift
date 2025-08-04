import SwiftUI

@main
struct RepAIWatchApp: App {
    // Create an instance of your SharedSessionManager as a StateObject.
    // This tells SwiftUI to keep this object alive for the lifetime of the app
    // and to observe its @Published properties for changes.
    @StateObject private var sessionManager = SharedSessionManager.shared

    var body: some Scene {
        WindowGroup {
            WatchContentView()
                // Provide the sessionManager instance to the environment.
                // Any child view using @EnvironmentObject can then access it.
                .environmentObject(sessionManager)
        }
    }
}
