//
//  RepAIApp.swift
//  RepAI
//
//  Created by Grant King on 7/2/25.
//

import SwiftUI

@main
struct RepAIApp: App {
    @StateObject private var sessionManager = SharedSessionManager.shared
    
    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(sessionManager)
        }
    }
}
