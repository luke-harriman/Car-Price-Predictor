
import SwiftUI
import AVFoundation
import Foundation
import AVFoundation

struct ContentView: View {
    @StateObject private var recorder = AudioRecorder()
    @StateObject private var networkManager = NetworkManager()
    @State private var predictedPrice: String?
    @State private var isRecording = false
    @State private var recordingTime: TimeInterval = 0
    @State private var timer: Timer?
    @State private var errorMessage: String? = nil
    @State private var micButtonPressed = false // Separate state for the microphone button press
    @State private var startButtonPressed = false // Separate state for the start button press
    
    var body: some View {
        ZStack {
            // Red to green gradient background
            LinearGradient(gradient: Gradient(colors: [Color.blue, Color.green]), startPoint: .top, endPoint: .bottom)
                .edgesIgnoringSafeArea(.all)
            
            VStack {
                Text("")
                    .font(.largeTitle)
                    .fontWeight(.bold) // Make the font bolder
                    .foregroundColor(.white)
                    .shadow(color: .black, radius: 2, x: 0, y: 1) // Add shadow to stand out
                    .padding()

                ZStack {
                    Circle()
                        .fill(isRecording ? Color.green : Color.red)
                        .frame(width: 200, height: 200)
                        .shadow(radius: 10)
                    Image(systemName: "mic.fill")
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(width: 80, height: 80)
                        .clipped()
                        .foregroundColor(.white)
                        .padding()
                        .scaleEffect(micButtonPressed ? 0.9 : 1.0)
                        .onTapGesture {
                            if self.isRecording {
                                self.stopRecording()
                            } else {
                                self.startRecording()
                            }
                        }
                        .onLongPressGesture(minimumDuration: 0.1, pressing: { inProgress in
                            withAnimation(.easeInOut(duration: 0.2)) {
                                self.micButtonPressed = inProgress
                            }
                        }, perform: {})
                }
                .shadow(color: isRecording ? Color.green : Color.black, radius: isRecording ? 20 : 10, x: 0, y: 10)
                
                Text(formatTime(recordingTime))
                    .font(.system(.title, design: .monospaced))
                    .foregroundColor(.white)
                    .shadow(color: .black, radius: 2, x: 0, y: 1) // Add shadow to stand out
                    .padding()
                
                // Display error message or predicted price
                if let error = errorMessage {
                    Text(error)
                        .font(.title) // Set the font size to match the predicted price
                        .fontWeight(.bold) // Make the font bolder
                        .foregroundColor(.red) // Error message in red color
                        .shadow(color: .black, radius: 2, x: 0, y: 1) // Add shadow to stand out
                } else if let price = predictedPrice {
                    Text("Predicted Price: \(price)")
                        .font(.title)
                        .fontWeight(.bold) // Make the font bolder
                        .foregroundColor(.white)
                        .shadow(color: .black, radius: 2, x: 0, y: 1) // Add shadow to stand out
                }
                
                Spacer()
                
                Button(action: {
                    self.startNewConversation()
                }) {
                    Text("Start New Conversation")
                        .foregroundColor(.white)
                        .padding()
                        .frame(maxWidth: .infinity)
                        .background(Color.red)
                        .cornerRadius(25)
                        .shadow(radius: 5)
                        .scaleEffect(startButtonPressed ? 0.95 : 1.0)
                }
                .padding(.horizontal, 30)
                .padding(.bottom, 20)
                .onLongPressGesture(minimumDuration: 0.1, pressing: { inProgress in
                    withAnimation(.easeInOut(duration: 0.2)) {
                        self.startButtonPressed = inProgress
                    }
                }, perform: {})
            }
        }
    }
    func startRecording() {
        isRecording = true
        recordingTime = 0
        timer = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { _ in
            self.recordingTime += 1
        }
        recorder.startRecording()
    }
    
    func stopRecording() {
        isRecording = false
        timer?.invalidate()
        timer = nil
        recorder.stopRecording()
        networkManager.uploadAudioFile { predictedPrice in
            DispatchQueue.main.async {
                if let price = predictedPrice {
                    self.predictedPrice = price
                    self.errorMessage = nil  // Clear any previous error message
                } else {
                    // Handle the nil case
                    self.errorMessage = "Prediction unavailable"
                    self.predictedPrice = nil  // Clear any previous prediction
                }
            }
        }
    }

    
    func startNewConversation() {
        isRecording = false
        recordingTime = 0
        timer?.invalidate()
        timer = nil
        predictedPrice = nil
        recorder.audioRecorder = nil // Assuming you have logic to properly reset and clear the recording
    }
    
    func formatTime(_ totalSeconds: TimeInterval) -> String {
        let hours = Int(totalSeconds) / 3600
        let minutes = Int(totalSeconds) / 60 % 60
        let seconds = Int(totalSeconds) % 60
        return String(format: "%02i:%02i:%02i", hours, minutes, seconds)
    }
}
