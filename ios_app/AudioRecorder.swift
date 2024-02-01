import Foundation
import AVFoundation

class AudioRecorder: NSObject, ObservableObject, AVAudioRecorderDelegate {
    var audioRecorder: AVAudioRecorder?
    @Published var isRecording = false
    
    func startRecording() {
        let audioFilename = getDocumentsDirectory().appendingPathComponent("recording.wav")
        
        // Call this function here to ensure the previous recording is deleted
        removePreviousRecording(fileURL: audioFilename)

        print("AudioRecorder: Starting recording process.")
        
        let settings = [
            AVFormatIDKey: Int(kAudioFormatLinearPCM),
            AVSampleRateKey: 16000,
            AVNumberOfChannelsKey: 1,
            AVEncoderAudioQualityKey: AVAudioQuality.high.rawValue
        ]

        do {
            audioRecorder = try AVAudioRecorder(url: audioFilename, settings: settings)
            audioRecorder?.delegate = self
            audioRecorder?.record()
            isRecording = true
            print("AudioRecorder: Recording started successfully.")
        } catch {
            print("AudioRecorder: Recording failed to start. Error: \(error)")
            isRecording = false
        }
    }


    func stopRecording() {
        guard let recorder = audioRecorder else {
            print("AudioRecorder: No active audio recorder found when trying to stop.")
            return
        }

        recorder.stop()
        isRecording = false
        audioRecorder = nil
        print("AudioRecorder: Recording stopped.")
    }
    
    func removePreviousRecording(fileURL: URL) {
        do {
            if FileManager.default.fileExists(atPath: fileURL.path) {
                try FileManager.default.removeItem(at: fileURL)
                print("AudioRecorder: Previous recording removed.")
            }
        } catch {
            print("AudioRecorder: Failed to remove previous recording. Error: \(error)")
        }
    }

    func getDocumentsDirectory() -> URL {
        let urls = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
        print("AudioRecorder: Document directory - \(urls[0])")
        return urls[0]
    }

}
