
import Foundation

class NetworkManager: ObservableObject {
    func uploadAudioFile(completion: @escaping (String?) -> Void) {
        let url = URL(string: "Include the GCP Endpoint Here")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        
        let boundary = "Boundary-\(UUID().uuidString)"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        
        let audioFilename = AudioRecorder().getDocumentsDirectory().appendingPathComponent("recording.wav")
        guard let audioData = try? Data(contentsOf: audioFilename) else {
            completion(nil) // If audio data is not available, complete with nil
            return
        }
        
        var body = Data()
        
        // Append the file data with the correct multipart/form-data format
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"file\"; filename=\"\(audioFilename.lastPathComponent)\"\r\n".data(using: .utf8)!)
        body.append("Content-Type: audio/wav\r\n\r\n".data(using: .utf8)!)
        body.append(audioData)
        body.append("\r\n".data(using: .utf8)!)
        body.append("--\(boundary)--\r\n".data(using: .utf8)!)
        
        request.httpBody = body
        
        // Perform the request
        let task = URLSession.shared.dataTask(with: request) { data, response, error in
            guard let data = data, error == nil else {
                completion(nil)
                return
            }

            // Parse the JSON response
            do {
                if let jsonResponse = try JSONSerialization.jsonObject(with: data, options: []) as? [String: Any],
                   let predictionString = jsonResponse["prediction"] as? String {
                    completion(predictionString)
                } else {
                    completion(nil)
                }
            } catch {
                completion(nil)
            }
        }

        task.resume()
    }
    
    private func roundToNearest500(_ value: Double) -> Int {
        return Int(round(value / 500.0) * 500)
    }
}

// Helper extension to append Data in a URLRequest
extension Data {
    mutating func append(_ string: String) {
        if let data = string.data(using: .utf8) {
            append(data)
        }
    }
}
