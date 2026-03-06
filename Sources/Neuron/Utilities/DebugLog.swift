//
//  DebugLog.swift
//  Neuron
//
//  Session instrumentation for GPU vs CPU performance debugging.
//

import Foundation

enum DebugLog {
  private static let path = "/Users/williamvabrinskas/Documents/xcode/Neuron/.cursor/debug-660a19.log"
  private static let lock = NSLock()

  static func write(location: String, message: String, data: [String: Any], hypothesisId: String) {
    let payload: [String: Any] = [
      "sessionId": "660a19",
      "location": location,
      "message": message,
      "data": data,
      "timestamp": Int(Date().timeIntervalSince1970 * 1000),
      "hypothesisId": hypothesisId
    ]
    guard let json = try? JSONSerialization.data(withJSONObject: payload),
          let line = String(data: json, encoding: .utf8) else { return }
    let lineWithNewline = line + "\n"
    guard let lineData = lineWithNewline.data(using: .utf8) else { return }
    lock.lock()
    defer { lock.unlock() }
    if !FileManager.default.fileExists(atPath: path) {
      FileManager.default.createFile(atPath: path, contents: nil)
    }
    if let handle = FileHandle(forUpdatingAtPath: path) {
      handle.seekToEndOfFile()
      handle.write(lineData)
      try? handle.close()
    }
  }
}
