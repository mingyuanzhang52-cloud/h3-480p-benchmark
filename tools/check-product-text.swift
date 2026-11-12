import AVFoundation
import Foundation
import Vision

struct TextObservation: Encodable {
    let text: String
    let confidence: Float
    let bounds: [Double]
}

struct FrameTextReport: Encodable {
    let clip: String
    let frame: Int
    let actualTimeSeconds: Double
    let exactUppercaseNOVAObserved: Bool
    let observations: [TextObservation]
}

let fullClip = CommandLine.arguments.dropFirst().contains("--all")
let clips = CommandLine.arguments.dropFirst().filter { $0 != "--all" }
guard !clips.isEmpty else {
    FileHandle.standardError.write(Data("Usage: swift check-product-text.swift [--all] clip.mp4 [clip.mp4 ...]\n".utf8))
    exit(2)
}
let encoder = JSONEncoder()
encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
let frameNumbers: [Int64] = fullClip ? Array(0..<73) : [12, 24, 36, 48, 60]
var failed = false
for file in clips {
    let asset = AVURLAsset(url: URL(fileURLWithPath: file))
    let generator = AVAssetImageGenerator(asset: asset)
    generator.appliesPreferredTrackTransform = true
    generator.requestedTimeToleranceBefore = .zero
    generator.requestedTimeToleranceAfter = .zero

    for frame in frameNumbers {
        do {
            var actualTime = CMTime.zero
            let image = try generator.copyCGImage(
                at: CMTime(value: frame, timescale: 24), actualTime: &actualTime
            )
            let request = VNRecognizeTextRequest()
            request.usesCPUOnly = true
            request.recognitionLevel = .accurate
            request.recognitionLanguages = ["en-US"]
            request.usesLanguageCorrection = false
            try VNImageRequestHandler(cgImage: image).perform([request])
            let observations: [TextObservation] = (request.results ?? []).compactMap { result in
                guard let candidate = result.topCandidates(1).first else { return nil }
                let rect = result.boundingBox
                return TextObservation(
                    text: candidate.string, confidence: candidate.confidence,
                    bounds: [rect.minX, rect.minY, rect.width, rect.height]
                )
            }
            let report = FrameTextReport(
                clip: file, frame: Int(frame), actualTimeSeconds: actualTime.seconds,
                exactUppercaseNOVAObserved: observations.contains { $0.text == "NOVA" },
                observations: observations
            )
            FileHandle.standardOutput.write(try encoder.encode(report))
            FileHandle.standardOutput.write(Data("\n".utf8))
        } catch {
            FileHandle.standardError.write(Data("\(file) frame \(frame): \(error.localizedDescription)\n".utf8))
            failed = true
        }
    }
}
exit(failed ? 1 : 0)
