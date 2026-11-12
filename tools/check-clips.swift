import AVFoundation
import AudioToolbox
import Foundation

struct DecodeError: LocalizedError {
    let message: String
    var errorDescription: String? { message }
}

struct ClipReport: Encodable {
    let clip: String
    let width: Int
    let height: Int
    let frameRate: Double
    let frames: Int
    let videoSeconds: Double
    let frameCadenceBreaks: Int
    let consecutiveIdenticalLumaFrames: Int
    let nearBlackFrames: Int
    let luminanceChangeP50: Double
    let luminanceChangeP95: Double
    let luminanceChangeMax: Double
    let audioSampleRate: Double
    let audioChannels: Int
    let audioSeconds: Double
    let audioPeakDBFS: Double?
    let audioRMSDBFS: Double?
    let audioSamplesAtOrAboveFullScale: Int
    let nonFiniteAudioSamples: Int
    let checks: [String: Bool]
}

func readerOutput(
    asset: AVAsset, track: AVAssetTrack, settings: [String: Any]
) throws -> (AVAssetReader, AVAssetReaderTrackOutput) {
    let reader = try AVAssetReader(asset: asset)
    let output = AVAssetReaderTrackOutput(track: track, outputSettings: settings)
    output.alwaysCopiesSampleData = false
    guard reader.canAdd(output) else {
        throw DecodeError(message: "Cannot attach track to decoder")
    }
    reader.add(output)
    guard reader.startReading() else {
        throw reader.error ?? DecodeError(message: "Cannot start decoding")
    }
    return (reader, output)
}

func percentile(_ values: [Double], _ fraction: Double) -> Double {
    let sorted = values.sorted()
    guard !sorted.isEmpty else { return 0 }
    return sorted[Int(Double(sorted.count - 1) * fraction)]
}

func inspect(_ file: String) throws -> ClipReport {
    let asset = AVURLAsset(url: URL(fileURLWithPath: file))
    guard let videoTrack = asset.tracks(withMediaType: .video).first,
          let audioTrack = asset.tracks(withMediaType: .audio).first else {
        throw DecodeError(message: "Missing video or audio track")
    }

    let (videoReader, videoOutput) = try readerOutput(asset: asset, track: videoTrack, settings: [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
    ])
    var width = 0
    var height = 0
    var times: [Double] = []
    var previous: [UInt8] = []
    var identicalFrames = 0
    var nearBlackFrames = 0
    var changes: [Double] = []
    while let sample = videoOutput.copyNextSampleBuffer() {
        guard let frame = CMSampleBufferGetImageBuffer(sample) else {
            throw DecodeError(message: "Decoded video sample has no pixel buffer")
        }
        let frameWidth = CVPixelBufferGetWidthOfPlane(frame, 0)
        let frameHeight = CVPixelBufferGetHeightOfPlane(frame, 0)
        if times.isEmpty {
            width = frameWidth
            height = frameHeight
        }
        guard frameWidth == width, frameHeight == height else {
            throw DecodeError(message: "Decoded video changes dimensions")
        }
        times.append(CMSampleBufferGetPresentationTimeStamp(sample).seconds)

        CVPixelBufferLockBaseAddress(frame, .readOnly)
        guard let base = CVPixelBufferGetBaseAddressOfPlane(frame, 0) else {
            CVPixelBufferUnlockBaseAddress(frame, .readOnly)
            throw DecodeError(message: "Cannot read decoded luminance plane")
        }
        let rowStride = CVPixelBufferGetBytesPerRowOfPlane(frame, 0)
        let luma = base.assumingMemoryBound(to: UInt8.self)
        var current = [UInt8](repeating: 0, count: width * height)
        current.withUnsafeMutableBufferPointer { buffer in
            for row in 0..<height {
                buffer.baseAddress!.advanced(by: row * width)
                    .update(from: luma.advanced(by: row * rowStride), count: width)
            }
        }
        CVPixelBufferUnlockBaseAddress(frame, .readOnly)

        // 16 is black in video range. A threshold of 24 allows small codec noise.
        // This is a technical flag; intentionally dark frames need visual review.
        let sampleStride = 4
        var brightest = 0
        var difference = 0
        var pixels = 0
        for y in stride(from: 0, to: height, by: sampleStride) {
            for x in stride(from: 0, to: width, by: sampleStride) {
                let i = y * width + x
                let value = Int(current[i])
                brightest = max(brightest, value)
                if !previous.isEmpty {
                    difference += abs(value - Int(previous[i]))
                }
                pixels += 1
            }
        }
        if brightest <= 24 { nearBlackFrames += 1 }
        if !previous.isEmpty {
            if current == previous { identicalFrames += 1 }
            changes.append(Double(difference) / (Double(pixels) * 219))
        }
        previous = current
    }
    guard videoReader.status == .completed, !times.isEmpty else {
        throw videoReader.error ?? DecodeError(message: "Video did not decode completely")
    }
    let fps = Double(videoTrack.nominalFrameRate)
    guard fps > 0 else { throw DecodeError(message: "No nominal frame rate") }
    let cadenceBreaks = zip(times.dropFirst(), times).filter {
        abs(($0.0 - $0.1) - 1 / fps) > 0.001
    }.count

    let (audioReader, audioOutput) = try readerOutput(asset: asset, track: audioTrack, settings: [
        AVFormatIDKey: kAudioFormatLinearPCM,
        AVLinearPCMBitDepthKey: 32,
        AVLinearPCMIsFloatKey: true,
        AVLinearPCMIsNonInterleaved: false,
        AVLinearPCMIsBigEndianKey: false,
    ])
    var sampleRate = 0.0
    var channels = 0
    var audioFrames = 0
    var values = 0
    var peak = 0.0
    var squares = 0.0
    var atFullScale = 0
    var nonFinite = 0
    while let sample = audioOutput.copyNextSampleBuffer() {
        guard let format = CMSampleBufferGetFormatDescription(sample),
              let pcm = CMAudioFormatDescriptionGetStreamBasicDescription(format),
              let block = CMSampleBufferGetDataBuffer(sample) else {
            throw DecodeError(message: "Cannot read decoded PCM audio")
        }
        sampleRate = pcm.pointee.mSampleRate
        channels = Int(pcm.pointee.mChannelsPerFrame)
        audioFrames += CMSampleBufferGetNumSamples(sample)
        let byteCount = CMBlockBufferGetDataLength(block)
        var data = [Float32](repeating: 0, count: byteCount / MemoryLayout<Float32>.stride)
        let status = data.withUnsafeMutableBytes {
            CMBlockBufferCopyDataBytes(block, atOffset: 0, dataLength: byteCount, destination: $0.baseAddress!)
        }
        guard status == kCMBlockBufferNoErr else {
            throw DecodeError(message: "Cannot copy decoded PCM audio")
        }
        for value in data {
            if !value.isFinite {
                nonFinite += 1
                continue
            }
            let amplitude = abs(Double(value))
            peak = max(peak, amplitude)
            squares += amplitude * amplitude
            values += 1
            if amplitude >= 1 { atFullScale += 1 }
        }
    }
    guard audioReader.status == .completed, sampleRate > 0, values > 0 else {
        throw audioReader.error ?? DecodeError(message: "Audio did not decode completely")
    }
    let rms = sqrt(squares / Double(values))
    let videoSeconds = times.last! - times.first! + 1 / fps
    let audioSeconds = Double(audioFrames) / sampleRate

    return ClipReport(
        clip: file, width: width, height: height, frameRate: fps, frames: times.count,
        videoSeconds: videoSeconds, frameCadenceBreaks: cadenceBreaks,
        consecutiveIdenticalLumaFrames: identicalFrames, nearBlackFrames: nearBlackFrames,
        luminanceChangeP50: percentile(changes, 0.5),
        luminanceChangeP95: percentile(changes, 0.95),
        luminanceChangeMax: changes.max() ?? 0,
        audioSampleRate: sampleRate, audioChannels: channels, audioSeconds: audioSeconds,
        audioPeakDBFS: peak > 0 ? 20 * log10(peak) : nil,
        audioRMSDBFS: rms > 0 ? 20 * log10(rms) : nil,
        audioSamplesAtOrAboveFullScale: atFullScale, nonFiniteAudioSamples: nonFinite,
        checks: [
            "854x480": width == 854 && height == 480,
            "73_frames_at_24fps": times.count == 73 && abs(fps - 24) < 0.001 && cadenceBreaks == 0,
            "32kHz_stereo_audio": sampleRate == 32000 && channels == 2,
            "audio_is_not_digital_silence": peak > 0,
            "audio_length_within_0.1s_of_video": abs(audioSeconds - videoSeconds) <= 0.1,
            "no_non_finite_audio_samples": nonFinite == 0,
            "no_sampled_near_black_frames": nearBlackFrames == 0,
            "no_identical_adjacent_luma_frames": identicalFrames == 0,
        ]
    )
}

guard CommandLine.arguments.count > 1 else {
    FileHandle.standardError.write(Data("Usage: swift check-clips.swift clip.mp4 [clip.mp4 ...]\n".utf8))
    exit(2)
}
let encoder = JSONEncoder()
encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
var failed = false
for file in CommandLine.arguments.dropFirst() {
    do {
        let report = try inspect(file)
        FileHandle.standardOutput.write(try encoder.encode(report))
        FileHandle.standardOutput.write(Data("\n".utf8))
    } catch {
        FileHandle.standardError.write(Data("\(file): \(error.localizedDescription)\n".utf8))
        failed = true
    }
}
exit(failed ? 1 : 0)
