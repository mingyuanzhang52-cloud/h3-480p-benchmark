import AppKit
import AVFoundation
import Foundation

struct Series {
    let path: String
    let title: String
    let generation: String
    let benchmark: String

    init(_ path: String, _ title: String, _ generation: String, _ benchmark: String) {
        self.path = path
        self.title = title
        self.generation = generation
        self.benchmark = benchmark
    }
}

struct Scene {
    let key: String
    let title: String
    let focus: String
    let rows: [Series]
}

let near = "768×432 放大至 854×480"
let low = "576×320 放大至 854×480"
let native = "864×480 裁至 854×480"
let pdmd = "PDMD · 2 步"
let turbo = "Turbo · 4 步"
let turboNative = "Turbo · 原生 4 步"

let scenes: [Scene] = [
    Scene(key: "car", title: "小车驶过桌面", focus: "观察轮辐、车身边缘和反光",
          rows: [
            Series("car-pdmd.mp4", pdmd, near, "car-pdmd"),
            Series("car-turbo4.mp4", turbo, low, "car-turbo4"),
            Series("car-native.mp4", turboNative, native, "car-native"),
          ]),
    Scene(key: "pottery", title: "手部与陶艺近景", focus: "观察手部轮廓、杯沿和湿泥纹理",
          rows: [
            Series("pottery-pdmd.mp4", pdmd, near, "pottery-pdmd"),
            Series("pottery-turbo4.mp4", turbo, low, "pottery-turbo4"),
          ]),
    Scene(key: "portrait", title: "窗边人像", focus: "观察五官、肤质、眨眼和转头前后的形态",
          rows: [
            Series("portrait-pdmd.mp4", pdmd, near, "portrait-pdmd"),
            Series("portrait-native.mp4", turboNative, native, "portrait-native"),
          ]),
    Scene(key: "motion", title: "雨天骑行", focus: "观察车轮形态、脚踏和水花",
          rows: [
            Series("motion-pdmd.mp4", pdmd, near, "motion-pdmd"),
            Series("motion-turbo4.mp4", turbo, low, "motion-turbo4"),
          ]),
    Scene(key: "motion-retry", title: "骑行方向复测", focus: "核对前轮位置、骑行方向和车轮形态",
          rows: [
            Series("motion-pdmd.mp4", "原提示词 · 2 步", near, "motion-pdmd"),
            Series("motion-right-pdmd.mp4", "明确方向 · 2 步", near, "motion-right-pdmd"),
          ]),
    Scene(key: "product", title: "转台产品与文字", focus: "观察 NOVA 拼写、字形稳定性和金属反光",
          rows: [
            Series("product-pdmd.mp4", pdmd, near, "product-pdmd"),
            Series("product-turbo4.mp4", turbo, low, "product-turbo4"),
          ]),
]

guard CommandLine.arguments.count >= 3 else {
    FileHandle.standardError.write(Data("Usage: swift make_comparison.swift RUN_DIR IMAGE_DIR [SCENE ...]\n".utf8))
    exit(2)
}
let runDirectory = URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
let outputDirectory = URL(fileURLWithPath: CommandLine.arguments[2], isDirectory: true)
try FileManager.default.createDirectory(at: outputDirectory, withIntermediateDirectories: true)

func speed(for row: Series) -> Double? {
    let url = runDirectory.appendingPathComponent("\(row.benchmark).result.json")
    guard let data = try? Data(contentsOf: url),
          let result = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
        return nil
    }
    if let seconds = result["wall_seconds_per_video_second"] as? Double {
        return seconds
    }
    guard let status = result["status"] as? [String: Any],
          let messages = status["messages"] as? [[Any]] else { return nil }
    var started: Double?
    var finished: Double?
    for message in messages where message.count == 2 {
        guard let event = message[0] as? String,
              let fields = message[1] as? [String: Any],
              let timestamp = fields["timestamp"] as? Double else { continue }
        if event == "execution_start" { started = timestamp }
        if event == "execution_success" { finished = timestamp }
    }
    if let started, let finished {
        return (finished - started) / 1000 / (73.0 / 24)
    }
    return nil
}

func color(_ rgb: UInt32) -> NSColor {
    NSColor(srgbRed: CGFloat((rgb >> 16) & 255) / 255,
            green: CGFloat((rgb >> 8) & 255) / 255,
            blue: CGFloat(rgb & 255) / 255, alpha: 1)
}

func makeSheet(_ scene: Scene) async throws {
    let canvasWidth = 1774
    let rowHeight = 279
    let canvasHeight = 154 + rowHeight * scene.rows.count + 56
    let margin: CGFloat = 40
    let firstFrameX: CGFloat = 282
    let frameWidth: CGFloat = 472
    let frameHeight: CGFloat = 265
    let frameGap: CGFloat = 18
    let timestamps: [Int64] = [12, 36, 60]

    var sampledFrames = [[CGImage]]()
    for row in scene.rows {
        let url = runDirectory.appendingPathComponent(row.path).standardizedFileURL
        let generator = AVAssetImageGenerator(asset: AVURLAsset(url: url))
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero
        var images = [CGImage]()
        for frame in timestamps {
            let sample = try await generator.image(at: CMTime(value: frame, timescale: 24))
            images.append(sample.image)
        }
        sampledFrames.append(images)
    }

    guard let bitmap = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: canvasWidth,
                                        pixelsHigh: canvasHeight, bitsPerSample: 8,
                                        samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
                                        colorSpaceName: .deviceRGB,
                                        bytesPerRow: 0, bitsPerPixel: 0),
          let context = NSGraphicsContext(bitmapImageRep: bitmap) else {
        throw CocoaError(.fileWriteUnknown)
    }
    NSGraphicsContext.saveGraphicsState()
    defer { NSGraphicsContext.restoreGraphicsState() }
    NSGraphicsContext.current = context
    context.imageInterpolation = .high

    func rect(_ x: CGFloat, _ y: CGFloat, _ width: CGFloat, _ height: CGFloat) -> NSRect {
        NSRect(x: x, y: CGFloat(canvasHeight) - y - height, width: width, height: height)
    }
    func fill(_ rgb: UInt32, _ x: CGFloat, _ y: CGFloat, _ width: CGFloat, _ height: CGFloat) {
        color(rgb).setFill()
        rect(x, y, width, height).fill()
    }
    func label(_ value: String, _ x: CGFloat, _ y: CGFloat, _ size: CGFloat,
               _ weight: NSFont.Weight, _ rgb: UInt32, centered: Bool = false) {
        let attributes: [NSAttributedString.Key: Any] = [
            .font: NSFont.systemFont(ofSize: size, weight: weight),
            .foregroundColor: color(rgb),
        ]
        let line = value as NSString
        let bounds = line.size(withAttributes: attributes)
        line.draw(in: rect(centered ? x - bounds.width / 2 : x, y, ceil(bounds.width), ceil(bounds.height)),
                  withAttributes: attributes)
    }

    fill(0x101722, 0, 0, CGFloat(canvasWidth), CGFloat(canvasHeight))
    label("MINIMAX H3  /  480p 画质对比", margin, 23, 18, .semibold, 0x80D8CA)
    label(scene.title, margin, 52, 42, .bold, 0xF4F7FC)
    label(scene.focus, margin, 104, 19, .regular, 0xB8C4D4)
    for (column, frame) in timestamps.enumerated() {
        let x = firstFrameX + CGFloat(column) * (frameWidth + frameGap)
        label(String(format: "%.2f 秒", Double(frame) / 24), x + frameWidth / 2, 113, 19, .semibold,
              0xDCE6F2, centered: true)
    }
    fill(0x324052, margin, 141, CGFloat(canvasWidth) - margin * 2, 1)

    for (index, row) in scene.rows.enumerated() {
        let y = 153 + CGFloat(index * rowHeight)
        let highlight = index == 0
        fill(highlight ? 0x182D34 : 0x1C2736, margin, y, 224, frameHeight)
        fill(highlight ? 0x64CBB9 : 0x8697AA, margin, y, 4, frameHeight)
        label(row.title, margin + 19, y + 27, 26, .semibold, 0xF4F7FC)
        label(row.generation, margin + 19, y + 70, 16, .medium, 0xB8C4D4)
        if let speed = speed(for: row) {
            label(String(format: "%.2f 秒", speed), margin + 19, y + 117, 31, .semibold,
                  highlight ? 0x80D8CA : 0xDCE6F2)
            label("每秒成片耗时", margin + 19, y + 156, 16, .regular, 0xB8C4D4)
        }

        for (column, image) in sampledFrames[index].enumerated() {
            let x = firstFrameX + CGFloat(column) * (frameWidth + frameGap)
            NSImage(cgImage: image, size: .zero)
                .draw(in: rect(x, y, frameWidth, frameHeight), from: .zero, operation: .sourceOver, fraction: 1)
        }
    }

    let footY = CGFloat(canvasHeight) - 39
    label("每列取同一时间点 · 各版本独立生成，构图可能不同", margin, footY, 17, .regular, 0xB8C4D4)
    let file = outputDirectory.appendingPathComponent("\(scene.key)-comparison.png")
    guard let png = bitmap.representation(using: .png, properties: [:]) else {
        throw CocoaError(.fileWriteUnknown)
    }
    try png.write(to: file, options: .atomic)
    print(file.standardizedFileURL.path)
}

let requested = Array(CommandLine.arguments.dropFirst(3))
let selected = scenes.filter { scene in
    if !requested.isEmpty { return requested.contains(scene.key) }
    return scene.rows.allSatisfy {
        FileManager.default.fileExists(atPath: runDirectory.appendingPathComponent($0.path).path)
    }
}
guard !selected.isEmpty && (requested.isEmpty || Set(selected.map(\.key)) == Set(requested)) else {
    FileHandle.standardError.write(Data("No matching scene with all required clips. Scenes: car, pottery, portrait, motion, motion-retry, product.\n".utf8))
    exit(2)
}
for scene in selected { try await makeSheet(scene) }

