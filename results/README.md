# Measurements

The benchmark ran on a Mac Studio with Apple M3 Ultra. Each 480p result is a single text-to-video run with seed 20261023. Each output has 73 frames at 24 fps (3.042 seconds) and is 854×480. Prompts and seeds match within a scene, except that the cycling retry explicitly changes the prompt to specify rightward movement.

The timing window runs from ComfyUI execution start to execution success. It includes model loading during execution, text encoding, denoising, audio and video decoding, resizing or cropping, and encoding and saving the MP4. It excludes the one-time BF16 merge, ComfyUI startup, and time in the queue. The rate divides this window by the video duration. A value of 30 means 30 seconds of wall time per second of output.

PDMD 2-step generated at 768×432 before bicubic upscaling. Fast Turbo 4-step generated at 576×320 before bicubic upscaling. Native-resolution trials generated at 864×480 and were center-cropped to 854×480. Comparisons at different generation resolutions also change the latent grid and can produce different framing and motion.

| Scene | PDMD 2-step, 768×432 | Reference |
| --- | ---: | ---: |
| Window portrait | 25.75 | 57.50 (native Turbo 4-step) |
| Cycling | 29.39 | 28.69 (fast Turbo 4-step) |
| Cycling, revised direction prompt | 30.03 | 29.39 (original prompt, same model and settings) |
| Product label | 30.62 | 28.57 (fast Turbo 4-step) |
| Toy car | 25.92 | 54.23 (native Turbo 4-step) |
| Pottery | 25.69 | 24.74 (fast Turbo 4-step) |

All table entries are wall seconds per output second. The native-resolution PDMD toy-car run took 37.50 seconds per output second. [Full timing data](benchmark.csv)

The window portrait retained visible freckles and the requested blink, head turn and closed-mouth smile. The first PDMD cycling clip kept both wheels visible but moved in the wrong direction. The revised prompt corrected direction and still cropped part of the wheels. In the product scene, the 2-step clip produced an A without a crossbar. The 4-step clip showed clearer front text and extra lettering on the lid. Apple Vision detected the exact requested word in 55 of 73 frames for PDMD and 73 of 73 frames for Turbo. Recognition counts are a diagnostic and visual checks are needed for lettering and composition. [OCR summary](product-ocr-summary.csv)

The decoder checks cover 12 clips and record dimensions, frame cadence, sampled near-black frames, identical adjacent luma frames, and audio track properties. Prompt adherence and visual quality were judged separately from the structural checks. Audio and video durations were checked for agreement within 0.1 seconds. The timing and visual observations are limited to the tested prompts, seed, clip length, and hardware. [Technical measurements](media-checks.csv)
