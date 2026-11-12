# H3 480p Benchmark

Reproduce a MiniMax H3 text-to-video benchmark on Apple Silicon: merge two distilled adapters into separate BF16 checkpoints, render the same prompts through ComfyUI, and compare timing, frames, and audio tracks.

On a Mac Studio with a 60-core M3 Ultra and 256 GiB unified memory, PDMD generated at 768×432 and upscaled to 854×480. The portrait took **25.75 seconds of wall time per second of video**. Across six prompts and revisions the same preset took **25.69–30.62 seconds per output second**. The original 480-line PDMD car run took 37.50 seconds per output second. Direction and lettering still needed review. See the [results and quality notes](results/README.md) for the 14 trials.

本仓库提供 M3 Ultra 上的 H3 480p 复现流程、两种 LoRA 的本地权重合并、测试提示词、画质检查和测速。速度按生成每秒视频需要的实际时间计算。快速版先生成 768×432，再放大到 854×480；实测并非每个场景都能压到 30 秒以内。

## Environment

- [ComfyUI v0.37.0](https://github.com/Comfy-Org/ComfyUI/tree/73c9bad4d21e7addbe1d13bc92eee0f1431b017d) with stock H3 nodes. The original trials used this commit with two local MPS guards in generic attention paths. A separate smoke run on a clean checkout rendered the PDMD portrait in 82.22 seconds and verified 73 frames and stereo audio. Use `--use-pytorch-cross-attention` as in both runs.
- Tested conversion stack: Python 3.13.12, PyTorch 2.13.0, comfy-kitchen 0.2.35, safetensors 0.8.0, NumPy 2.5.2. Run the conversion scripts with the same Python environment as ComfyUI.
- Allow about 200 GB of disk for the inputs, all three converted files, ComfyUI, and short render output. The encoder and merged checkpoints use about 52 GB and 40 GB per file, respectively. The tested 256 GiB machine is not a measured minimum.

## Model files

Obtain the checkpoints independently from the linked upstream pages under their applicable terms. The repository's MIT license covers its original code and documentation. The [MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE) governs H3 weights and their derivatives and specifies territorial restrictions on generated output. The public repository contains source and measurements. It does not distribute model weights or generated media.

| Input | Source | SHA-256 |
| --- | --- | --- |
| FL2VA pruned INT8 ConvRot | [Comfy-Org](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors) | `e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a` |
| Qwen3-VL H3 NVFP4/AWQ | [Comfy-Org](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors) | `35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6` |
| H3 video VAE FP16 | [Comfy-Org](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/vae/minimax_h3_video_vae_fp16.safetensors) | `7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522` |
| H3 audio VAE FP32 | [Comfy-Org](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/vae/minimax_h3_audio_vae_fp32.safetensors) | `8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48` |
| PDMD 2-NFE LoRA, ComfyUI conversion | [Iwannapose](https://huggingface.co/Iwannapose/minimax_h3_pdmd_2nfe_comfyui/blob/3acb11d6f9bc65e68bb907a5c6b596cd719c4360/minimax_h3_pdmd_2nfe_comfyui.safetensors) | `5feaeef13df31f98f36364a95e7ce9e8ff0afd71570d1c53d7c3be599e3a0f22` |
| LightX2V Turbo 4-step LoRA, ComfyUI conversion | [Kijai](https://huggingface.co/Kijai/MiniMax-H3_comfy/blob/main/loras/minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy.safetensors) | `fc9b6500f0331fe925b004738baaa31bd34104741c8bf9334816f9ac3005c8c1` |

Credit: [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3), [PDMD](https://github.com/ZeamoxWang/pdmd) and its [original adapter](https://huggingface.co/pdmd2026/pdmd_2NFE_lora), [LightX2V](https://huggingface.co/lightx2v/Minimax-h3-Turbo), and [ComfyUI](https://github.com/Comfy-Org/ComfyUI). The two ComfyUI adapter conversions use different alpha conventions. The Turbo conversion already contains its `0.125` factor.

## Reproduce a run

Install ComfyUI's `requirements.txt` in a Python environment and activate that environment in the terminals used for conversion and rendering. From this repository, set `COMFY_DIR` to the absolute path of your ComfyUI checkout and `INPUTS` to the directory containing the downloaded checkpoints. Keep the upstream filenames above. Place the two VAE files in `$COMFY_DIR/models/vae/`.

```sh
python scripts/predequant_text_encoder.py \
  "$INPUTS/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors" \
  "$COMFY_DIR/models/text_encoders/qwen3vl_32b_minimax_h3_predequant_bf16.safetensors"

python scripts/merge_checkpoint.py \
  "$INPUTS/minimax_h3_fl2va_pruned_int8_convrot.safetensors" \
  "$INPUTS/minimax_h3_pdmd_2nfe_comfyui.safetensors" \
  "$COMFY_DIR/models/diffusion_models/minimax_h3_fl2va_pdmd2_merged_bf16.safetensors" \
  --adapter-format pdmd

python scripts/merge_checkpoint.py \
  "$INPUTS/minimax_h3_fl2va_pruned_int8_convrot.safetensors" \
  "$INPUTS/minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy.safetensors" \
  "$COMFY_DIR/models/diffusion_models/minimax_h3_fl2va_turbo4_merged_bf16.safetensors" \
  --adapter-format turbo4
```

Start a local ComfyUI server in another terminal with its Python environment active. The benchmark places the text encoder on CPU and the denoiser and VAEs on MPS.

```sh
cd "$COMFY_DIR"
PYTORCH_ENABLE_MPS_FALLBACK=1 python main.py \
  --listen 127.0.0.1 --port 8188 --disable-api-nodes --disable-auto-launch \
  --disable-all-custom-nodes --use-pytorch-cross-attention --cache-none
```

Run from this repository:

```sh
python scripts/benchmark.py portrait pdmd
python scripts/benchmark.py portrait native
swift tools/make_comparison.swift runs comparisons portrait
swift tools/check-clips.swift runs/portrait-pdmd.mp4 runs/portrait-native.mp4
```

The runner saves an MP4, the submitted API workflow, and timing JSON under `runs/`. `--dry-run` exports the graph without a running server. Use `python scripts/benchmark.py --help` for the scene presets, alternate model filenames, and server URL. To check every frame of the product's `NOVA` label with macOS Vision, render `product` and run `swift tools/check-product-text.swift --all runs/product-pdmd.mp4`. Review the generated frames as well as the OCR results.

The timings cover execution through video save. One-time conversions, queueing and transfer back to the runner are outside that window. A timing of 30 means 30 seconds of wall time per second of video.
