"""Render a fixed H3 benchmark scene through ComfyUI's local HTTP API."""

import argparse
import json
from pathlib import Path
import shutil
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import uuid


SCENES = {
    'portrait': (
        "Photorealistic close portrait of a middle-aged woman with short dark hair and fine freckles, "
        "by a window with soft natural daylight. She blinks once and turns her head gently to her left, "
        "then gives a subtle closed-mouth smile. Fixed camera. Quiet room tone. No dialogue or music."
    ),
    'motion': (
        "Side tracking shot of a single cyclist in a bright yellow rain jacket riding left to right "
        "past a row of brick storefronts on a wet street. Both wheels remain visible. The bicycle "
        "rolls continuously, with a small splash from a shallow puddle. Overcast natural light, "
        "realistic motion, distant traffic and tire spray. No dialogue or music."
    ),
    'motion-right': (
        "Wide, side-on tracking shot. A cyclist in a bright yellow rain jacket faces screen right. "
        "The bicycle's front wheel is on the right side of the rider. The cyclist pedals toward the "
        "right edge of the image while the camera tracks right in parallel. Leave space in front of "
        "the rider and keep both full wheels visible above the lower edge. Wet street, brick storefronts "
        "and a small splash as the front wheel crosses a shallow puddle. Natural tire and street sounds."
    ),
    'product': (
        "Macro product shot of a matte black tin on a slowly rotating turntable. The label has a "
        "single clear white word NOVA in large uppercase letters and a thin silver line underneath. "
        "A soft highlight moves across the brushed metal rim. Camera locked off, dark studio background. "
        "Soft turntable hum. No dialogue or music."
    ),
    'car': (
        "A small red toy car drives smoothly from left to right on a sunlit wooden table. "
        "Static camera, crisp edges and soft shadows. Quiet room tone and a gentle motor sound."
    ),
    'pottery': (
        "Close-up of a potter's hands slowly shaping a clay cup on a spinning wheel. "
        "Fine grooves on the wet clay, soft window light, static camera. Quiet rhythmic wheel hum."
    ),
}
CONFIGS = {
    'pdmd': (768, 432, 2, 'pdmd2', 6, 'euler'),
    'pdmd-native': (864, 480, 2, 'pdmd2', 6, 'euler'),
    'native': (864, 480, 4, 'turbo4', 3, 'res_multistep'),
    'turbo4': (576, 320, 4, 'turbo4', 3, 'res_multistep'),
    'turbo4-native-2step': (864, 480, 2, 'turbo4', 3, 'res_multistep'),
}
FRAMES = 73
FPS = 24
VIDEO_SECONDS = FRAMES / FPS


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')


def build_graph(args, name):
    width, height, steps, model, audio_shift, sampler = CONFIGS[args.variant]
    template = Path(__file__).with_name('h3_t2v_api.json')
    graph = json.loads(template.read_text(encoding='utf-8'))
    graph['1']['inputs']['unet_name'] = args.checkpoint or f'minimax_h3_fl2va_{model}_merged_bf16.safetensors'
    graph['3']['inputs']['shift_audio'] = audio_shift
    graph['4']['inputs'].update(clip_name=args.text_encoder, device=args.text_encoder_device)
    graph['5']['inputs']['vae_name'] = args.video_vae
    graph['6']['inputs']['vae_name'] = args.audio_vae
    graph['7']['inputs'].update(width=width, height=height, length=FRAMES, prompt=SCENES[args.scene])
    graph['9']['inputs']['sampler_name'] = sampler
    graph['10']['inputs']['steps'] = steps
    graph['11']['inputs']['noise_seed'] = args.seed
    graph['15']['inputs']['fps'] = FPS
    graph['16']['inputs']['filename_prefix'] = name
    return graph


def fetch_video(server, descriptor, destination):
    query = urlencode({key: descriptor[key] for key in ('filename', 'subfolder', 'type')})
    temporary = None
    try:
        with urlopen(f'{server}/view?{query}', timeout=60) as response:
            signature = response.read(12)
            if signature[4:8] != b'ftyp':
                raise RuntimeError('ComfyUI did not return an MP4 for the reported video output')
            with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as output:
                temporary = Path(output.name)
                output.write(signature)
                shutil.copyfileobj(response, output)
        temporary.replace(destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def render(args, name, graph):
    server = args.server.rstrip('/')
    request = Request(
        server + '/prompt',
        data=json.dumps({'prompt': graph, 'client_id': str(uuid.uuid4())}).encode(),
        headers={'Content-Type': 'application/json'},
    )
    started = time.monotonic()
    with urlopen(request, timeout=30) as response:
        job_id = json.load(response)['prompt_id']
    print(f'{name}: {job_id}', flush=True)

    result_path = args.output_dir / f'{name}.result.json'
    while time.monotonic() - started < args.timeout:
        with urlopen(f'{server}/history/{job_id}', timeout=20) as response:
            history = json.load(response)
        if job_id not in history:
            time.sleep(0.75)
            continue
        result = history[job_id]
        result['observed_wall_seconds'] = time.monotonic() - started
        if result['status']['status_str'] != 'success':
            save_json(result_path, result)
            raise RuntimeError(f'ComfyUI render failed. See {result_path}.')
        events = dict(result['status']['messages'])
        seconds = (events['execution_success']['timestamp'] - events['execution_start']['timestamp']) / 1000
        result.update(
            execution_seconds=seconds,
            video_seconds=VIDEO_SECONDS,
            wall_seconds_per_video_second=seconds / VIDEO_SECONDS,
            video_seconds_per_wall_second=VIDEO_SECONDS / seconds if seconds else None,
        )
        save_json(result_path, result)
        destination = args.output_dir / f'{name}.mp4'
        fetch_video(server, result['outputs']['16']['images'][0], destination)
        print(f'{destination} | {seconds:.3f} s | {seconds / VIDEO_SECONDS:.2f} s per second of video')
        return
    raise TimeoutError(f'Job {job_id} did not finish within {args.timeout:g} seconds. Check ComfyUI.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scene', choices=SCENES)
    parser.add_argument('variant', choices=CONFIGS)
    parser.add_argument('--server', default='http://127.0.0.1:8188', help='ComfyUI base URL')
    parser.add_argument('--output-dir', type=Path, default=Path('runs'))
    parser.add_argument('--checkpoint', help='Filename as listed by ComfyUI UNETLoader')
    parser.add_argument('--text-encoder', default='qwen3vl_32b_minimax_h3_predequant_bf16.safetensors')
    parser.add_argument('--text-encoder-device', choices=('default', 'cpu'), default='cpu')
    parser.add_argument('--video-vae', default='minimax_h3_video_vae_fp16.safetensors')
    parser.add_argument('--audio-vae', default='minimax_h3_audio_vae_fp32.safetensors')
    parser.add_argument('--seed', type=int, default=20261023)
    parser.add_argument('--timeout', type=float, default=900, help='Render wait limit in seconds')
    parser.add_argument('--dry-run', action='store_true', help='Write the workflow JSON without submitting it')
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error('--timeout must be greater than zero')
    name = f'{args.scene}-{args.variant}'
    graph = build_graph(args, name)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    workflow_path = args.output_dir / f'{name}.workflow.json'
    save_json(workflow_path, graph)
    if args.dry_run:
        print(workflow_path)
        return
    try:
        render(args, name, graph)
    except HTTPError as error:
        detail = error.read().decode('utf-8', errors='replace')
        parser.exit(1, f'ComfyUI HTTP {error.code}: {detail}\n')
    except (URLError, RuntimeError, TimeoutError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
