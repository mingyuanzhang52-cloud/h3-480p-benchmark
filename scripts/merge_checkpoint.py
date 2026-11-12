"""Merge a ComfyUI-format H3 adapter into the INT8 ConvRot FL2VA checkpoint."""

import argparse
import math
from pathlib import Path
import time

import torch
from comfy_kitchen.backends.eager.quantization import dequantize_int8_convrot_weight
from safetensors import safe_open

from safetensors_stream import bf16_bytes, layout, quant_configs, read_header, write_checkpoint


def find_patches(base, adapter, adapter_format):
    patches = {}
    unused = set(adapter.keys())
    for name in sorted(unused):
        if not name.endswith('.lora_A.weight'):
            continue
        prefix = name.removesuffix('.lora_A.weight')
        if not prefix.startswith('diffusion_model.'):
            raise ValueError(f'Expected a ComfyUI-format adapter key, got {name}')
        target = prefix.removeprefix('diffusion_model.') + '.weight'
        b_name = prefix + '.lora_B.weight'
        a_shape = adapter.get_slice(name).get_shape()
        b_shape = adapter.get_slice(b_name).get_shape()
        target_shape = base.get_slice(target).get_shape()
        if (
            len(a_shape) != 2 or len(b_shape) != 2
            or a_shape[0] != b_shape[1] or target_shape != [b_shape[0], a_shape[1]]
        ):
            raise ValueError(f'LoRA shape mismatch for {target}')
        unused.difference_update((name, b_name))
        if adapter_format == 'pdmd':
            alpha_name = prefix + '.alpha'
            if adapter.get_tensor(alpha_name).item() != a_shape[0]:
                raise ValueError(f'Expected alpha equal to LoRA rank for {target}')
            unused.remove(alpha_name)
        patches[target] = (name, b_name)
    if unused:
        raise ValueError(f'Unexpected tensor in {adapter_format} adapter: {min(unused)}')
    if not patches:
        raise ValueError('The adapter has no LoRA pairs')
    return patches


def merge(args):
    started = time.monotonic()
    with (
        args.base.open('rb') as raw,
        safe_open(args.base, framework='pt', device='cpu') as base,
        safe_open(args.adapter, framework='pt', device='cpu') as adapter,
    ):
        source_header, source_start = read_header(raw)
        quant = {}
        omitted = set()
        for prefix, config in quant_configs(base).items():
            name = prefix + 'weight'
            if config['format'] != 'int8_tensorwise' or not config.get('convrot'):
                raise ValueError(f'Expected INT8 ConvRot for {name}')
            if source_header[name]['dtype'] != 'I8':
                raise ValueError(f'Unexpected storage type for {name}')
            quant[name] = config['convrot_groupsize']
            omitted.update((prefix + 'comfy_quant', prefix + 'weight_scale'))
        if not quant:
            raise ValueError('The base checkpoint has no INT8 ConvRot weights')

        patches = find_patches(base, adapter, args.adapter_format)
        edited = quant.keys() | patches.keys()
        replacements = {}
        for name in edited:
            if name not in quant and source_header[name]['dtype'] != 'BF16':
                raise ValueError(f'Expected a BF16 LoRA target for {name}')
            shape = source_header[name]['shape']
            replacements[name] = (shape, math.prod(shape) * 2)
        header, size = layout(source_header, replacements, omitted)
        print(f'{len(quant)} ConvRot tensors, {len(patches)} LoRA targets, {size / 1e9:.2f} GB of output tensor data', flush=True)
        if args.dry_run:
            return

        count = 0

        def edited_chunks(name):
            nonlocal count
            weight = base.get_slice(name)
            rows = weight.get_shape()[0]
            scale = base.get_slice(name + '_scale') if name in quant else None
            if name in patches:
                a, b = (adapter.get_tensor(key).float() for key in patches[name])
            for start in range(0, rows, 1024):
                end = min(start + 1024, rows)
                if scale is None:
                    decoded = weight[start:end].float()
                else:
                    decoded = dequantize_int8_convrot_weight(weight[start:end], scale[start:end], quant[name])
                if name in patches:
                    decoded.addmm_(b[start:end], a)
                yield bf16_bytes(decoded)
            count += 1
            if count % 20 == 0:
                print(f'{count}/{len(edited)} tensors written', flush=True)

        write_checkpoint(raw, source_header, source_start, args.output, header,
                         lambda name: edited_chunks(name) if name in edited else None)
    print(f'{args.output} | {args.output.stat().st_size} bytes | {time.monotonic() - started:.1f} s')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base', type=Path, help='Pruned H3 FL2VA INT8 ConvRot safetensors')
    parser.add_argument('adapter', type=Path, help='ComfyUI-layout safetensors, with fused QKV')
    parser.add_argument('output', type=Path, help='New BF16 safetensors file')
    parser.add_argument('--adapter-format', choices=('pdmd', 'turbo4'), required=True,
                        help='PDMD has per-layer unit alpha/rank. The Turbo4 file has the scale baked into B.')
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--dry-run', action='store_true', help='Check tensor names, shapes, and scaling')
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f'Output already exists: {args.output}')
    if args.threads < 1:
        parser.error('--threads must be at least 1')
    torch.set_num_threads(args.threads)
    merge(args)


if __name__ == '__main__':
    main()
