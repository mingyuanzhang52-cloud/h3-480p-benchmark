"""Expand the H3 Qwen3-VL NVFP4/AWQ text encoder into BF16 for CPU use."""

import argparse
import math
from pathlib import Path
import time

import torch
from comfy_kitchen.backends.eager.quantization import dequantize_nvfp4
from safetensors import safe_open

from safetensors_stream import bf16_bytes, layout, quant_configs, read_header, write_checkpoint


def round_up(value, multiple):
    return (value + multiple - 1) // multiple * multiple


def expand(args):
    started = time.monotonic()
    with args.source.open('rb') as raw, safe_open(args.source, framework='pt', device='cpu') as source:
        source_header, source_start = read_header(raw)
        configs = {}
        omitted = set()
        replacements = {}
        for prefix, config in quant_configs(source).items():
            name = prefix + 'weight'
            kind = config['format']
            rows, columns = source_header[name]['shape']
            if kind == 'nvfp4':
                columns *= 2
                scale_shape = source_header[prefix + 'weight_scale']['shape']
                if columns % 16 or scale_shape != [round_up(rows, 128), round_up(columns // 16, 4)]:
                    raise ValueError(f'Unexpected NVFP4 scale layout for {name}')
                omitted.add(prefix + 'weight_scale_2')
            elif kind != 'int8_tensorwise' or config.get('convrot'):
                raise ValueError(f'Unsupported text-encoder quantization for {name}: {kind}')
            omitted.update((prefix + 'comfy_quant', prefix + 'weight_scale'))
            awq_key = prefix + 'pre_quant_scale'
            if awq_key in source_header:
                if source_header[awq_key]['shape'] != [columns]:
                    raise ValueError(f'Unexpected AWQ input scale for {name}')
                omitted.add(awq_key)
            configs[name] = (prefix, kind)
            replacements[name] = ([rows, columns], math.prod((rows, columns)) * 2)
        if not configs:
            raise ValueError('The input encoder has no supported quantized tensors')

        metadata = {
            'conversion': 'Expanded NVFP4/AWQ and INT8 into BF16, with AWQ input scaling folded into weights.',
        }
        header, size = layout(source_header, replacements, omitted, metadata)
        if any(entry['dtype'] != 'BF16' for name, entry in header.items() if name != '__metadata__'):
            raise ValueError('The encoder contains unexpected non-BF16 auxiliary tensors')
        print(f'{len(configs)} quantized tensors, {size / 1e9:.2f} GB of output tensor data', flush=True)
        if args.dry_run:
            return

        count = 0

        def decoded_chunks(name):
            nonlocal count
            prefix, kind = configs[name]
            weight = source.get_slice(name)
            scale = source.get_slice(prefix + 'weight_scale')
            rows = weight.get_shape()[0]
            outer = source.get_tensor(prefix + 'weight_scale_2') if kind == 'nvfp4' else None
            awq_key = prefix + 'pre_quant_scale'
            awq = source.get_tensor(awq_key).float() if awq_key in omitted else None

            # Each swizzled NVFP4 scale tile covers 128 weight rows.
            for start in range(0, rows, 512):
                end = min(start + 512, rows)
                if kind == 'nvfp4':
                    decoded = dequantize_nvfp4(
                        weight[start:end], outer, scale[start:round_up(end, 128)],
                        output_type=torch.float32,
                    )
                else:
                    decoded = weight[start:end].float() * scale[start:end].float()
                if awq is not None:
                    decoded.mul_(awq)
                yield bf16_bytes(decoded)
            count += 1
            if count % 20 == 0:
                print(f'{count}/{len(configs)} tensors written', flush=True)

        write_checkpoint(raw, source_header, source_start, args.output, header,
                         lambda name: decoded_chunks(name) if name in configs else None)
    print(f'{args.output} | {args.output.stat().st_size} bytes | {time.monotonic() - started:.1f} s')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='H3 Qwen3-VL NVFP4/AWQ safetensors')
    parser.add_argument('output', type=Path, help='New BF16 safetensors file')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--dry-run', action='store_true', help='Check tensor names and shapes')
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f'Output already exists: {args.output}')
    if args.threads < 1:
        parser.error('--threads must be at least 1')
    torch.set_num_threads(args.threads)
    expand(args)


if __name__ == '__main__':
    main()
