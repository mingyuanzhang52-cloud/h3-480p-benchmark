"""Write a safetensors checkpoint one tensor or row block at a time."""

import json
import os
import struct
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open


def read_header(file):
    length = struct.unpack('<Q', file.read(8))[0]
    return json.loads(file.read(length)), 8 + length


def bf16_bytes(tensor):
    return tensor.to(torch.bfloat16).view(torch.uint16).contiguous().numpy().tobytes()


def quant_configs(checkpoint):
    return {
        key.removesuffix('comfy_quant'): json.loads(checkpoint.get_tensor(key).numpy().tobytes())
        for key in checkpoint.keys() if key.endswith('.comfy_quant')
    }


def layout(source_header, replacements, omitted, metadata=None):
    """Use the original tensor order and recalculate byte offsets."""
    header = {'__metadata__': metadata} if metadata else {}
    offset = 0
    for name, entry in source_header.items():
        if name == '__metadata__' or name in omitted:
            continue
        start, end = entry['data_offsets']
        shape, size = replacements.get(name, (entry['shape'], end - start))
        header[name] = {
            'dtype': 'BF16' if name in replacements else entry['dtype'],
            'shape': shape,
            'data_offsets': [offset, offset + size],
        }
        offset += size
    return header, offset


def write_checkpoint(source_file, source_header, source_start, output, header, convert):
    """Write and validate a temporary file before creating the destination."""
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(header, separators=(',', ':')).encode()
    encoded += b' ' * (-len(encoded) % 8)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix='.h3-', delete=False) as file:
            temporary = Path(file.name)
            file.write(struct.pack('<Q', len(encoded)))
            file.write(encoded)
            for name, entry in header.items():
                if name == '__metadata__':
                    continue
                start = file.tell()
                chunks = convert(name)
                if chunks is None:
                    first, last = source_header[name]['data_offsets']
                    source_file.seek(source_start + first)
                    remaining = last - first
                    while remaining:
                        block = source_file.read(min(remaining, 8 * 1024 * 1024))
                        if not block:
                            raise ValueError(f'Unexpected end of input at {name}')
                        file.write(block)
                        remaining -= len(block)
                else:
                    for block in chunks:
                        file.write(block)
                first, last = entry['data_offsets']
                if file.tell() - start != last - first:
                    raise ValueError(f'Incorrect output size for {name}')
        with safe_open(temporary, framework='pt') as written:
            if set(written.keys()) != set(header) - {'__metadata__'}:
                raise ValueError('Output checkpoint has unexpected tensor keys')
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
