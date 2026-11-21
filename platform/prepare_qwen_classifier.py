"""Explicit offline-adapter setup; downloading the base requires an opt-in flag."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).parent
MODEL = 'Qwen/Qwen3-0.6B'
REVISION = 'c1899de289a04d12100db370d81485cdf75e47ca'


def prepare(bundle, destination, download=False):
    destination = Path(destination)
    with zipfile.ZipFile(bundle) as archive:
        config = json.loads(archive.read('adapter/adapter_config.json'))
        if config.get('base_model_name_or_path') != MODEL or config.get('r') != 16:
            raise ValueError('Unexpected adapter model or rank')
        settings = json.loads(archive.read('selection.json'))['candidates']['qlora']
        allowed = {'adapter_config.json', 'adapter_model.safetensors', 'tokenizer.json',
                   'tokenizer_config.json', 'special_tokens_map.json', 'added_tokens.json',
                   'chat_template.jinja', 'merges.txt', 'vocab.json'}
        for name in allowed:
            member = 'adapter/' + name
            if member in archive.namelist():
                data = archive.read(member)
                target = destination / 'adapter' / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
        serving = {key: settings[key] for key in ('temperature', 'threshold', 'margin')}
        serving.update(model_id=MODEL, revision=REVISION,
                       adapter_sha256=hashlib.sha256(archive.read('adapter/adapter_model.safetensors')).hexdigest(),
                       selected_using='existing_v2_development_set',
                       training_quantization='nf4', serving_dtype='float32_cpu')
        (destination / 'serving.json').write_text(json.dumps(serving, indent=2))
    if download:
        from huggingface_hub import snapshot_download
        snapshot_download(MODEL, revision=REVISION, local_dir=destination / 'base', token=False,
                          allow_patterns=['config.json', 'generation_config.json', '*.safetensors',
                                          'tokenizer.json', 'tokenizer_config.json', 'merges.txt', 'vocab.json'])
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--destination', type=Path, default=ROOT / 'routing_models' / 'qwen-qlora-v2')
    parser.add_argument('--download-base', action='store_true')
    args = parser.parse_args()
    print(prepare(args.bundle, args.destination, args.download_base))
