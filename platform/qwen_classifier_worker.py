"""Offline adapter inference; JSON-lines transport, no training or downloads."""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                  HF_HUB_DISABLE_TELEMETRY='1', TOKENIZERS_PARALLELISM='false')


def load(directory):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from qlora_training.protocol import LABELS, encode_example, softmax

    directory = Path(directory)
    settings = json.loads((directory / 'serving.json').read_text())
    adapter = directory / 'adapter'
    if settings['model_id'] != 'Qwen/Qwen3-0.6B':
        raise ValueError('Unexpected classifier base model')
    if hashlib.sha256((adapter / 'adapter_model.safetensors').read_bytes()).hexdigest() != settings['adapter_sha256']:
        raise ValueError('Classifier adapter checksum mismatch')
    adapter_config = json.loads((adapter / 'adapter_config.json').read_text())
    if adapter_config.get('base_model_name_or_path') != settings['model_id']:
        raise ValueError('Classifier adapter and base do not match')
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(str(directory / 'base'), local_files_only=True,
                                             trust_remote_code=False)
    tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        str(directory / 'base'), local_files_only=True, trust_remote_code=False,
        use_safetensors=True, torch_dtype=torch.float32,
    )
    model = PeftModel.from_pretrained(base, str(adapter), local_files_only=True,
                                     is_trainable=False).eval()

    def predict(text):
        scores = []
        # Keep the exact six-completion likelihood scorer used by the experiment.
        with torch.inference_mode():
            for label in LABELS:
                example = encode_example(tokenizer, text, label, 512)
                ids = torch.tensor([example['input_ids']])
                targets = torch.tensor(example['labels'])
                length = int((targets != -100).sum().item())
                logits = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                               use_cache=False, logits_to_keep=length + 1).logits[0, :-1].float()
                values = torch.log_softmax(logits, dim=-1).gather(1, targets[-length:, None]).mean()
                scores.append(float(values.item()))
        probabilities = softmax(scores, settings['temperature'])
        first, second = sorted(range(len(LABELS)), key=probabilities.__getitem__, reverse=True)[:2]
        score = probabilities[first]
        margin = score - probabilities[second]
        return {'available': True, 'accepted': score >= settings['threshold'] and margin >= settings['margin'],
                'model': 'Qwen3-0.6B-QLoRA-v2', 'domain': LABELS[first],
                'score': round(score, 5), 'margin': round(margin, 5),
                'threshold': settings['threshold'], 'margin_threshold': settings['margin'],
                'selection': 'qwen_qlora_primary', 'score_kind': 'development_temperature_scaled_likelihood',
                'serving': 'local_float32_cpu', 'training_quantization': 'nf4',
                'adapter_sha256': settings['adapter_sha256']}
    return predict


def main():
    predictor = None
    for line in sys.stdin:
        try:
            with contextlib.redirect_stdout(sys.stderr):
                if predictor is None:
                    predictor = load(sys.argv[1])
                result = predictor(json.loads(line)['text'])
            response = {'ok': True, 'result': result}
        except Exception:
            response = {'ok': False, 'error': 'Fine-tuned classifier could not score this request'}
        print(json.dumps(response), flush=True)


if __name__ == '__main__':
    main()
