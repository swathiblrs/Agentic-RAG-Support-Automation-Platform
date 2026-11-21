"""Bounded resident process for the existing fine-tuned Qwen classifier."""
import atexit
import json
import os
from pathlib import Path
import queue
import subprocess
import threading

ROOT = Path(__file__).parent
LABELS = {'cloud', 'data', 'it', 'other', 'production', 'security'}


class QwenClassifier:
    def __init__(self):
        self.process = None
        self.responses = None
        self.lock = threading.Lock()

    def _stop(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            self.process.stdin.close()
            self.process.stdout.close()
        self.process = None

    def close(self):
        with self.lock:
            self._stop()

    def predict(self, text):
        unavailable = {'available': False, 'accepted': False, 'model': 'Qwen3-0.6B-QLoRA-v2',
                       'selection': 'qwen_qlora_primary',
                       'reason': 'Fine-tuned Qwen unavailable or input exceeds its 512-token budget'}
        directory = Path(os.getenv('QWEN_CLASSIFIER_DIR', str(ROOT / 'routing_models' / 'qwen-qlora-v2')))
        python = Path(os.getenv('QWEN_CLASSIFIER_PYTHON', str(ROOT / '.venv-training' / 'bin' / 'python')))
        required = ['serving.json', 'base/config.json', 'adapter/adapter_config.json',
                    'adapter/adapter_model.safetensors']
        if not python.is_file() or not all((directory / name).is_file() for name in required):
            return {**unavailable, 'reason': 'Local Qwen classifier files or runtime are missing; no automatic download'}
        if not self.lock.acquire(timeout=120):
            return {**unavailable, 'reason': 'Fine-tuned Qwen worker busy'}
        try:
            if self.process is None or self.process.poll() is not None:
                self._stop()
                self.responses = queue.Queue()
                self.process = subprocess.Popen(
                    [str(python), '-u', str(ROOT / 'qwen_classifier_worker.py'), str(directory)],
                    cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL, text=True, bufsize=1,
                    env=dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1'),
                )
                process, responses = self.process, self.responses

                def read():
                    try:
                        for line in process.stdout:
                            responses.put(line)
                    finally:
                        responses.put(None)
                threading.Thread(target=read, daemon=True).start()
            self.process.stdin.write(json.dumps({'text': text}) + '\n')
            self.process.stdin.flush()
            line = self.responses.get(timeout=120)
            if line is None:
                raise RuntimeError('Classifier process exited')
            response = json.loads(line)
            if not response.get('ok'):
                return unavailable
            result = response['result']
            if result.get('domain') not in LABELS or not isinstance(result.get('accepted'), bool):
                raise ValueError('Invalid classifier output')
            return result
        except (OSError, ValueError, KeyError, RuntimeError, queue.Empty):
            self._stop()
            return unavailable
        finally:
            self.lock.release()


worker = QwenClassifier()
atexit.register(worker.close)
predict = worker.predict
