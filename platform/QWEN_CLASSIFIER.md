# Fine-tuned Qwen Primary Domain Classifier

`run_local.py` defaults to `DOMAIN_CLASSIFIER=qwen_qlora`. Classification still happens before the parent LangGraph. The trained Qwen3-0.6B v2 adapter predicts cloud, data, IT, other, production or security. Existing IT category/priority classification is reused downstream, not retrained.

## Runtime

- `qwen_classifier.py` owns a bounded, resident JSON-lines subprocess, serialized with a lock and a 120-second wait.
- `qwen_classifier_worker.py` loads the pinned base and existing safetensors adapter with Transformers/PEFT. Runtime loading is local-only, with remote code disabled and no model downloads.
- Local serving is float32 CPU, with four PyTorch threads, not 4-bit NF4 CUDA. The training method remains QLoRA; its Colab test scores are not revalidated Mac scores.
- Six length-normalized candidate completion likelihoods use the original experiment protocol. Existing development temperature 0.5, acceptance threshold 0.4 and margin 0.0 are retained; they are not tuned on the challenge test again.
- Explicit domain input bypasses model inference. Recognized security policy signals can override Qwen; generic/definitional guards still clarify. An accepted `other` label or low confidence asks for clarification.
- Unavailable files/processes or over-budget input use the existing policy/MiniLM fallback and expose `classifier_fallback_used` plus `primary_prediction`. Do not describe fallback requests as Qwen-classified.
- The 512-token classifier budget is separate from the incident model's context budget. Oversize requests are not silently truncated.

The classifier is not Ollama's pretrained `qwen3:1.7b` investigation generator. A classifier adapter cannot be applied to that different-size model.

## Setup

The model files are local, ignored by Git, and require explicit preparation. No startup download occurs. Use the existing training environment with `peft==0.17.1`, `transformers==4.57.6`, `torch==2.8.0` and `accelerate==1.10.1`; no bitsandbytes/CUDA install is needed for CPU inference.

```sh
.venv-training/bin/python prepare_qwen_classifier.py --bundle /absolute/path/qlora_v2_results.zip --download-base
.venv-api/bin/python run_local.py
```

The optional setup download uses only the official `Qwen/Qwen3-0.6B` repository at revision `c1899de289a04d12100db370d81485cdf75e47ca`, with a safetensors/JSON/tokenizer allowlist. No credentials, paid API or remote Python code are needed. The existing adapter bundle supplies its adapter and development settings.

Override `QWEN_CLASSIFIER_DIR` or `QWEN_CLASSIFIER_PYTHON` when using different local locations. `DOMAIN_CLASSIFIER=minilm` restores compatibility behavior. The router and local launcher otherwise default to Qwen.

## Verification

`test_qwen_classifier.py` covers primary routing, security overrides, explicit routes, abstention, `other`, visible fallback, intake reuse and no automatic downloads. Contract tests do not establish model quality. A live local smoke test must show `routing_mode=qwen_qlora_primary`, the v2 adapter hash, and `classifier_fallback_used=false`; report its timing separately from Colab results.

Two real local smoke requests passed through the adapter and parent LangGraph with no fallback: a VPN/password support question selected IT and LlamaIndex support; an HTTP 503 checkout incident selected production and the investigation branch. End-to-end times were 16.106 seconds for the first request including cold model/support loading, then 2.561 seconds for the second request with the classifier resident. Incident synthesis was deliberately offline in this test. These are two functional checks, not latency percentiles, a full Ollama benchmark or a new accuracy evaluation. Adapter SHA-256: `b8713c0e472a2a12319df83eedf50e11f30a2cf520a6f67e1604aed5ad24a77b`.
