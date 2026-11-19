# Background Job Flow

The [parent LangGraph](PARENT_WORKFLOW.md) now owns request-level orchestration, including unified intake, LlamaIndex support, incident subgraphs and bounded Qwen-requested documentation retrieval. API helpers provide services to graph nodes rather than manually sequencing the workflows.

The [complete flowcharts](END_TO_END_FLOW.md) describe the updated local Qwen path. The older bitmap/vector diagrams below do not include the parser and live local-model node.

Authenticated browser -> POST /jobs -> durable SQLite queue -> HTTP 202/job ID -> independent worker -> existing support/investigation workflows -> atomic case + event + notification -> authenticated polling.

See [BACKGROUND_JOBS.md](BACKGROUND_JOBS.md) for endpoints, lease recovery, retry limits, and delivery guarantees. The diagram below predates this queue and illustrates the underlying workflow engines.

# Combined prototype architecture

![Current local architecture](architecture.svg)

## Request lifecycle

1. The browser obtains a JWT from the local login endpoint. FastAPI checks the token before executing a case.
2. The request carries the question, severity, domain and optional evidence. The JSON API also accepts the original source, title, description, tags, identity fields, service metadata and timestamp.
3. A deterministic policy with a trained fallback identifies active domain signals, accounts for common negation/background clauses, prioritizes security signals and uses an explicitly selected domain when provided. Ambiguous or unknown requests ask for clarification. Scores are weights, not calibrated probabilities.
4. The unified intake stage calls the existing domain policy and, for IT/support context, the resident worker's classification-only operation. It returns the routing decision, category, priority, team and retrieval hints together. LlamaIndex then reuses those ticket fields for retrieval and drafting without another classification pass. Standalone support calls also compute ticket classification once. This is one intake entry point around existing models, not a newly trained joint model.
5. Incidents run a LangGraph graph for security, production, cloud, data or IT. An IT incident can execute both engines in sequence. Both graphs first parse supplied logs/events and correlate request IDs. The launcher enables local Qwen analysis after evidence collection, including low-risk cases; imported API code defaults to offline mode unless configured. Qwen receives only bounded supplied evidence and runbooks, not simulated tool records. Local specialist/tool simulations remain separately labeled. Schema, completion and evidence-ID checks gate the advisory model response; they do not verify factual correctness. No source is asserted to be independently verified.
6. Before draft composition, `route_verification.verify_route()` checks recognized event codes from submitted telemetry. Retrieved guides, generated findings, and simulated evidence cannot confirm the domain. A single different domain can trigger one alternate investigation on automatic routes; explicit routes, conflicting signals, or a remaining disagreement require review. No recognized signal is reported as unverified. The result preserves the initial route and investigation when it reroutes.
7. The response composer attaches the final investigation findings to the support draft. It preserves IT specialist ownership and support citations; investigation citations are stored separately. Security cases recommend SOC and retain the original support team as context. Priority cannot fall below the existing support priority or an explicitly supplied severity.
8. The platform stores the result with the user's identity in SQLite. For a queued job, the browser polls and displays the answer, investigation, suggested team and route verification status. No team receives the result automatically. All drafts require review; external actions are separate opt-in operations.

## Where RAG and ML live

The support lane contains the trainable PyTorch category/priority classifier, with its existing fallback rules and Logistic Regression support. Its RAG path uses SentenceTransformer embeddings and ChromaDB when available, BM25 search, metadata/intents, reranking and source-backed answer generation.

The incident lane retrieves runbooks and playbooks for the LangGraph investigation. This demo uses local lexical retrieval and deterministic reasoning. The copied engine has optional PostgreSQL/pgvector and LLM provider paths, but those are not enabled by this demo.

The top-level domain router combines explicit policy with a character-TF-IDF Logistic Regression fallback trained for five domains plus other. Low confidence or a small score margin triggers clarification. The separate PyTorch classifier predicts IT category and priority. See MODEL_CARD.md for the small synthetic dataset and evaluation limits.

## Process and deployment boundaries

The API or independent job worker invokes the parent LangGraph. One long-running subprocess uses the support virtual environment for classification, LlamaIndex support and documentation retrieval. Calls are serialized with a bounded wait; a timed-out or exited worker is stopped and restarted on the next call. Shutdown closes the worker. Qwen runs through local-only Ollama; a documentation request can make one round trip through the LlamaIndex subprocess before reassessment.

Docker would package these processes and dependencies. Kubernetes would run, restart and scale containers with persistent volumes. Neither is a stage through which a question passes, and neither is running this local demo. The existing deployment files have not been consolidated for this combined application.

## Current boundaries

- Demo login is a local account, not company SSO or a production identity service.
- Source documents and retrieval stores remain separate. Shared uploads/indexing are not exposed through the new interface.
- External connectors, postmortems and monitoring remain in copied source, not fully integrated API features.
- The worker serializes support inference; horizontal scaling and sustained load testing remain future work.
- Local simulated evidence is not proof of a real incident. True groundedness and faithfulness require a separate semantic evaluation.

No merge or push is required to try this local prototype.
