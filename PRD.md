# Product Requirements Document (PRD)

## Project: cluster Multi-Notebook Distributed GPU Mesh

### 1. Vision and Objective

The objective of cluster Distributed Mesh is to pool multiple ephemeral cloud notebook runtimes (Google Colab, Kaggle, and user-provided cloud GPU environments) into a unified, high-throughput, and fault-tolerant inference cluster.

Key targets:
* Scale aggregate token generation throughput linearly (for example, aggregating 3-4 nodes yielding 150 tokens/sec each to deliver 450+ tokens/sec across concurrent streams).
* Enable team-wide shared access across 15-16 member accounts with transparent load balancing, rate limiting, and unified API keys.
* Support deployment of large foundation models (such as GLM, DeepSeek-R1, and 70B+ parameter models) across multiple nodes using distributed sharding or pipelining.
* Provide dynamic elasticity and self-healing when ephemeral cloud notebooks restart, sleep, or exhaust runtime allocations.

### 2. Architecture and Core Topology

The system comprises three primary planes:

#### A. Coordinator and Gateway Plane (Control Plane)
* Ingress API: Serves standard Ollama and OpenAI-compatible endpoints (`/v1/chat/completions`, `/api/generate`, `/api/chat`) for local developer tools, IDE extensions (Cursor, VS Code), and team web UIs.
* Node Registry: Tracks registered notebook worker instances, their active GPU hardware (Tesla T4, 2x T4, A100, L4), memory headroom, active model weights, and round-trip ping latency.
* Smart Scheduler / Router: Routes incoming prompts and token streams according to scheduling policies:
  * Least-connections / Queue-depth aware routing
  * Model-affinity routing
  * Speculative or sharded pipeline streaming
* Health and Keepalive Monitor: Actively probes worker heartbeats and automatically evicts dead nodes without dropping in-flight user prompts (prompt replay on backup node).

#### B. Worker Node Plane (Compute Plane)
* Deployed inside Google Colab or Kaggle instances.
* Runs lightweight supervisor daemon connected via secure Cloudflare or Tailscale mesh tunnels.
* Worker Execution Engine:
  * Mode 1 (Replicated / High Throughput): Each node runs an independent instance of a model; requests are load-balanced to achieve aggregate token output (3x 150 t/s = 450 t/s total throughput).
  * Mode 2 (Model Sharding / Pipeline Parallelism): Large models (e.g. 70B+ or large GLM models) are partitioned across 2-4 nodes via pipeline parallelism (similar to Petals or Exo architecture), allowing models that exceed single-GPU VRAM to load and execute across nodes.

#### C. Multi-Tenant and Team Layer
* Team Authentication: User/team tokens and API keys.
* Fair Share Scheduling: Prevents a single team member from starving cluster capacity during batch generation.
* Usage and VRAM Telemetry: Live dashboard displaying cluster health, aggregate tokens per second, and active nodes.

### 3. Functional Requirements

#### 3.1 Node Registration and Discovery
* Worker nodes register with the coordinator using a single command or notebook execution cell.
* Registration payload contains GPU specifications, total VRAM, installed model tags, and secure tunnel URL.
* Node authentication token ensures only authorized workers join the cluster.

#### 3.2 Request Routing and Streaming
* Transparent proxying of streaming Server-Sent Events (SSE) and unbuffered chunked HTTP transfers.
* Automatic retry and failover: If a worker node disconnects during generation, the coordinator fails over to an available node.

#### 3.3 Model Synchronization
* Coordinator can dispatch cluster-wide model pull commands (`cluster cluster models pull <name>`).
* Workers download models in parallel from Ollama registry or HuggingFace caches.

#### 3.4 Multi-User Quota and Concurrency
* Support for concurrent team requests with priority queuing.
* Graceful queuing when all nodes are at maximum compute saturation.

### 4. Non-Functional Requirements

* Latency Overhead: Coordinator routing overhead must be under 5 milliseconds per request.
* Zero Open Ports: Workers communicate via outbound secure tunnels, requiring no static public IP or port forwarding on worker nodes.
* Resiliency: Ephemeral notebook drops must not crash the local proxy or interrupt other active worker streams.
* Security: End-to-end TLS encryption with mutual authentication tokens between coordinator and workers.
* No External Cloud Costs: Entire mesh operates using free cloud runtimes (Colab, Kaggle) and free tier tunnel services.

### 5. Implementation Roadmap

* Phase 1: Multi-node Load Balancer & Health Orchestration (Replicated Mode for 450+ t/s throughput across 3-4 nodes).
* Phase 2: Multi-tenant Team Gateway with API key management and fair-share queue.
* Phase 3: Distributed Model Sharding (Pipeline Parallelism across WAN) for massive models exceeding single-node VRAM.
* Phase 4: Cluster Management TUI and Web Telemetry Dashboard.
