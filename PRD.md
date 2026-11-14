# Product Requirements Document (PRD)

## Project: cluster Multi-Notebook Distributed GPU Mesh

### 1. Vision and Objective

The objective of cluster Distributed Mesh is to pool multiple ephemeral cloud notebook runtimes (Google Colab, Kaggle, and team member GPU instances) into an orchestrated, unified multi-GPU inference cluster.

Key targets:
* Unified Multi-GPU Model Deployment: A single model (e.g. Qwen 2.5 14B/32B/72B, DeepSeek, or GLM) is downloaded once and cooperatively deployed across all 4+ GPUs simultaneously.
* VRAM Pooling: Combines the VRAM of all participating notebooks (e.g. 4x 16GB T4 = 64GB pooled VRAM, or 4x 2xT4 = 128GB pooled VRAM) to run massive models that cannot fit on any single notebook.
* Collaborative Token Generation: Powered by `llama.cpp RPC`, all GPUs compute their assigned transformer layers collaboratively, streaming tokens to client applications.
* Interactive Multi-Provider Model Resolver: `cluster fetch <model>` searches across multiple registries (Ollama, Hugging Face, GGUF Hubs) and presents an interactive selection list before deploying.
* Single-Command Notebook Orchestration: A Leader Notebook generates dedicated one-liner commands for all worker notebooks, admin control, and team members.

### 2. Leader Notebook Orchestration Workflow

The entire cluster lifecycle is initialized and coordinated from a Primary Leader Notebook.

#### 2.1 Initialization Flow
1. The cluster lead opens the Leader Notebook on Google Colab or Kaggle.
2. The user runs the bootstrap cell. The script prompts for cluster dimensions:
   "How many worker notebooks will participate in this cluster? [Default: 3]"
3. The Leader Notebook establishes its Cloudflare Tunnel and displays a clean control board with generated command links:
   * Worker Join Links (notebook-1, notebook-2, ... notebook-N):
     One-line commands to copy and paste into secondary worker notebooks.
   * Admin / Leader Link:
     Command for the team lead laptop to deploy models and manage the cluster:
     `cluster admin connect <leader-tunnel-url> --token <admin-token>`
   * Team Member Link:
     Command shared with team members for inference access:
     `cluster connect <leader-tunnel-url> --token <member-token>`

#### 2.2 Cooperative Multi-GPU Deployment (llama.cpp RPC)
* Engine: Pre-compiled `llama.cpp` CUDA 12 binaries (`llama-server` and `rpc-server`).
* Worker Execution: Each worker notebook runs `rpc-server -p 50052 -H 0.0.0.0`.
* Leader Orchestration: Leader runs `llama-server -m <model.gguf> --rpc <worker1>:50052,<worker2>:50052... -ngl 999`.
* Layer Sharding: For a 48-layer transformer model running on 4 notebooks:
  * Notebook 1 (Leader): Embedding layer + Layers 1 to 12
  * Notebook 2 (Worker 1): Layers 13 to 24
  * Notebook 3 (Worker 2): Layers 25 to 36
  * Notebook 4 (Worker 3): Layers 37 to 48 + LM Head / Token Generation
* Pooled Memory: 4x Tesla T4 instances provide 64GB of pooled VRAM, allowing 32B, 70B, and GLM models to be loaded with full context.

#### 2.3 Interactive Multi-Provider Model Fetcher (`cluster fetch <model>`)
* Searching across registries:
  * Ollama Library
  * Hugging Face Hub (verified GGUF publishers such as bartowski, Qwen, DeepSeek, TheBloke)
* Interactive Terminal Picker:
  * Displays model name, provider, quantization level (Q4_K_M, Q5_K_M, Q8_0), file size, and VRAM fit indicator.
  * Allows user to select by index.
* High-Speed Multi-Threaded Download:
  * Leader downloads the chosen GGUF at Colab/Kaggle datacenter speeds (300MB/s - 1GB/s) using parallel chunk streaming.
  * Once downloaded, Leader immediately distributes layer shards across the worker RPC pool.

#### 2.4 Role-Based Access Control (RBAC)
* Leader / Admin Role:
  * Full cluster management.
  * Runs `cluster fetch` to search, select, and deploy models across the multi-GPU mesh.
  * Views live GPU, VRAM, temperature, and latency metrics across every node (`cluster gpu`).
* Team Member Role:
  * Inference access only (`cluster chat`, `cluster proxy`).
  * Connects IDEs, Cursor, Continue, or Open WebUI directly to the unified cluster endpoint.
  * Fair-share queuing and request multiplexing.
* Worker Node Role:
  * Operates as a remote RPC GPU accelerator node.
  * Receives layer weights and processes forward-pass tensor activations.

### 3. Architecture and Core Topology

The system comprises three primary layers:

#### A. Leader Node (Coordinator & Primary GPU)
* Ingress Gateway: Exposes public Cloudflare Tunnel serving both Admin management APIs and Member inference APIs.
* Worker Registry: Tracks connected worker notebooks (`notebook-1`, `notebook-2`, etc.), active GPU hardware, VRAM headroom, and round-trip ping latency.
* Multi-GPU Inference Engine:
  * Orchestrates distributed layer placement across all connected RPC GPU nodes.
  * Manages activation streaming and token decoding loop.
* Fault Tolerance: Actively monitors worker health. If a worker goes offline, alerts the Admin to reconnect the node.

#### B. Worker Nodes (Remote GPU Compute Shards)
* Run on individual Colab or Kaggle instances.
* Paste one-liner from the Leader board to join (`curl ... | bash -s -- --join <leader-url> --token <worker-token> --name notebook-X`).
* Starts an optimized GPU RPC compute backend and connects back to the Leader.
* Executes GPU matrix multiplications and attention operations for its assigned transformer layers.

#### C. Local Client & Developer Integration
* Team members run `cluster connect <leader-url> --token <member-token>` on their laptops.
* Local proxy binds to `127.0.0.1:11434`.
* Any application (Cursor, VS Code Continue, Open WebUI, Chatbox, or Python SDK) interacts with the cluster as if it were a single ultra-fast local Ollama instance.

### 4. Functional Requirements

#### 4.1 Automated Worker Pairing
* Worker commands generated by the Leader are pre-authenticated with unique node pairing tokens.
* When a worker notebook executes its command, the Leader displays real-time connection status on its dashboard.

#### 4.2 Multi-Provider Search & Selection
* `cluster fetch <query>` searches Ollama and Hugging Face registries, parses available GGUFs, and formats an interactive selection list.
* Selected model is fetched directly to the Leader and loaded across all worker GPUs.

#### 4.3 Streaming Token Delivery
* Generated tokens are streamed via unbuffered Server-Sent Events (SSE) directly to the end user with low first-byte latency.

### 5. Non-Functional Requirements

* Memory Scaling: Linear VRAM pooling across 2 to 16+ GPU nodes.
* Zero Firewall Config: Pure outbound HTTPS / Cloudflare tunnel connectivity on all nodes.
* Resilience: Automatic keepalive pings preventing Colab/Kaggle idle timeouts.
* Cost: 100% free tier infrastructure compatible (Google Colab free T4, Kaggle free 2x T4).

### 6. Implementation Milestones

* Milestone 1: Leader Coordinator core with interactive node count prompt and dynamic one-liner command board generator.
* Milestone 2: Worker join script launching high-performance llama.cpp RPC backend.
* Milestone 3: Interactive multi-provider model search and download engine (`cluster fetch`).
* Milestone 4: Cooperative multi-GPU layer placement and streaming execution.
* Milestone 5: Admin CLI commands for cluster telemetry and model management.
* Milestone 6: Multi-user fair share request queueing for team inference.
