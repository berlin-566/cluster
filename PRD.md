# Product Requirements Document (PRD)

## Project: cluster Multi-Notebook Distributed GPU Mesh

### 1. Vision and Objective

The objective of cluster Distributed Mesh is to pool multiple ephemeral cloud notebook runtimes (Google Colab, Kaggle, and team member GPU instances) into an orchestrated, unified multi-GPU inference cluster, accessible via both a streamlined CLI and a sleek, modern desktop browser application.

Key targets:
* Unified Multi-GPU Model Deployment: A single model (e.g. Qwen 2.5 14B/32B/72B, DeepSeek, or GLM) is downloaded once and cooperatively deployed across all 4+ GPUs simultaneously.
* VRAM Pooling: Combines the VRAM of all participating notebooks (e.g. 4x 16GB T4 = 64GB pooled VRAM, or 4x 2xT4 = 128GB pooled VRAM) to run massive models that cannot fit on any single notebook.
* Collaborative Token Generation: Powered by pre-compiled `llama.cpp RPC` CUDA 12 backends, all GPUs compute their assigned transformer layers collaboratively, streaming tokens to client applications with minimal network overhead.
* Interactive Multi-Provider Model Resolver: `cluster fetch <model>` searches across multiple registries (Ollama, Hugging Face Hub, GGUF publishers) and presents an interactive selection list before deploying.
* Dual-Port Architecture:
  * Port 2424: Core Hub Engine and Cortex.io-style Matte-Black Web Dashboard.
  * Port 8080: Transparent API Proxy ingress; all client tools (Cursor, VS Code, Open WebUI) connect to 8080, which routes through the 2424 Hub for live traffic telemetry and mesh visualization.
* Single-Command Notebook Orchestration: A Leader Notebook generates dedicated one-liner commands for all worker notebooks, admin control, and team members.
* Resilient Hot-Reconnection: If a worker notebook disconnects due to Colab timeout, prompts are held in a queue while the Leader alerts the Admin with the exact command to re-paste in a fresh cell without restarting the cluster.

### 2. Leader Notebook Orchestration Workflow

The entire cluster lifecycle is initialized and coordinated from a Primary Leader Notebook.

#### 2.1 Initialization Flow
1. The cluster lead opens the Leader Notebook on Google Colab or Kaggle.
2. The user runs the bootstrap cell. The script prompts for cluster dimensions:
   "How many worker notebooks will participate in this cluster? [Default: 3]"
3. The Leader Notebook establishes its Cloudflare Tunnel and displays a clean control board with generated command links:
   * Worker Join Links (notebook-1, notebook-2, ... notebook-N):
     One-line commands to copy and paste into secondary blank worker notebooks:
     `!curl -fsSL https://raw.githubusercontent.com/berlin-566/cluster/main/remote/worker.sh | bash -s -- --leader <url> --token <token> --name notebook-X`
   * Admin / Leader Link:
     Command for the team lead laptop to deploy models and manage the cluster:
     `cluster admin connect <leader-tunnel-url> --token <admin-token>`
   * Team Member Link:
     Command shared with team members for inference access:
     `cluster connect <leader-tunnel-url> --token <member-token>`

#### 2.2 Cooperative Multi-GPU Deployment (llama.cpp RPC)
* Engine: Pre-compiled `llama.cpp` CUDA 12 binaries (`llama-server` and `rpc-server`). Installs in seconds with zero Python interpreter overhead during inference.
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
* Interactive Terminal & UI Picker:
  * Displays model name, provider, quantization level (Q4_K_M, Q5_K_M, Q8_0), file size, and VRAM fit indicator.
  * Allows user to select by index or click to deploy.
* High-Speed Multi-Threaded Download:
  * Leader downloads the chosen GGUF at Colab/Kaggle datacenter speeds (300MB/s - 1GB/s) using parallel chunk streaming.
  * Once downloaded, Leader immediately distributes layer shards across the worker RPC pool.

#### 2.4 Role-Based Access Control (RBAC)
* Leader / Admin Role:
  * Full cluster management.
  * Runs `cluster fetch` to search, select, and deploy models across the multi-GPU mesh.
  * Views live GPU, VRAM, temperature, and latency metrics across every node (`cluster gpu`).
  * Views worker connectivity status (`cluster cluster status`, `cluster cluster workers`).
* Team Member Role:
  * Inference access only (`cluster chat`, `cluster proxy`).
  * Connects IDEs, Cursor, Continue, or Open WebUI directly to the unified cluster endpoint.
  * Fair-share queuing and request multiplexing.
* Worker Node Role:
  * Operates as a remote RPC GPU accelerator node.
  * Receives layer weights and processes forward-pass tensor activations.

#### 2.5 Resilient Hot-Reconnection
* Heartbeat Monitoring: The Leader monitors worker RPC health every 3 seconds.
* Disconnect Handling: If a worker notebook disconnects (e.g. Colab timeout or browser close):
  * Active prompts are paused in an in-memory queue.
  * The Leader console and Admin CLI print an urgent reconnect banner:
    `[ALERT] notebook-2 disconnected. Run this one-liner in a fresh cell to restore cluster: <command>`
  * Once re-pasted, the worker rejoins, RPC reconnects, and queued generation resumes seamlessly without cluster restart.

### 3. Dual-Port Runtime & Desktop Dashboard Architecture

The desktop dashboard operates as a lightweight, zero-dependency browser-based command center inspired by Cortex.io.

#### 3.1 Dual-Port Network Topology
* Port 2424 (Central Hub & Web Dashboard):
  * Serves the Cortex.io-style matte-black single page application.
  * Hosts the local WebSocket telemetry bus pushing real-time GPU stats, token speeds, and node connectivity states.
  * Manages configuration, model deployment triggers, and cluster connection logic.
* Port 8080 (Transparent Proxy Ingress):
  * Binds to `http://127.0.0.1:8080`.
  * External applications (Cursor, VS Code Continue, Open WebUI, Ollama CLI) direct their requests to port 8080.
  * Port 8080 forwards all incoming inference requests to the 2424 Hub.
  * The 2424 Hub inspects the active stream, feeds live token pulse telemetry to the dashboard, injects authentication tokens, and proxies to the cloud Leader.

#### 3.2 Design System & Aesthetics
* Color Palette:
  * Dark Mode (Primary): Deep Matte Black (`#09090B`), Charcoal Surface Cards (`#141416`), Subtle Zinc Borders (`#27272A`), Crisp White Typography (`#FAFAFA`), and Muted Slate Accents (`#71717A`).
  * Light Mode: Pure White Canvas (`#FFFFFF`), Soft Gray Cards (`#F4F4F5`), Crisp Charcoal Text (`#09090B`), Slate Borders (`#E4E4E7`).
* Typography: High-legibility sans-serif for UI chrome paired with monospace data fonts for GPU metrics, token counters, and terminal logs.
* Iconography: Strictly zero emojis. 100% clean, hand-crafted vector SVGs for all controls, status pills, and navigation icons.
* Identity & Logo: Minimalist geometric cluster emblem consisting of interconnected vertices on a dark background, representing distributed computing nodes.

#### 3.3 Visual Layout and Hierarchy
* Top Header Bar:
  * Top Left: `cluster` wordmark, geometric SVG emblem, active model badge, and live cluster health pill (`HEALTHY`, `SYNCING`, `DEGRADED`).
  * Center/Right: Live Mesh Topology Graph Widget:
    * Visual node hub displaying the Leader notebook at center.
    * Radiating dynamic links to active Worker notebooks (`notebook-1`, `notebook-2`, etc.) with throughput pulse animations indicating live token generation.
    * Active team member connection count badge.
  * Controls: Dual-mode toggle (Admin Mode vs User Mode) and Theme Switcher (Dark/Light).
* Left Navigation Sidebar:
  * Home / Dashboard: Quick telemetry overview, total cluster tokens/second gauge, aggregate VRAM utilization meter, and 1-click proxy start/stop button.
  * Clusters Overview: Interactive node catalog. Clicking any notebook reveals hardware specs, allocated vs free VRAM, assigned transformer layers, GPU temperature, and round-trip ping.
  * Models & Deployments: Searchable model browser with multi-provider integration (Ollama & HuggingFace), quantization inspector, one-click cluster deploy button, and active model switcher.
  * Team & Traffic: Real-time table of 15-16 team members, active prompt queue depth, token usage statistics, and fair-share concurrency distribution.
  * Settings & Tunnels: Cloudflare tunnel status, custom proxy port configuration, security tokens, and diagnostic log stream.

#### 3.4 Dual-Mode Persona Experience
* Admin Mode:
  * Full cluster management capabilities.
  * Model deployment and sharding orchestration.
  * Worker eviction, reconnect command generator, and RPC health inspector.
  * Raw diagnostic log terminal.
* User / Member Mode:
  * Clean, distraction-free interface for everyday inference.
  * Built-in Interactive Chat Playground supporting streaming tokens, parameter adjustment (temperature, context length), and markdown rendering.
  * Local Proxy Status (`http://127.0.0.1:8080`) with 1-click copy configs for Cursor, VS Code Continue, Open WebUI, and Python scripts.
  * Live generation speed and latency scoreboard.

### 4. CLI Reference

#### Admin Commands
* `cluster admin connect <url> --token <token>`: Authenticate as cluster administrator.
* `cluster fetch <query>`: Interactive multi-provider model search and deployment.
* `cluster gpu`: Live multi-node GPU dashboard with pooled VRAM telemetry.
* `cluster cluster status`: Comprehensive cluster health and latency report.
* `cluster cluster workers`: List worker node states and reconnect commands.
* `cluster ui`: Launch the local Hub and open the browser dashboard on `http://localhost:2424`.

#### Team Member Commands
* `cluster connect <url> --token <token>`: Authenticate as team member.
* `cluster chat [model]`: Interactive terminal chat streaming at multi-GPU speed.
* `cluster proxy [--port 8080]`: Local reverse proxy for Cursor, VS Code, and Open WebUI pointing through the 2424 Hub.
* `cluster status`: Check cluster connectivity and current loaded model.
* `cluster version`: Check installed version and live update status against main branch.

### 5. Non-Functional Requirements

* Desktop Runtime Footprint: Zero bloated runtimes (Electron/Chromium eliminated); served by Python standard library with native browser rendering under 25MB RAM.
* Zero Firewall Config: Pure outbound HTTPS / Cloudflare tunnel connectivity on all nodes.
* No Bloat or AI Slop: Focused, functional controls with clear utility; strictly SVG iconography and zero emojis.
* Resilience: Automatic keepalive pings preventing Colab/Kaggle idle timeouts.
* Cost: 100% free tier infrastructure compatible (Google Colab free T4, Kaggle free 2x T4).

### 6. Implementation Milestones

* Milestone 1: Leader Coordinator core with interactive node count prompt and dynamic one-liner command board generator.
* Milestone 2: Worker join script launching high-performance llama.cpp RPC backend.
* Milestone 3: Interactive multi-provider model search and download engine (`cluster fetch`).
* Milestone 4: Dual-port runtime architecture (Port 2424 Hub Engine + Port 8080 Proxy Bridge).
* Milestone 5: Cortex.io-style Matte-Black Web Dashboard (HTML/CSS/SVG, Live Mesh Graph, Node Inspector).
* Milestone 6: Dual-Mode Persona UI (Admin deployment dashboard + Member chat playground).
* Milestone 7: Telemetry streaming bus and fair-share team queue.
