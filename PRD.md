# Product Requirements Document (PRD)

## Project: cluster Multi-Notebook Distributed GPU Mesh

### 1. Vision and Objective

The objective of cluster Distributed Mesh is to pool multiple ephemeral cloud notebook runtimes (Google Colab, Kaggle, and team member GPU instances) into an orchestrated, unified multi-GPU inference cluster. Instead of a generic browser-based interface, cluster provides a rich, premium, cyber-aesthetic Terminal User Interface (TUI) desktop dashboard powered by Python **Textual**, alongside a streamlined CLI and background transparent proxy.

Key targets:
* **Unified Multi-GPU Model Deployment**: A single model (e.g. Qwen 2.5 14B/32B/72B, DeepSeek, or GLM) is downloaded once and cooperatively deployed across all 4+ GPUs simultaneously.
* **VRAM Pooling**: Combines the VRAM of all participating notebooks (e.g. 4x 16GB T4 = 64GB pooled VRAM, or 4x 2xT4 = 128GB pooled VRAM) to run massive models that cannot fit on any single notebook.
* **Collaborative Token Generation**: Powered by pre-compiled `llama.cpp RPC` CUDA 12 backends, all GPUs compute their assigned transformer layers collaboratively, streaming tokens to client applications with minimal network overhead.
* **Rich Premium TUI Dashboard**: A full-fledged terminal application built with `Textual`, featuring:
  * A live **Hierarchical Card DAG** topology visualizing the 1 Admin manager, 1 Leader coordinator node, 3+ Worker nodes, and concurrent Team Member clients.
  * Live VRAM gauges, token speed meters, and animated token-generation activity.
  * Node **Inspector Drawer** with quick hotkeys (`[c]` copy reconnect one-liner, `[r]` ping RPC, `[e]` evict).
  * 5-Tab workspace: `[1] Mesh Topology`, `[2] Models & Deploy`, `[3] Chat Playground`, `[4] Team & Traffic`, `[5] System Logs`.
* **Dual-Port Runtime Architecture**:
  * Port 2424: Core Hub Engine, REST API, and Server-Sent Events (SSE) telemetry bus.
  * Port 8080: Transparent OpenAI-compatible API Proxy ingress; client tools (Cursor, VS Code, Open WebUI) connect to 8080, which routes through the 2424 Hub for live traffic telemetry and mesh visualization.
* **Default TUI Launch**: Running `cluster ui` boots the 2424 Hub and 8080 Proxy in background daemon threads and launches the full Textual TUI directly in the terminal (with `--web` flag retained for optional browser dashboard fallback).
* **Resilient Hot-Reconnection**: If a worker notebook disconnects due to Colab timeout, prompts are held in a queue while the Leader alerts the Admin with the exact command to re-paste in a fresh cell without restarting the cluster.

---

### 2. Leader Notebook Orchestration Workflow

The entire cluster lifecycle is initialized and coordinated from a Primary Leader Notebook.

#### 2.1 Initialization Flow
1. The cluster lead opens the Leader Notebook on Google Colab or Kaggle.
2. The user runs the bootstrap cell. The script prompts for cluster dimensions:
   `How many worker notebooks will participate in this cluster? [Default: 3]`
3. The Leader Notebook establishes its Cloudflare Tunnel and displays a clean control board with generated command links:
   * **Worker Join Links** (notebook-1, notebook-2, ... notebook-N):
     One-line commands to copy and paste into secondary blank worker notebooks:
     `!curl -fsSL https://raw.githubusercontent.com/berlin-566/cluster/main/remote/worker.sh | bash -s -- --leader <url> --token <token> --name notebook-X`
   * **Admin / Leader Link**:
     Command for the team lead laptop to deploy models and manage the cluster:
     `cluster admin connect <leader-tunnel-url> --token <admin-token>`
   * **Team Member Link**:
     Command shared with team members for inference access:
     `cluster connect <leader-tunnel-url> --token <member-token>`

#### 2.2 Cooperative Multi-GPU Deployment (llama.cpp RPC)
* **Engine**: Pre-compiled `llama.cpp` CUDA 12 binaries (`llama-server` and `rpc-server`). Installs in seconds with zero Python interpreter overhead during inference.
* **Worker Execution**: Each worker notebook runs `rpc-server -p 50052 -H 0.0.0.0`.
* **Leader Orchestration**: Leader runs `llama-server -m <model.gguf> --rpc <worker1>:50052,<worker2>:50052... -ngl 999`.
* **Layer Sharding**: For a 48-layer transformer model running on 4 notebooks:
  * Notebook 1 (Leader): Embedding layer + Layers 1 to 12
  * Notebook 2 (Worker 1): Layers 13 to 24
  * Notebook 3 (Worker 2): Layers 25 to 36
  * Notebook 4 (Worker 3): Layers 37 to 48 + LM Head / Token Generation
* **Pooled Memory**: 4x Tesla T4 instances provide 64GB of pooled VRAM, allowing 32B, 70B, and GLM models to be loaded with full context.

#### 2.3 Interactive Multi-Provider Model Fetcher (`cluster fetch <model>`)
* **Registry Search**: Ollama Library and Hugging Face Hub (verified GGUF publishers such as bartowski, Qwen, DeepSeek, TheBloke).
* **Interactive TUI & Terminal Picker**:
  * Displays model name, provider, quantization level (Q4_K_M, Q5_K_M, Q8_0), file size, and VRAM fit indicator.
  * Allows user to select by arrow keys or hotkeys to deploy directly to the cluster.
* **High-Speed Multi-Threaded Download**:
  * Leader downloads the chosen GGUF at datacenter speeds (300MB/s - 1GB/s) using parallel chunk streaming.
  * Once downloaded, Leader distributes layer shards across the worker RPC pool.

#### 2.4 Role-Based Access Control (RBAC)
* **Leader / Admin Role**:
  * Full cluster management.
  * Runs `cluster fetch` to search, select, and deploy models across the multi-GPU mesh.
  * Views live GPU, VRAM, temperature, and latency metrics across every node (`cluster gpu` or TUI Topology Tab).
  * Worker eviction, reconnect command generator, and RPC health inspector.
* **Team Member Role**:
  * Inference access only (`cluster chat`, `cluster serve`, or TUI Chat Playground).
  * Connects IDEs, Cursor, Continue, or Open WebUI directly to the unified cluster endpoint (`http://127.0.0.1:8080`).
  * Fair-share queuing and request multiplexing.
* **Worker Node Role**:
  * Remote RPC GPU accelerator node processing forward-pass tensor activations.

#### 2.5 Resilient Hot-Reconnection
* **Heartbeat Monitoring**: The Leader monitors worker RPC health every 3 seconds.
* **Disconnect Handling**: If a worker notebook disconnects:
  * Active prompts are paused in an in-memory queue.
  * TUI and CLI display an urgent reconnect alert with the exact copy-pastable one-liner command.
  * Once re-pasted, the worker rejoins, RPC reconnects, and queued generation resumes seamlessly without cluster restart.

---

### 3. Dual-Port Runtime & Rich Textual TUI Dashboard Architecture

The desktop dashboard operates as a native, high-performance, rich Terminal User Interface (TUI) powered by Python **Textual**, replacing browser dependencies with a cyber-aesthetic terminal cockpit.

#### 3.1 Dual-Port Network Topology
* **Port 2424 (Central Hub & SSE Telemetry Engine)**:
  * Background daemon thread running the HTTP/SSE Hub server.
  * Hosts the local `/api/telemetry/stream` (SSE) and `/api/cluster/status` endpoints.
  * Coordinates configuration, proxy metrics, model deployment triggers, and cluster connection state.
* **Port 8080 (Transparent API Proxy Ingress)**:
  * Binds to `http://127.0.0.1:8080`.
  * External applications (Cursor, VS Code Continue, Open WebUI, Ollama CLI) direct their requests to port 8080.
  * Port 8080 forwards requests to the cloud Leader via Cloudflare Tunnel, feeding live token telemetry and request counters to the Hub.

#### 3.2 Design System & Cyber Matte Aesthetic
* **Color Palette (Tokyo Night / Cyber Matte Dark)**:
  * **Background**: Deep Slate Charcoal (`#0d1117` / `#16161e`)
  * **Surface Cards**: Charcoal Zinc (`#1a1b26` / `#24283b`)
  * **Leader Accent**: Electric Cyan (`#00e5ff`)
  * **Worker Health & VRAM**: Emerald Green (`#10b981`)
  * **Admin Persona & Badges**: Vivid Magenta (`#ec4899`)
  * **Alerts & Warnings**: Warm Amber (`#f59e0b`)
  * **Borders**: Rounded Unicode box borders (`╭───╮`, `│   │`, `╰───╯`)
* **Glyphs & Iconography**: Strictly zero emojis. 100% clean Unicode glyphs (`•`, `━`, `▶`, `▲`, `■`, `◆`, `├──`, `└──`).
* **Typography**: Monospace system font with styled bold headers, dim auxiliary notes, and highlighted keybindings.

#### 3.3 Visual Layout and Multi-Tab Navigation
The TUI is structured with a Top Tab Bar, a reactive Main Viewport, and a Persistent Bottom Status Bar.

* **Top Tab Bar**:
  * `[1] Mesh Topology` (F1): Hierarchical Node DAG, VRAM meters, and Node Inspector Drawer.
  * `[2] Models & Deploy` (F2): Multi-provider model browser (HF/Ollama), quantization picker, and 1-click deployer.
  * `[3] Chat Playground` (F3): Interactive streaming chat with markdown formatting and parameter sliders.
  * `[4] Team & Traffic` (F4): Real-time list of connected team members, prompt queue depth, and token scoreboard.
  * `[5] System Logs` (F5): Live diagnostic logs, RPC ping stream, and tunnel connection status.

* **Persistent Bottom Status Bar**:
  * Left: Active Model badge (`[qwen2.5:14b]`) and Cluster Health status (`[HEALTHY]`).
  * Center: Pooled VRAM meter (`[21.8 GB / 64.0 GB]`) and Token Speed gauge (`[48.2 tok/s]`).
  * Right: Ingress endpoint (`Proxy: 127.0.0.1:8080`) and quick hotkey reminders (`[Tab] Next Tab | [?] Help | [q] Quit`).

#### 3.4 Hierarchical Card DAG Topology (Tab 1)
The primary view features a structured columnar DAG visualizing the cluster hierarchy:
```
╭───────────────────────╮
│  [ADMIN] Team Lead    │
│  Role: Cluster Master │
╰───────────┬───────────╯
            │
            ▼
╭───────────────────────────────────────────────╮
│  [LEADER] Colab Master Node                   │
│  GPU: Tesla T4 (16GB)  •  Temp: 48°C          │
│  VRAM: [████████░░░░░░░░] 6.4 / 16.0 GB       │
│  Layers: 0..12 (Embeddings)  •  Ping: 0ms     │
╰───────┬───────────────────────┬───────────────╯
        │                       │
        ├───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
╭─────────────────────╮ ╭─────────────────────╮ ╭─────────────────────╮
│ [WORKER 1] Colab    │ │ [WORKER 2] Kaggle   │ │ [WORKER 3] Colab    │
│ GPU: Tesla T4       │ │ GPU: Tesla T4       │ │ GPU: Tesla T4       │
│ VRAM: [██████░░░░░] │ │ VRAM: [██████░░░░░] │ │ VRAM: [██████░░░░░] │
│ 5.2 / 16.0 GB       │ │ 5.1 / 16.0 GB       │ │ 5.1 / 16.0 GB       │
│ Layers: 13..24      │ │ Layers: 25..36      │ │ Layers: 37..48 (LM) │
│ Ping: 12ms          │ │ Ping: 18ms          │ │ Ping: 14ms          │
╰─────────────────────╯ ╰─────────────────────╯ ╰─────────────────────╯
        ▲                       ▲                       ▲
        ╰───────────────────────┴───────────────────────╯
                                │
                    ╭───────────┴───────────╮
                    │ [MEMBERS] 15 Active   │
                    │ Ingress: :8080 Proxy  │
                    ╰───────────────────────╯
```

* **Interactive Node Selection & Inspector Drawer**:
  * Use Arrow Keys (`←`, `→`, `↑`, `↓`) or Mouse Clicks to focus on any node card.
  * Pressing `Enter` or clicking opens the **Inspector Drawer** on the right/bottom:
    * Detailed hardware metrics: GPU Model, VRAM Used/Free, CUDA utilization %, temperature, fan speed.
    * Assigned transformer layer indices and weight sharding split.
    * RPC socket connection latency and heartbeat history.
    * Hotkey Actions:
      * `[c]`: Copy reconnect one-liner command to clipboard.
      * `[r]`: Test RPC latency with instant ping packet.
      * `[e]`: Evict or pause node in cluster pool.

#### 3.5 Real-Time Telemetry & SSE Streaming Integration
* The Textual TUI runs an asynchronous background worker that subscribes to `http://127.0.0.1:2424/api/telemetry/stream` (SSE).
* Sub-second token pulses animate live activity indicators in the Leader and Worker cards during token generation.
* Periodic 2-second REST polling to `http://127.0.0.1:2424/api/cluster/status` keeps node health, VRAM, and ping states up-to-date.
* Automatic self-boot: If the Hub (port 2424) or Proxy (port 8080) is not already active, the TUI automatically spawns them in background daemon threads on startup.

---

### 4. CLI Reference

#### Core Commands
* `cluster ui [--port 2424] [--proxy-port 8080] [--web]`:
  * Default: Boots Hub & Proxy in background and launches the full Textual TUI dashboard.
  * With `--web`: Launches background Hub and opens the legacy browser dashboard at `http://localhost:2424`.
* `cluster connect <url> --token <token>`: Authenticate client as team member or admin.
* `cluster admin connect <url> --token <token>`: Authenticate as cluster administrator.
* `cluster chat [model]`: Quick terminal chat streaming directly from command line.
* `cluster serve [--port 8080]`: Standalone API server and Ollama bridge for Cursor, VS Code, Open WebUI, and native Ollama CLI (alias: `cluster proxy`).
* `cluster fetch <query>`: Interactive CLI model search across registries.
* `cluster gpu`: Quick terminal summary of pooled VRAM and remote GPU telemetry.
* `cluster cluster status`: Terminal summary of cluster connectivity and node health.
* `cluster cluster workers`: List worker node states and reconnect commands.
* `cluster version`: Version check and self-updater.

---

### 5. Non-Functional Requirements

* **Runtime Footprint**: Ultra-lightweight terminal runtime; zero Electron/Chromium overhead, runs comfortably under 30MB RAM within the user's existing terminal emulator (Ghostty, Alacritty, Kitty, WezTerm, iTerm2, tmux).
* **Dependencies**: Native Python dependencies (`textual`, `rich`, `httpx`, `click`) integrated cleanly in `pyproject.toml`.
* **Zero Firewall Config**: Outbound Cloudflare tunnel connectivity for all remote nodes.
* **Aesthetic Standard**: Zero emojis, clean Unicode box-drawing characters, cyber-matte palette.
* **Resilience**: Auto-reconnect handling with held prompt queues and immediate notification alerts.

---

### 6. Implementation Milestones

* **Milestone 1**: Leader Coordinator core with interactive node count prompt and dynamic one-liner command board generator.
* **Milestone 2**: Worker join script launching high-performance llama.cpp RPC backend.
* **Milestone 3**: Interactive multi-provider model search and download engine (`cluster fetch`).
* **Milestone 4**: Dual-port runtime architecture (Port 2424 Hub Engine + Port 8080 Proxy Bridge).
* **Milestone 5**: Textual TUI Framework & Layout:
  * 5-Tab Navigation structure (`[1] Mesh Topology`, `[2] Models`, `[3] Chat`, `[4] Team`, `[5] Logs`).
  * Cyber Matte Dark theme and persistent status bar.
* **Milestone 6**: Hierarchical Card DAG & Inspector:
  * Custom Textual widget rendering 1 Admin -> 1 Leader -> 3 Workers -> Team Members.
  * Node selection, VRAM progress meters, and Inspector Drawer (`[c]` reconnect copy, `[r]` ping, `[e]` evict).
* **Milestone 7**: TUI Interactive Tabs:
  * Models & Deploy tab with live search and 1-click deployment.
  * Chat Playground tab with streaming tokens and markdown formatting.
  * Team & Traffic tab with active client monitoring.
* **Milestone 8**: Background Hub & SSE Integration:
  * Seamless auto-launch of Hub (2424) and Proxy (8080) from `cluster ui`.
  * Real-time SSE telemetry consumer driving TUI animations.
