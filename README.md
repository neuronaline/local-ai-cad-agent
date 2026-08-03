# 🏗️ Local AI CAD Agent

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)

A local-first web app that lets you **chat with an AI agent to create parametric CAD models**. Built with [build123d](https://github.com/gumyr/build123d) for solid modeling, [Three.js](https://threejs.org/) for in-browser preview, and [OpenRouter](https://openrouter.ai/) for LLM access — all sandboxed with Bubblewrap.

<p align="center">
  <em>(screenshot coming soon)</em>
</p>

## ✨ Features

- **Chat-driven modeling** — Describe what you want in natural language; the agent writes and runs build123d Python code
- **Live reasoning** — Watch the model's thinking stream in real-time while it works
- **STL preview** — Rotate, pan, and inspect generated models directly in the browser
- **Reference images** — Upload up to 5 images (10 MB each) to guide the agent
- **Sandboxed execution** — All generated code runs in a Bubblewrap container with blocked network and resource limits
- **Finalize** — Export validated models as STEP, STL, and a journey report
- **Project management** — Create, rename, and switch between multiple CAD projects with persisted conversation history
- **Dark theme UI** — Compact, responsive interface with Markdown rendering and syntax highlighting

## 📋 Requirements

- **Python** 3.10+
- **Linux** with `bubblewrap` and `libseccomp2`
- **OpenRouter API key** ([get one here](https://openrouter.ai/keys))

## 🚀 Quick Start

```bash
# Clone and install dependencies
git clone https://github.com/neuronaline/local-ai-cad-agent.git
cd local-ai-cad-agent
./install.sh

# Add your OPENROUTER_API_KEY to .env (or use the setup page on first launch)

# Run
./run.sh
```

Open `http://127.0.0.1:5000` (or whatever `server.host`/`server.port` you configured).

## 📁 Project Structure

```
local-ai-cad-agent/
├── app.py                     # Flask HTTP/SSE server
├── agent/
│   ├── core.py                # AgentRunner tool-calling loop
│   ├── openrouter.py          # Streaming chat-completions client
│   ├── finalize.py            # Model validation & atomic export
│   ├── sandbox.py             # Bubblewrap workspace isolation
│   ├── settings.py            # Config merging & Settings dataclass
│   ├── prompt.py              # System prompt with build123d playbook
│   ├── images.py              # Reference image normalization
│   └── tools/                 # Agent tools (file, terminal, cad, question, experience)
├── static/
│   ├── js/app.js              # SSE client, chat & UI logic
│   ├── js/viewer.js           # Three.js CadViewer
│   └── css/style.css          # Dark theme
│   └── vendor/                # Pinned, local frontend dependencies
├── templates/
│   ├── index.html             # Chat + 3D viewer page
│   └── projects.html          # Project management page
├── tests/                     # 111 tests across 10 files
├── run.sh                     # Startup script
├── gunicorn.conf.py           # Single-worker WSGI config
├── config.example.yaml        # Shareable defaults
└── requirements.txt
```

## ⚙️ Configuration

Copy `config.example.yaml` to `config.yaml` (git-ignored). Personal overrides go in `~/.cad-agent/config.yaml` — the two files merge, personal settings win.

| Key | Default | Description |
|---|---|---|
| `workspace_root` | `~/CAD-Agent-Projects` | Where project data lives |
| `agent.tool_call_limit` | `12` | Max tool rounds per task |
| `openrouter.model` | `openai/gpt-4o-mini` | OpenRouter model slug |
| `openrouter.reasoning_effort` | `""` | `automatic`, `minimal`, `low`, `medium`, or `high` |
| `openrouter.provider` | `""` | Preferred provider slug (e.g., `openai`, `deepinfra/turbo`) |
| `openrouter.force_provider` | `false` | Disable fallbacks (strict `provider.only`) |
| `server.host` / `server.port` | `127.0.0.1` / `5000` | Bind address |
| `ui.show_info_messages` | `true` | Show tool-status info messages in chat |

## 🧪 Development

```bash
.venv/bin/pip install -r requirements-dev.txt

# Run tests
.venv/bin/python -m pytest -q

# Lint
.venv/bin/python -m ruff check .

# Verify dependencies
.venv/bin/python -m pip check
```

## Frontend dependencies

Three.js (including the STL loader and orbit controls), Marked, Highlight.js, and the Highlight.js theme are committed under `static/vendor/` at explicitly pinned versions. The UI does not depend on a CDN, so it remains usable offline and is not affected by third-party CDN availability.

To update a frontend library, choose an exact upstream release, download its browser distribution and license from the project's official release or repository, and replace only that library's directory in `static/vendor/`. Preserve upstream notices, update the version and source URL in [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md), verify the import paths in `templates/index.html`, then run `pytest -q`.

## 🔒 Security

- `.env` and `config.yaml` are **git-ignored** — never commit API keys
- All generated code runs in a **Bubblewrap sandbox** with clean environment, blocked network syscalls, and `prlimit` resource caps
- The app is **local-only** — do not expose it to the public internet
- Sandbox assumes standard FHS layout (`/usr`, `/etc`, `/bin`, `/lib`); non-standard distros (NixOS, Guix, Fedora Silverblue) may need adjustments

## 📄 License

[MIT](LICENSE)
