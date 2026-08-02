# Local AI CAD Agent

A local-first web application for creating and refining parametric CAD models with an OpenRouter-backed coding agent. It combines a Flask API, build123d model generation, an in-browser STL viewer, and controlled project workspaces.

## Features

- Create separate CAD projects and retain their conversation history.
- Ask an AI agent to inspect, edit, and render build123d models.
- Follow model-provided reasoning live while a task is running.
- Upload up to five reference images (10 MB each).
- Preview generated STL files in the browser with Three.js.
- Export a reviewed model as STEP, STL, and a report through the **Finalize** action.
- Keep model workspaces local; AI-generated code cannot directly trigger final exports.

## Requirements

- Python 3.10 or newer
- Linux with `bubblewrap` and `libseccomp` for sandboxed model execution
- An [OpenRouter](https://openrouter.ai/) API key for AI chat requests

## Quick start

```bash
cd local-ai-cad-agent
sudo apt install bubblewrap libseccomp2
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
cp config.example.yaml config.yaml
```

Edit `.env` and add your key:

```dotenv
OPENROUTER_API_KEY=your_key_here
```

Start the application:

```bash
./run.sh
```

Open the address printed by `run.sh` (from `server.host` and `server.port`). Run `./run.sh --help` to display the startup instructions.

## Configuration

Copy `config.example.yaml` to the ignored `config.yaml` file for project-local settings. Personal overrides may also be stored in `~/.cad-agent/config.yaml`. The two files are merged, with personal settings taking precedence. If neither file exists, the application uses the example values shown below.

Example override:

```yaml
workspace_root: ~/CAD-Agent-Projects
agent:
  tool_call_limit: 12 # maximum model/tool rounds per task
openrouter:
  model: openai/gpt-4o-mini
  reasoning_effort: "" # automatic, minimal, low, medium, or high
  provider: ""         # optional slug, for example openai or deepinfra/turbo
  force_provider: false
server:
  port: 5000
ui:
  show_info_messages: true # green task, tool, and usage messages
```

## Model, reasoning, and provider routing

Set the OpenRouter model, reasoning effort, and optional provider in `config.yaml`, then restart the application. These settings apply to every agent task.

Selecting **Force this provider** sends OpenRouter `provider.only`, disables fallbacks, and requires the provider to support every requested parameter. This is intentionally strict: an unavailable provider or an unsupported model/parameter combination will fail instead of silently using another provider. Leave it unchecked to prioritize the provider while retaining OpenRouter fallbacks.

## Development checks

Install the development dependencies first:

```bash
.venv/bin/python -m pip install -r requirements-dev.txt
```

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check .
.venv/bin/python -m pip check
bash -n run.sh
```

## Security and data

The local `.env` and `config.yaml` files are ignored by Git. Never commit API keys or other credentials. Project conversations, inputs, previews, and exports are stored under the configured local workspace. Generated Python runs in a Bubblewrap filesystem sandbox with a clean environment, blocked network syscalls, and resource limits. The sandbox assumes a standard Linux Filesystem Hierarchy Standard layout (`/usr`, `/etc`, `/bin`, `/lib`); non-standard distros (NixOS, Guix, Fedora Silverblue) may require adjustments. The application is designed for local use; do not expose it directly to the public internet.

## License

Released under the [MIT License](LICENSE).
