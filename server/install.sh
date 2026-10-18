#!/usr/bin/env bash
# Installs the Mira Voice PC server for this user: a Python environment with the models' libraries, the server
# itself, and a systemd user service that starts it at login. Everything goes under ~/.local/share/mira-voice-server,
# so this repo can move; re-run after updating mira_server.py. Needs uv and an NVIDIA GPU (CUDA 13 driver).
# Remove: systemctl --user disable --now mira-voice-server; rm ~/.config/systemd/user/mira-voice-server.service
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
home=$HOME/.local/share/mira-voice-server
venv=$home/venv
command -v uv >/dev/null || { echo "install.sh needs uv: https://docs.astral.sh/uv/" >&2; exit 1; }

[ -x "$venv/bin/python" ] || uv venv -q -p 3.12 "$venv"
# PyTorch with CUDA from its own index; torch and torchaudio minor versions must match
VIRTUAL_ENV=$venv uv pip install -q "torch==2.11.*" "torchaudio==2.11.*" --index-url https://download.pytorch.org/whl/cu130
VIRTUAL_ENV=$venv uv pip install -q -r "$here/requirements.txt"
install -m 644 "$here/mira_server.py" "$home/mira_server.py"

unit=$HOME/.config/systemd/user/mira-voice-server.service
mkdir -p "$(dirname "$unit")" "$HOME/.cache/mira-voice-server/inductor"
cat > "$unit" <<EOF
[Unit]
Description=Mira Voice PC server (the phone app's "On my PC" mode)
After=network-online.target

[Service]
ExecStart=$venv/bin/python $home/mira_server.py
Environment=PYTHONUNBUFFERED=1
# compiled GPU kernels survive reboots (the very first model load compiles them for about 3 minutes)
Environment=TORCHINDUCTOR_CACHE_DIR=%h/.cache/mira-voice-server/inductor
Restart=on-failure
RestartSec=10
# a background service: low priority, and capped so a runaway can't take the desktop down
Nice=10
CPUWeight=20
IOWeight=20
MemoryMax=24G
MemorySwapMax=0

[Install]
WantedBy=default.target
EOF
systemctl --user daemon-reload
systemctl --user enable mira-voice-server.service
systemctl --user restart mira-voice-server.service
echo "Mira Voice server installed and started: systemctl --user status mira-voice-server"
