#!/usr/bin/env bash
# setup_ollama_llava_g4dn.sh
# Install NVIDIA driver (if missing), Ollama, and pull a LLaVA model on EC2 g4dn.xlarge.
# Supports: Ubuntu 20.04/22.04/24.04, Amazon Linux 2023.
set -euo pipefail

MODEL_TAG="${MODEL_TAG:-llava-llama3:latest}"   # override e.g. MODEL_TAG=llava:13b bash setup_ollama_llava_g4dn.sh
AUTO_START="${AUTO_START:-true}"                # start ollama service after install
NEEDS_REBOOT_FLAG="/var/run/needs-nvidia-reboot"

log() { echo -e "\033[1;32m[INFO]\033[0m $*"; }
warn(){ echo -e "\033[1;33m[WARN]\033[0m $*"; }
err() { echo -e "\033[1;31m[ERR]\033[0m  $*"; }

detect_distro() {
  . /etc/os-release
  echo "${ID}:${VERSION_ID:-}"
}

has_nvidia() {
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1
}

install_nvidia_drivers_ubuntu() {
  log "Updating apt and installing Ubuntu NVIDIA drivers..."
  sudo apt-get update -y
  sudo apt-get install -y ubuntu-drivers-common dkms build-essential
  # Autoselect a compatible driver for T4 (R535+ typically fine)
  sudo ubuntu-drivers autoinstall -g || sudo ubuntu-drivers autoinstall
  touch "${NEEDS_REBOOT_FLAG}"
}

install_nvidia_drivers_amzn2023() {
  log "Installing NVIDIA drivers on Amazon Linux 2023..."
  # Ensure up-to-date kernel and tools
  sudo dnf -y update
  sudo dnf -y install gcc make kernel-devel-$(uname -r) dkms
  # AWS-recommended NVIDIA driver (from AWS/NVIDIA repos)
  # Try aws-nvidia-driver if present; otherwise fall back to NVIDIA repo
  if sudo dnf list --available 2>/dev/null | grep -qi '^nvidia-driver'; then
    sudo dnf -y install nvidia-driver nvidia-modprobe
  else
    # Fallback: enable NVIDIA CUDA repo for driver
    sudo dnf -y install dnf-plugins-core
    sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo
    sudo dnf -y install nvidia-driver nvidia-modprobe
  fi
  touch "${NEEDS_REBOOT_FLAG}"
}

install_ollama() {
  if ! command -v ollama >/dev/null 2>&1; then
    log "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
  else
    log "Ollama already installed."
  fi

  if [ "${AUTO_START}" = "true" ]; then
    if command -v systemctl >/dev/null 2>&1; then
      sudo systemctl enable ollama || true
      sudo systemctl restart ollama || true
    fi
  fi
}

pull_llava() {
  log "Pulling model: ${MODEL_TAG}"
  ollama pull "${MODEL_TAG}"
  log "Done. Test with:  ollama run ${MODEL_TAG}"
}

main() {
  log "Detected instance type should be g4dn.xlarge (T4). Ensure you're on that instance for GPU."
  DISTRO="$(detect_distro)"
  log "Distro: ${DISTRO}"

  if has_nvidia; then
    log "NVIDIA driver detected:"
    nvidia-smi || true
  else
    warn "No working NVIDIA driver detected. Installing..."
    case "$DISTRO" in
      ubuntu:20.04|ubuntu:22.04|ubuntu:24.04|ubuntu:*)
        install_nvidia_drivers_ubuntu
        ;;
      amzn:2023|amzn:*)
        install_nvidia_drivers_amzn2023
        ;;
      *)
        err "Unsupported distro: $DISTRO. Use Ubuntu 22.04/24.04 or Amazon Linux 2023."
        exit 1
        ;;
    esac
  fi

  if [ -f "${NEEDS_REBOOT_FLAG}" ]; then
    warn "Reboot required to load NVIDIA driver. Rebooting now..."
    sudo reboot
  fi

  # From here on, we expect the driver to be active.
  if ! has_nvidia; then
    err "NVIDIA driver still not available. Please ensure Secure Boot is disabled and rerun after manual driver install."
    exit 1
  fi

  install_ollama
  pull_llava

  log "All set. Example (image understanding):"
  cat <<'EOF'
# After this, you can do:
# 1) Copy an image to the server, e.g. ./test.jpg
# 2) Run:
#    ollama run llava-llama3:latest
#    >> Send a system prompt, then when prompted drag&drop isn't available over SSH,
#       instead use the API:
#    curl -s http://localhost:11434/api/generate -d '{
#      "model":"llava-llama3:latest",
#      "prompt":"Describe this image",
#      "images":["'"$(base64 -w0 test.jpg)"'"]
#    }' | jq -r '.response' | tr -d '\n'
EOF
}

main "$@"
