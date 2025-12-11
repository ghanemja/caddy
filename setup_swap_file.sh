#!/bin/bash
# Setup massive swap file for system RAM (helps with 16GB RAM limitation)
# This allows the system to use disk space as virtual RAM when physical RAM is full

set -e

SWAP_SIZE_GB=${1:-32}  # Default to 32GB swap, can override: ./setup_swap_file.sh 64
SWAP_FILE="/swapfile"

echo "=========================================="
echo "Setting up ${SWAP_SIZE_GB}GB swap file"
echo "=========================================="

# Check if swap already exists
if [ -f "$SWAP_FILE" ]; then
    echo "⚠ Swap file already exists at $SWAP_FILE"
    read -p "Remove existing swap and create new one? (yes/no): " CONFIRM
    if [ "$CONFIRM" != "yes" ]; then
        echo "Cancelled."
        exit 0
    fi
    echo "Removing existing swap..."
    sudo swapoff "$SWAP_FILE" 2>/dev/null || true
    sudo rm -f "$SWAP_FILE"
fi

# Check available disk space
AVAILABLE_GB=$(df -BG / | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "$AVAILABLE_GB" -lt "$SWAP_SIZE_GB" ]; then
    echo "⚠ Warning: Only ${AVAILABLE_GB}GB available, but requesting ${SWAP_SIZE_GB}GB swap"
    read -p "Continue anyway? (yes/no): " CONFIRM
    if [ "$CONFIRM" != "yes" ]; then
        echo "Cancelled."
        exit 0
    fi
fi

echo ""
echo "Creating ${SWAP_SIZE_GB}GB swap file (this may take a few minutes)..."
sudo fallocate -l ${SWAP_SIZE_GB}G "$SWAP_FILE" || sudo dd if=/dev/zero of="$SWAP_FILE" bs=1G count=$SWAP_SIZE_GB

echo "Setting secure permissions..."
sudo chmod 600 "$SWAP_FILE"

echo "Formatting as swap..."
sudo mkswap "$SWAP_FILE"

echo "Enabling swap..."
sudo swapon "$SWAP_FILE"

echo ""
echo "=========================================="
echo "✓ Swap file created and enabled!"
echo "=========================================="
echo ""
echo "Current swap status:"
free -h
echo ""
echo "To make this permanent (survive reboots), add to /etc/fstab:"
echo "  $SWAP_FILE none swap sw 0 0"
echo ""
echo "To remove swap later:"
echo "  sudo swapoff $SWAP_FILE"
echo "  sudo rm $SWAP_FILE"
echo ""
