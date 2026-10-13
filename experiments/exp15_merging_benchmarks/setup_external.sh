#!/bin/bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$DIR/external"
cd "$DIR/external"

echo "[Setup] Cloning official HC-SMoE repository..."
if [ ! -d "HC-SMoE" ]; then
    git clone --depth 1 https://github.com/wazenmai/HC-SMoE.git
fi

echo "[Setup] Cloning official MC-SMoE repository..."
if [ ! -d "MC-SMoE" ]; then
    git clone --depth 1 https://github.com/UNITES-Lab/MC-SMoE.git
fi

echo "[Setup] Upstream repositories cloned successfully!"
