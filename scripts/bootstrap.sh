#!/usr/bin/env bash
set -euo pipefail

# TODO: Add environment bootstrap checks for CUDA, ROS2, and GStreamer.
python -m pip install --upgrade pip
python -m pip install -e .[dev]
