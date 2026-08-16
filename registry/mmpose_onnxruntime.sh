#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ONNX_CPU_VER="1.8.1"
ONNX_GPU_VER="1.8.1"

cpu_dir="${ROOT_DIR}/onnxruntime-linux-x64-${ONNX_CPU_VER}"
gpu_dir="${ROOT_DIR}/onnxruntime-linux-x64-gpu-${ONNX_GPU_VER}"

if [[ ! -d "${cpu_dir}" ]]; then
    echo "Downloading ONNX Runtime CPU ${ONNX_CPU_VER}..."
    wget "https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_CPU_VER}/onnxruntime-linux-x64-${ONNX_CPU_VER}.tgz"
    tar -zxvf "onnxruntime-linux-x64-${ONNX_CPU_VER}.tgz"
fi

if [[ ! -d "${gpu_dir}" ]]; then
    echo "Downloading ONNX Runtime GPU ${ONNX_GPU_VER}..."
    wget "https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_GPU_VER}/onnxruntime-linux-x64-gpu-${ONNX_GPU_VER}.tgz"
    tar -zxvf "onnxruntime-linux-x64-gpu-${ONNX_GPU_VER}.tgz"
fi

export ONNXRUNTIME_DIR="${gpu_dir}"
export LD_LIBRARY_PATH="${ONNXRUNTIME_DIR}/lib:${LD_LIBRARY_PATH:-}"

python -m pip install -U pip
python -m pip install onnxruntime-gpu=="${ONNX_GPU_VER}"
python -m pip install -U openmim

declare -a MODEL_OPTIONS=(
    "rtmpose-m_8xb64-270e_coco-wholebody-256x192"
    "rtmpose-l_8xb32-270e_coco-wholebody-384x288"
    "rtmpose-s_8xb256-420e_coco-256x192"
)

echo "Choose which MMPose model config to download:"
for i in "${!MODEL_OPTIONS[@]}"; do
    idx=$((i + 1))
    echo "  ${idx}) ${MODEL_OPTIONS[$i]}"
done
echo "  c) Custom config"

read -r -p "Selection [1-${#MODEL_OPTIONS[@]} or c]: " selection

case "${selection}" in
    1|2|3)
        model_config="${MODEL_OPTIONS[$((selection - 1))]}"
        ;;
    c|C)
        read -r -p "Enter full MMPose config name: " model_config
        if [[ -z "${model_config}" ]]; then
            echo "Custom config cannot be empty."
            exit 1
        fi
        ;;
    *)
        echo "Invalid selection: ${selection}"
        exit 1
        ;;
esac

read -r -p "Download destination directory [default: ${ROOT_DIR}]: " dest_dir
dest_dir="${dest_dir:-${ROOT_DIR}}"
mkdir -p "${dest_dir}"

echo "Downloading MMPose config: ${model_config}"
mim download mmpose --config "${model_config}" --dest "${dest_dir}"

echo "Done. Model files downloaded to: ${dest_dir}"