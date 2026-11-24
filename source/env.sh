# source me. Lab 3 cores: 0-15 only (devices 0-3).
LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export V=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0
export PATH=$V/bin:$PATH
export NEURON_PLATFORM_TARGET_OVERRIDE=trn2
export NEURON_RT_VISIBLE_CORES=${NEURON_RT_VISIBLE_CORES:-0-3}
export NEURON_RT_DISABLE_EXECUTION_BARRIER=1
export PYTHONPATH=$LAB_DIR/kernels:$LAB_DIR/probes:$LAB_DIR/tests:$PYTHONPATH
