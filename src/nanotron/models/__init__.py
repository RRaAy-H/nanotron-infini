# flake8: noqa
from .base import DTypeInvariantTensor, NanotronModel, build_model, check_model_has_grad, init_on_device_and_dtype

# Add LlamaForCausalLM as an alias for LlamaForTraining for compatibility
from .llama import LlamaForTraining
LlamaForCausalLM = LlamaForTraining
