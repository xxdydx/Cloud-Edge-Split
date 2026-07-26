"""Shared partial-model loading for the edge/cloud split.

Loads only the layer range a device actually needs, plus whichever of
embed_tokens/lm_head that device requires, instead of materializing the
full checkpoint on both sides. This is what makes splitting a large model
(where loading it twice would exceed either device's memory) feasible.
"""

from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM


def num_hidden_layers(model_name):
    return AutoConfig.from_pretrained(model_name).num_hidden_layers


def load_partial_model(model_name, torch_dtype, device, start_layer, end_layer,
                        need_embed, need_lm_head):
    """Load model_name, keep only layers[start_layer:end_layer] plus the
    requested head, then move the reduced model to device.

    Loading happens on CPU first so the discarded layers/heads are freed
    before anything is moved to a (potentially memory-constrained) GPU.
    """
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype)
    model.model.layers = nn.ModuleList(model.model.layers[start_layer:end_layer])
    if not need_embed:
        model.model.embed_tokens = None
    if not need_lm_head:
        model.lm_head = None
    model.eval()
    return model.to(device)
