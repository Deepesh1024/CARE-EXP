import torch
import types
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import BASE_MODEL_ID, HF_REVISION, TARGET_LAYER_IDX, DEVICE, DTYPE

# The patched forward function
def patched_experts_forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor) -> torch.Tensor:
    final_hidden_states = torch.zeros_like(hidden_states)
    mask_config = getattr(self, "_mask_config", None)
    if not hasattr(self, "_captured_activations"): self._captured_activations = []
    
    with torch.no_grad():
        expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    for expert_idx in expert_hit:
        expert_idx_val = expert_idx[0].item()
        if expert_idx_val == self.num_experts: continue
            
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx_val])
        current_state = hidden_states[token_idx]
        
        gate, up = torch.nn.functional.linear(current_state, self.gate_up_proj[expert_idx_val]).chunk(2, dim=-1)
        current_hidden_states = self.act_fn(gate) * up
        
        leaf_tensor = current_hidden_states.detach().requires_grad_(True)
        current_hidden_states = leaf_tensor
            
        self._captured_activations.append({
            "expert_idx": expert_idx_val,
            "tensor": leaf_tensor,
            "num_tokens": leaf_tensor.size(0)
        })

        out = torch.nn.functional.linear(current_hidden_states, self.down_proj[expert_idx_val])
        out = out * top_k_weights[token_idx, top_k_pos, None]
        
        final_hidden_states = final_hidden_states.index_add(0, token_idx, out.to(final_hidden_states.dtype))

    return final_hidden_states

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID, 
    revision=HF_REVISION,
    torch_dtype=DTYPE, 
    device_map=DEVICE, 
    trust_remote_code=True
)

for param in model.parameters():
    param.requires_grad = False

experts_module = model.model.layers[TARGET_LAYER_IDX].mlp.experts
experts_module.forward = types.MethodType(patched_experts_forward, experts_module)
experts_module._captured_activations = []

model.train()

input_ids = tokenizer("Hello world", return_tensors="pt")["input_ids"].to(DEVICE)
print("Running forward...")
outputs = model(input_ids)

print(f"logits requires_grad: {outputs.logits.requires_grad}")
