# EXPERIMENT 8: GWS VALIDITY — DRY-RUN AUDIT

## 1. Architectural Configuration
1. **Exact model architecture**: `allenai/OLMoE-1B-7B-0924`
2. **Exact 0-indexed selected layer**: Layer `8` (middle layer of the 16 layers).
3. **Hidden dimension**: `2048`
4. **Number of experts**: `64`
5. **Router top-k**: `8`
6. **Whether router weights are renormalized**: Yes, OLMoE-1B-7B routes top-8 and renormalizes.

## 2. Mathematical Tensors & Hooks
7. **Exact tensor representing h**: The accumulated output of the selected MoE layer *before* the residual addition, i.e., $h(x) = \sum_{k \in \text{top-8}} g_k(x) e_k(x)$. Shape: `[batch_size, seq_len, 2048]`.
8. **Exact point where grad_h is captured**: By registering a backward hook (`register_hook`) directly on the tensor $h$ returned by the MoE router execution inside the layer forward pass. The gradient $\nabla_h L$ represents the derivative of the next-token prediction CrossEntropy loss with respect to this un-residualized hidden state.
9. **Exact expert-output tensor shape for a single chunk**: `[num_routed_tokens, 2048]`. For dense calculation of all 64 experts for dot-product caching, it would be `[total_tokens, 64, 2048]`.
10. **Exact router-weight tensor shape**: `[total_tokens, 64]`.

## 3. Implementation of GWS (Split A)
11. **Exact s_i calculation**: $s_i(x) = \langle \nabla_h(x), e_i(x) \rangle$. This is the dot product over the 2048 dimension. Shape of $s_i$: `[total_tokens]`. We store $S \in \mathbb{R}^{N \times 64}$.
12. **How GWS is computed without materializing [N, 64, 2048]**: 
    - In Split A, during the forward pass, we cache the pre-activations (inputs to the MLP) for layer 8.
    - We compute the full dense expert outputs for all 64 experts: $E_{dense} \in \mathbb{R}^{N_{chunk} \times 64 \times 2048}$.
    - We run the backward pass to obtain $\nabla_h L \in \mathbb{R}^{N_{chunk} \times 2048}$.
    - Inside the hook (or post-backward), we compute $S = \text{einsum}('nd,ned \rightarrow ne', \nabla_h L, E_{dense})$.
    - We discard $E_{dense}$ and $\nabla_h L$ to free VRAM. We accumulate only $S$ and the routing weights $G \in \mathbb{R}^{N \times 64}$ to CPU memory.
    - Finally, GWS for $j \rightarrow i$ is calculated as $\mathbb{E}_x [ g_j(x) (S_{x, i} - S_{x, j}) ]$.

## 4. Implementation of Actual Damage (Split B)
13. **Exact h_sub implementation**: During the forward pass on Split B, we inject a forward hook in layer 8. If testing $j \rightarrow i$, for any token where $j$ is in the top-k:
    - We compute the standard $h(x) = \sum_{k} g_k(x) e_k(x)$.
    - We explicitly subtract $g_j(x) e_j(x)$ and add $g_j(x) e_i(x)$.
    - The hook returns $h_{sub}(x) = h(x) - g_j(x) e_j(x) + g_j(x) e_i(x)$.
14. **Exact CE token-shift/indexing**: `shift_logits = logits[..., :-1, :].contiguous()` and `shift_labels = labels[..., 1:].contiguous()`. The loss is evaluated via standard PyTorch `CrossEntropyLoss(reduction='mean')`.

## 5. Execution Parameters
15. **Split A / Split B sizes**: Using a 512-sequence subset of WikiText-2 (seq_len=1024). Split A = 256 sequences (262,144 tokens), Split B = 256 sequences.
16. **Expected peak VRAM**: With `batch_size=1`, calculating $E_{dense}$ for 64 experts takes $1024 \times 64 \times 2048 \times 2 = 268$ MB. Model in `bfloat16` takes ~14 GB. Peak VRAM will be around 16 GB, fitting easily on a 24 GB GPU.
17. **Actual expected dtype**: `torch.bfloat16` for model weights and forward pass.
18. **Whether the computation graph is retained only where required**: Yes. `requires_grad=False` globally, except for the input to Layer 8, enabling gradient flow *only* from the loss down to Layer 8, drastically saving memory.
19. **Numerical precision used for gradient accumulation**: FP32 for the dot products and loss aggregation in CPU to avoid underflow/overflow.
20. **Baseline noise-floor procedure**: Before substitutions, we run Split B twice with zero substitutions. We calculate $\text{CE}_1$ and $\text{CE}_2$. The noise floor is $| \text{CE}_1 - \text{CE}_2 |$. Substitutions with $\Delta \text{CE} \le \text{noise\_floor}$ are marked as "UNRESOLVED".
