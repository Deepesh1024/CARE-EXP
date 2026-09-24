# Equations

1. **Capability Representation**
$$ C_i \in \mathbb{R}^d $$

2. **Capability Distance**
$$ d_C(i, j) = \|C_i - C_j\|_2 $$

3. **Functional Intervention Damage**
$$ D(i, j \mid M_t) = \mathbb{E}_x \left[ \text{KL}(P_{M_t}(\cdot \mid x) \| P_{M_t^{(i,j)}}(\cdot \mid x)) \right] $$

4. **Candidate Set**
$$ P_t = \text{NearestPairs}(C_t, K) $$

5. **Selection Rule**
$$ (i^*, j^*) = \text{argmin}_{(i,j) \in P_t} D(i, j \mid M_t) $$

6. **Physical Merge**
$$ W_{i^*}' = \alpha W_i + (1-\alpha) W_j $$
*(Implemented via linear combination weighted by activation frequency).*

7. **State Update**
$$ M_{t+1} = \text{Merge}(M_t, i^*, j^*) $$

8. **Capability Recomputation**
$$ C_{t+1} = \text{ExtractCapabilities}(M_{t+1}) $$
