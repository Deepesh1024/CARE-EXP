```mermaid
flowchart TD
    subgraph Static Compression
        A1[Initial Model M_0] --> B1[Compute Static Geometry]
        B1 --> C1[Generate Frozen Ranking]
        C1 --> D1[Merge Pair 1]
        D1 --> E1[Merge Pair 2]
        E1 --> F1[Merge Pair 3]
    end

    subgraph CARE-COM (Adaptive)
        A2[Model State M_t] --> B2[Extract Capability State C_t]
        B2 --> C2[Generate Candidate Set P_t]
        C2 --> D2[Evaluate Intervention Damage D(i,j)]
        D2 --> E2[Execute Lowest-Damage Merge]
        E2 --> F2[Updated Model M_t+1]
        F2 -->|Recompute| A2
    end
```
