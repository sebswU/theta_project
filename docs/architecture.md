# Architecture

Universal-CV-Adapter uses a composable, adapter-first architecture where each research model is
integrated through a thin plugin layer.

```mermaid
flowchart LR
    S[Sources] --> D[Discovery Providers]
    D --> C[Capability Detector]
    C --> P[Pipeline Planner]
    P --> M[Model Registry]
    P --> F[Fusion Registry]
    M --> X[Execution Graph]
    F --> X
    X --> G[Scene Graph]
    G --> V[Visualization Backends]
```

## TODO

- TODO: Add runtime sequence and fault-handling diagrams.
