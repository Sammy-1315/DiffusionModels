# Diffusion Models from Scratch

PyTorch implementations of DDPM, Flow Matching, and Latent Diffusion built around a custom Diffusion Transformer (DiT) backbone.

| File | Description |
|------|-------------|
| `Diffusion.ipynb` | Training + sampling for all three paradigms |
| `Diffusion_Transformer.py` | Full DiT architecture from scratch |

**Stack:** Python · PyTorch · einops

---

## DDPM

```mermaid
flowchart LR
    subgraph fwd ["⬆️ Forward Process  q(xₜ | xₜ₋₁)  —  fixed"]
        direction LR
        x0(["x₀\nclean image"]):::clean -->|"+ ε ~ N(0,I)"| x1(["x₁"]):::mid
        x1 -->|"+ ε ~ N(0,I)"| xd(["  ...  "]):::mid
        xd -->|"+ ε ~ N(0,I)"| xT(["xT\nN(0,I)"]):::noise
    end

    subgraph rev ["⬇️ Reverse Process  pθ(xₜ₋₁ | xₜ)  —  learned"]
        direction LR
        xT2(["xT\nN(0,I)"]):::noise -->|"ε_θ(xₜ, t)"| xd2(["  ...  "]):::mid
        xd2 -->|"ε_θ(xₜ, t)"| x12(["x₁"]):::mid
        x12 -->|"ε_θ(xₜ, t)"| x02(["x₀\ngenerated"]):::clean
    end

    subgraph train ["🎯 Training Objective"]
        direction LR
        s1(["x₀, ε"]):::input --> s2["xₜ = √ᾱₜ x₀ + √1-ᾱₜ ε"] --> s3["ε_θ(xₜ, t)"] --> s4(["MSE(ε, ε_θ)"]):::loss
    end

    classDef clean fill:#1e293b,stroke:#22c55e,color:#86efac
    classDef noise fill:#1e293b,stroke:#ef4444,color:#fca5a5
    classDef mid fill:#1e293b,stroke:#64748b,color:#94a3b8
    classDef input fill:#1e293b,stroke:#3b82f6,color:#93c5fd
    classDef loss fill:#1e293b,stroke:#f59e0b,color:#fde68a
```

---

## Flow Matching

```mermaid
flowchart LR
    subgraph path ["Conditional Flow  x(t) = (1-t)z + tx₀"]
        direction LR
        z(["z ~ N(0,I)\nt = 0"]):::noise
        m1(["x(0.33)"]):::mid
        m2(["x(0.66)"]):::mid
        x0(["x₀ ~ p_data\nt = 1"]):::clean
        z -->|"straight line"| m1 --> m2 --> x0
    end

    subgraph vf ["🎯 Learning the Vector Field"]
        direction LR
        xt(["x(t)"]):::mid --> vtheta["vθ(x,t)"]
        target(["target: x₀ - z"]):::clean --> loss2(["MSE(vθ, x₀ - z)"]):::loss
        vtheta --> loss2
    end

    subgraph sample ["🔁 Sampling  (ODE integration)"]
        direction LR
        s1(["z ~ N(0,I)"]):::noise --> s2["dz/dt = vθ(z,t)"] --> s3(["x₀ generated"]):::clean
    end

    classDef clean fill:#1e293b,stroke:#22c55e,color:#86efac
    classDef noise fill:#1e293b,stroke:#ef4444,color:#fca5a5
    classDef mid fill:#1e293b,stroke:#3b82f6,color:#93c5fd
    classDef loss fill:#1e293b,stroke:#f59e0b,color:#fde68a
```

---

## Latent Diffusion

```mermaid
flowchart LR
    img(["🖼️ Image\nH × W × C"]):::input

    subgraph vae_enc ["VAE Encoder"]
        enc["E(x)"]
    end

    subgraph latent ["Diffusion in Latent Space"]
        z(["z\nh × w × d"]):::latent
        znoise["add noise\nz₀ → zₜ"]
        zdenoise["denoise\nε_θ(zₜ, t)"]
        z0hat(["ẑ₀"]):::latent
        z --> znoise --> zdenoise --> z0hat
    end

    subgraph vae_dec ["VAE Decoder"]
        dec["D(ẑ₀)"]
    end

    out(["🖼️ Generated\nImage"]):::output

    img --> enc --> z
    z0hat --> dec --> out

    classDef input fill:#1e293b,stroke:#3b82f6,color:#93c5fd
    classDef output fill:#1e293b,stroke:#22c55e,color:#86efac
    classDef latent fill:#1e293b,stroke:#a855f7,color:#d8b4fe
```

Diffusion runs in a compressed latent space — much cheaper than pixel space — then the decoder reconstructs the final image.

---

## DiT Architecture

```mermaid
flowchart TD
    img(["🖼️ Image\nB × C × H × W"]):::input
    t(["⏱️ Timestep t"]):::input

    subgraph prep ["Preprocessing"]
        direction LR
        patch["Patchify\nB×C×H×W → B×N×C·P²"]
        proj["Linear Projection\nB×N×C·P² → B×N×D"]
        temb["Embedding + MLP\nt → t_emb ∈ ℝᴰ"]
        patch --> proj
    end

    subgraph block ["DiT Block  (repeated × N)"]
        direction LR
        modulate["MLP(t_emb)\n→ γ₁ β₁ α₁  γ₂ β₂ α₂"]
        ln1["adaLN\nγ₁ · LN(x) + β₁"]
        attn["Multi-Head\nSelf-Attention"]
        res1(["residual  α₁·Attn + x"]):::res
        ln2["adaLN\nγ₂ · LN(x) + β₂"]
        ff["Feed\nForward"]
        res2(["residual  α₂·FFN + x"]):::res

        modulate --> ln1 --> attn --> res1
        modulate --> ln2 --> ff --> res2
    end

    subgraph decode ["Output"]
        direction LR
        norm["LayerNorm"]
        outproj["Linear  (zero-init)\nB×N×D → B×N×C·P²"]
        unpatch["Unpatchify\nB×N×C·P² → B×C×H×W"]
        norm --> outproj --> unpatch
    end

    img --> patch
    t --> temb
    proj --> block
    temb --> modulate
    res2 --> decode
    unpatch --> out(["ε_θ or vθ\nB × C × H × W"]):::output

    classDef input fill:#1e293b,stroke:#3b82f6,color:#93c5fd
    classDef output fill:#1e293b,stroke:#22c55e,color:#86efac
    classDef res fill:#1e293b,stroke:#a855f7,color:#d8b4fe
```

The `α`, `β`, `γ` modulation params are zero-initialized so each block starts as an identity — stable early in training.

---

## Setup

```bash
pip install -r requirements.txt
jupyter notebook Diffusion.ipynb
```
