import torch
import torch.nn as nn
from einops import rearrange
import math
import torch.nn.functional as F


"""
Patchify:
    - Input: (batch_size, channels, height, width)
    - Output: (batch_size, num_patches, patch_dim)
    - Patch dim is the number of channels * the number of pixels in the patch
    - Num patches is the number of patches in the image
    - Patch size is the size of the patch
    - Patchify is a function that takes an image and returns a tensor of patches
    - You can assume the image is square
    - Naive application of .reshape will not organize the pixels into the correct patches! You must do this manually (or with einops)
"""


class Patchify(nn.Module):
    def __init__(self, patch_size=8, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Implement Patchify
        self.patch_size = patch_size

    def forward(self, x):
        # Implement Patchify
        P = self.patch_size
        return rearrange(
            x,
            "b c (h ph) (w pw) -> b (h w) (c ph pw)",
            ph=P,
            pw=P
        )


"""
Unpatchify:
    - Input: (batch_size, num_patches, patch_dim)
    - Output: (batch_size, channels, height, width)
    - Patch dim is the number of channels * the number of pixels in the patch
    - Num patches is the number of patches in the image
    - Patch size is the size of the patch
    - Unpatchify is a function that takes a tensor of patches and returns an image
    - You can assume the image is square
"""


class Unpatchify(nn.Module):
    def __init__(self, patch_size, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.patch_size = patch_size

    def forward(self, x):
        batches, patches, patch_dim = x.shape
        h = w = int(math.sqrt(patches))
        channels = patch_dim // (self.patch_size * self.patch_size)

        return rearrange(
            x,
            "b (h w) (c ph pw) -> b c (h ph) (w pw)",
            h=h,
            w=w,
            ph=self.patch_size,
            pw=self.patch_size,
            c=channels
        )




'''
FeedForward:
    - Input: (batch_size, num_patches, hidden_dim)
    - Output: (batch_size, num_patches, hidden_dim)
    - Hidden dim is the dimension of the hidden state
    - Num patches is the number of patches in the image
    - FeedForward is a function that takes a tensor of patches and returns a tensor of patches
    - You can assume the image is square
    - Refer to Attention is all you need Section 3.3
'''
class FeedForward(nn.Module):
    def __init__(self, hidden_dim, inner_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        ## Implement FeedForward
        ## Modules needed: 2x Linear, ReLU
        self.linear_1 = nn.Linear(hidden_dim, inner_dim)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(inner_dim, hidden_dim)

    def forward(self, x):
        ## Implement FeedForward
        out = self.linear_1(x)
        out = self.relu(out)
        out = self.linear_2(out)
        return out



'''
SelfAttention:
    - Input: (batch_size, num_patches, hidden_dim)
    - Output: (batch_size, num_patches, hidden_dim)
    - Hidden dim is the dimension of the hidden state
    - Num patches is the number of patches in the image
    - SelfAttention is a function that takes a tensor of patches and returns a tensor of patches
    - You can assume the image is square
    - Refer to Attention is all you need Section 3.2.1 
'''
class SelfAttention(nn.Module):
    def __init__(self, hidden_dim, inner_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        ## Implement SelfAttention
        ## Modules needed: 3x Linear
        self.Q = nn.Linear(hidden_dim, inner_dim)
        self.K = nn.Linear(hidden_dim, inner_dim)
        self.V = nn.Linear(hidden_dim, inner_dim)
        self.out = nn.Linear(inner_dim, hidden_dim)
        self.scale = inner_dim ** 0.5

    def forward(self, x, cos=None, sin=None):
        ## Implement SelfAttention
        q = self.Q(x)
        k = self.K(x)
        v = self.V(x)

        if cos is not None and sin is not None:
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) 
        attn_scores = attn_scores / self.scale
        attn_weights = F.softmax(attn_scores, dim=-1) @ v 
        return attn_weights


'''
MultiHeadSelfAttn:
    - Input: (batch_size, num_patches, hidden_dim)
    - Output: (batch_size, num_patches, hidden_dim)
    - Hidden dim is the dimension of the hidden state
    - Num patches is the number of patches in the image
    - MultiHeadSelfAttn is a function that takes a tensor of patches and returns a tensor of patches
    - You can assume the image is square
    - Refer to Attention is all you need Section 3.2.2
'''

class MultiHeadSelfAttn(nn.Module):
    def __init__(self, hidden_dim, num_heads, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        ## Implement MultiHeadSelfAttn, you must use the SelfAttention modules you implemented above
        ## Modules needed: num_heads x SelfAttention, Linear 
        self.attn_heads = nn.ModuleList([SelfAttention(hidden_dim, hidden_dim // num_heads) for _ in range(num_heads)])
        self.out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, cos=None, sin=None):
        ## Implement MultiHeadSelfAttn
        res = [head(x, cos, sin) for head in self.attn_heads]
        res = torch.cat(res, dim=-1)
        return self.out(res)


"""
DiTBlock:
    - Input: (batch_size, num_patches, hidden_dim)
    - Output: (batch_size, num_patches, hidden_dim)
    - Hidden dim is the dimension of the hidden state
    - Num patches is the number of patches in the image
    - DiTBlock is a block that takes a tensor of patches and returns a tensor of patches
    - You can assume the image is square
"""

class DiTBlock(nn.Module):

    def __init__(self, hidden_dim, num_heads, ff_dim, time_emb_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        ## Implement DiTBlock
        ## Modules needed: 2x LN, MLP (can be implemented using nn.Sequential), FeedForward, MultiHeadSelfAttention
        ## zero initialize the linear layers in the MLP
        self.model = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim // 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim // 4, hidden_dim * 6)
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.attn = MultiHeadSelfAttn(hidden_dim, num_heads)
        self.ff = FeedForward(hidden_dim, ff_dim)
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
        
    ## X is the input patches, cond is the time embedding
    def forward(self, x, cond, cos=None, sin=None):

        ## Implement DiTBlock, x is the input patches, cond is the time embedding

        ## Step 1: Get the parameters from the condition
        mod = self.model(cond)
        alpha_1, beta_1, y_1, alpha_2, beta_2, y_2 = mod.chunk(6, dim=-1)

        ## Step 2: Apply layer normalization
        layer_norm = self.norm1(x)

        ## Step 3: Apply scale and shift
        out = (y_1.unsqueeze(1) + 1) * layer_norm + beta_1.unsqueeze(1) 

        ## Step 4: Apply multi-head self-attention
        attn_out = self.attn(out, cos, sin)
        out = out + attn_out

        ## Step 5: Scale output
        scaled_out = (alpha_1.unsqueeze(1)) * out
        ## Step 6: Add the original input
        scaled_out = scaled_out + x

        ## Step 7: Apply layer normalization
        norm_2 = self.norm2(scaled_out)

        ## Step 8: Apply scale and shift
        out_2 = (y_2.unsqueeze(1) + 1) * norm_2 + beta_2.unsqueeze(1) 

        ## Step 9: Apply feedforward
        out_ff = self.ff(out_2)

        ## Step 10: Scale output
        scaled_out_2 = (alpha_2.unsqueeze(1)) * out_ff

        ## Step 11: Add residual connection
        scaled_out_2 = scaled_out_2 + scaled_out

        return scaled_out_2



## (FOR FLOW MATCHING ONLY) 
class ContinuousTimestepEmbedder(nn.Module):
    def __init__(self, emb_dim, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t = t.view(t.size(0))
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class DiT(nn.Module):

    def get_position_embedding(self, num_patches, patch_size, hidden_dim):
        grid_size = int(patch_size ** 0.5)
        dim = hidden_dim // 2
        
        pos = torch.arange(num_patches, dtype=torch.float32)
        r = pos // grid_size
        c = pos % grid_size
        
        omega = torch.arange(0, dim, 2, dtype=torch.float32)
        omega = 1.0 / (10000 ** (omega / dim))
        
        out_r = r[:, None] @ omega[None, :]
        out_c = c[:, None] @ omega[None, :]
        
        row_emb = torch.zeros((1, num_patches, dim))
        col_emb = torch.zeros((1, num_patches, dim))
        
        row_emb[:, :, 0::2] = torch.sin(out_r)
        row_emb[:, :, 1::2] = torch.cos(out_r)
        col_emb[:, :, 0::2] = torch.sin(out_c)
        col_emb[:, :, 1::2] = torch.cos(out_c)
        
        pos_emb = torch.cat([row_emb, col_emb], dim=-1)
        pos_emb = nn.Parameter(pos_emb, requires_grad=False)
        return pos_emb

    def __init__(self, patch_size, num_blocks, num_heads, ff_dim, time_emb_dim, num_timesteps, hidden_dim, num_patches, num_channels=3, training_type="ddpm", pos_embed_type="fixed", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if pos_embed_type == "fixed":
            self.pos_emb = self.get_position_embedding(num_patches, patch_size, hidden_dim)
        elif pos_embed_type == "rope":
            cos, sin = precompute_rope_freqs_2d(hidden_dim // num_heads, num_patches, patch_size)
            self.register_buffer("rope_cos", cos)
            self.register_buffer("rope_sin", sin)
        
        self.pos_embed_type = pos_embed_type

        ## Implement DiT
        ## Modules needed: Patchify, Linear, Embedding, Positional Embedding (use provided get_position_embedding), num_blocks x DiTBlock, LayerNorm, Linear, Unpatchify
        ## zero initialize the final linear layer
        self.patch_size = patch_size
        self.training_type = training_type

        patch_dim = num_channels * patch_size * patch_size

        self.patchify = Patchify(patch_size)
        self.patch_proj = nn.Linear(patch_dim, hidden_dim)

        if training_type == "ddpm":
            self.time_emb = nn.Embedding(num_timesteps, time_emb_dim)
        else:
            self.time_emb = ContinuousTimestepEmbedder(time_emb_dim)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads, ff_dim, time_emb_dim)
            for _ in range(num_blocks)
        ])

        self.norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, patch_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        self.unpatchify = Unpatchify(patch_size)

    def forward(self, image, timestep):

        ## Step 1: Patchify the image
        x = self.patchify(image)

        ## Step 2: Project the patches to the hidden dimension
        x = self.patch_proj(x)

        ## Step 3: Add the positional encoding
        # x = x + self.pos_emb.to(x.device)
        if self.pos_embed_type == "fixed":
            x = x + self.pos_emb.to(x.device)
            cos, sin = None, None
        else:
            cos, sin = self.rope_cos, self.rope_sin

        ## Step 4: Create the time embedding
        if self.training_type == "ddpm":
            t_emb = self.time_emb(timestep.view(-1).long())
        else:
            t_emb = self.time_emb(timestep)

        ## Step 5: Apply the DiT blocks
        for block in self.blocks:
            x = block(x, t_emb, cos, sin)

        ## Step 6: Apply the layer normalization
        x = self.norm(x)

        ## Step 7: Project the hidden dimension back to the patch dimension
        x = self.out_proj(x)

        ## Step 8: Unpatchify the image
        x = self.unpatchify(x)

        return x
    




#### ROPE HELPER FUNCTIONS ####    

def precompute_rope_freqs_2d(head_dim, num_patches, patch_size):
    """
    Precompute 2D RoPE cos/sin frequencies for image patches.
    Each patch has a (row, col) position, so we split head_dim in half:
    half for row frequencies, half for col frequencies.
    """
    grid_size = int(num_patches ** 0.5)
    half_dim = head_dim // 2  # half for row, half for col

    omega = 1.0 / (10000 ** (torch.arange(0, half_dim, 2).float() / half_dim))

    pos = torch.arange(num_patches).float()
    row = pos // grid_size  # (num_patches,)
    col = pos % grid_size   # (num_patches,)

    row_freqs = torch.outer(row, omega)
    col_freqs = torch.outer(col, omega)

    
    row_cos = torch.cos(row_freqs)  # (num_patches, half_dim/2)
    row_sin = torch.sin(row_freqs)
    col_cos = torch.cos(col_freqs)
    col_sin = torch.sin(col_freqs)

    cos = torch.cat([row_cos, col_cos], dim=-1)  # (num_patches, half_dim)
    sin = torch.cat([row_sin, col_sin], dim=-1)

    return cos, sin  # each: (num_patches, head_dim//2)


def apply_rope(x, cos, sin):
    """
    Apply RoPE to a query or key tensor.
    x shape: (batch, num_patches, dim)
    cos/sin shape: (num_patches, dim//2)
    """
    x1 = x[..., ::2]   # even indices: (batch, num_patches, dim//2)
    x2 = x[..., 1::2]  # odd indices:  (batch, num_patches, dim//2)

    # Rotate
    x_rotated = torch.stack([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos
    ], dim=-1)  # (batch, num_patches, dim//2, 2)

    # Flatten last two dims back to (batch, num_patches, dim)
    return x_rotated.flatten(-2)