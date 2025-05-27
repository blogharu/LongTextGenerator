import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryEmbedding(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, hidden_size, 2).float() / hidden_size)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        batch_size, sequence_length, hidden_size = x.shape
        device = x.device

        t = torch.arange(sequence_length, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)

        emb = torch.cat((freqs.sin(), freqs.cos()), dim=-1)

        emb = emb.unsqueeze(0).expand(batch_size, -1, -1)

        x1, x2 = x[..., ::2], x[..., 1::2]
        emb_sin, emb_cos = emb[..., ::2], emb[..., 1::2]
        x_rotated = torch.cat(
            [x1 * emb_cos - x2 * emb_sin, x1 * emb_sin + x2 * emb_cos], dim=-1
        )
        return x_rotated


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * x


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, window_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.window_size = window_size

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)

        self.rotary_embed = RotaryEmbedding(hidden_size)

    def forward(self, x):
        batch_size, sequence_length, hidden_size = x.size()
        pad = self.window_size // 2

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        rotery_Q = self.rotary_embed(Q)
        rotery_K = self.rotary_embed(K)

        padded_K = F.pad(rotery_K, (0, 0, pad, pad))
        padded_V = F.pad(V, (0, 0, pad, pad))

        K_windows = padded_K.unfold(dimension=1, size=self.window_size, step=1)
        V_windows = padded_V.unfold(dimension=1, size=self.window_size, step=1)

        K_windows = K_windows.permute(0, 1, 3, 2)
        V_windows = V_windows.permute(0, 1, 3, 2)

        attn_scores = torch.einsum("bth,btwh->btw", rotery_Q, K_windows) / (
            hidden_size**0.5
        )
        attn_weights = F.softmax(attn_scores, dim=-1)

        output = torch.einsum("btw,btwh->bth", attn_weights, V_windows)

        return self.output(output)


class MoEGenreGate(nn.Module):
    def __init__(self, hidden_size, genre_emb_dim, num_experts, moe_dim, top_k):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts

        self.rms_norm = RMSNorm(hidden_size)
        self.word_gate = nn.Linear(hidden_size, num_experts)
        self.genre_gate = nn.Linear(genre_emb_dim, num_experts)

        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, moe_dim),
                    nn.ReLU(),
                    nn.Linear(moe_dim, moe_dim),
                    nn.ReLU(),
                    nn.Linear(moe_dim, hidden_size),
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, x, genre_embed):
        rms_norm_x = self.rms_norm(x)
        word_gate = self.word_gate(rms_norm_x)
        genre_gate = self.genre_gate(genre_embed)

        gate = word_gate + genre_gate
        gate_weights = F.softmax(gate, dim=-1)

        topk_weights, topk_indices = torch.topk(gate_weights, self.top_k, dim=-1)

        batch_size, sequence_length, hidden_size = x.shape
        output = torch.zeros_like(x)

        x_flat = x.view(batch_size * sequence_length, hidden_size)

        for k in range(self.top_k):
            expert_ids = topk_indices[:, :, k].reshape(-1)
            expert_w = topk_weights[:, :, k].reshape(-1)

            for expert_i in range(self.num_experts):
                mask = expert_ids == expert_i
                if mask.sum() == 0:
                    continue
                selected_x = x_flat[mask]
                expert_out = self.experts[expert_i](selected_x)

                weighted_out = expert_out * expert_w[mask].unsqueeze(-1)
                output.view(batch_size * sequence_length, hidden_size)[
                    mask
                ] += weighted_out

        return output


class DecoderBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        genre_emb_dim,
        window_size,
        num_experts,
        moe_dim,
        top_k,
    ):
        super().__init__()

        self.self_attention = SelfAttention(
            hidden_size=hidden_size,
            window_size=window_size,
        )

        self.moe_genre_gate = MoEGenreGate(
            hidden_size=hidden_size,
            genre_emb_dim=genre_emb_dim,
            num_experts=num_experts,
            moe_dim=moe_dim,
            top_k=top_k,
        )

    def forward(self, x, genre_emb):
        residual_x = x
        attention_x = self.self_attention(x)

        y = residual_x + attention_x

        residual_y = y
        moe_genre_gate_y = self.moe_genre_gate(y, genre_emb)

        return residual_y + moe_genre_gate_y


class MixtralGenreGateModel(nn.Module):
    def __init__(
        self,
        num_layers,
        vocab_size,
        hidden_size,
        moe_dim,
        genre_emb_dim,
        window_size,
        num_experts,
        top_k,
    ):
        super().__init__()

        self.word_embed = nn.Embedding(
            vocab_size,
            hidden_size,
        )

        self.genre_embed = nn.Embedding(
            1,
            genre_emb_dim,
        )

        self.decoders = nn.ModuleList(
            [
                DecoderBlock(
                    hidden_size=hidden_size,
                    genre_emb_dim=genre_emb_dim,
                    window_size=window_size,
                    num_experts=num_experts,
                    moe_dim=moe_dim,
                    top_k=top_k,
                )
                for _ in range(num_layers)
            ]
        )
        self.rms_norm = RMSNorm(
            hidden_size=hidden_size,
        )
        self.output = nn.Linear(
            hidden_size,
            vocab_size,
        )

    def forward(self, x, genre_id):
        embed_x = self.word_embed(x)

        genre_embed = self.genre_embed(genre_id)

        decoder_x = embed_x

        for decoder in self.decoders:
            decoder_x = decoder(
                decoder_x,
                genre_embed,
            )

        rms_norm_x = self.rms_norm(decoder_x)

        output_x = self.output(rms_norm_x)

        return output_x
