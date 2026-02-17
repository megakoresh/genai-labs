from dataclasses import dataclass
import torch


@dataclass
class GPTConfig:
    context_length: int
    vocab_size: int
    emb_dim: int
    n_attn_heads: int
    n_layers: int
    drop_rate: float
    qkv_bias: bool
    hf_repo_id: str

    _device: torch.device | None = None

    @property
    def device(self):
        if self._device == None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self._device


@dataclass
class OpenAIModelConfigs:
    gpt2_small_124m = GPTConfig(
        emb_dim=768,
        n_layers=12,
        n_attn_heads=12,
        context_length=1024,
        vocab_size=50257,
        drop_rate=0.1,
        qkv_bias=True,
        hf_repo_id="openai-community/gpt2",
    )
    gpt2_med_255m = GPTConfig(
        emb_dim=1024,
        n_layers=24,
        n_attn_heads=16,
        context_length=1024,
        vocab_size=50257,
        drop_rate=0.1,
        qkv_bias=True,
        hf_repo_id="openai-community/gpt2-medium",
    )
    gpt2_lg_755m = GPTConfig(
        emb_dim=1280,
        n_layers=36,
        n_attn_heads=20,
        context_length=1024,
        vocab_size=50257,
        drop_rate=0.1,
        qkv_bias=True,
        hf_repo_id="openai-community/gpt2-large",
    )
    gpt2_xlg_1558m = GPTConfig(
        emb_dim=1600,
        n_layers=48,
        n_attn_heads=25,
        context_length=1024,
        vocab_size=50257,
        drop_rate=0.1,
        qkv_bias=True,
        hf_repo_id="openai-community/gpt2-xl",
    )
