"""M8: rebuild the M7 models and export them to MLX.

Mlx-free at module level (imports on the Linux pods). The `.pt` files of M7 hold adapter state (and, for KV-pruned
models, the group list and head means), not merged models. `rebuild` replays the M7 test path on a fresh fp32 CPU
model: excise, (load the k-heal, merge, fold KV groups), load the final adapter, merge. `export_mlx` writes
config.json + one model.safetensors; when a layer is KV-pruned or carries an o_proj bias the config also names the
model file `qwen3_pruned.py` (copied beside the weights) so stock mlx-lm can load it with trust_remote_code.
"""
from __future__ import annotations

import json
import math
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import torch

MODEL = "Qwen/Qwen3-0.6B"
SHORT = MODEL.split("/")[-1]
N_LAYERS = 28
M5 = Path("var/results/m5")
M7 = Path("var/results/m7")
BASELINE = Path("var/results/baseline")
CKPT = Path("var/ckpt/m7")
MODEL_FILE = "qwen3_pruned.py"
MODEL_FILE_SRC = Path(__file__).resolve().parent / "mlx_models" / MODEL_FILE
QV = ["q_proj", "v_proj"]
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}


# ---------------------------------------------------------------- registry

def _cfg(lr, steps=300, rank=10, modules=QV) -> dict:
    return {"lr": lr, "steps": steps, "rank": rank, "modules": list(modules)}


@dataclass(frozen=True)
class Spec:
    name: str
    variant: str | None  # None: unmodified, both variants
    k: int = 0
    base: str | None = None  # k-heal checkpoint, loaded and merged before the KV fold (A models only)
    final: str | None = None
    cfg: dict = field(default_factory=dict)
    m7: tuple | None = None  # where M7's selection accuracy sits in m7_{variant}.json


def _spec(name, variant, k, final, cfg, base=None, m7=None) -> Spec:
    return Spec(name, variant, k, base, final, cfg, m7)


def _heal(v, k, lr) -> str:
    return f"{v}_k{k}_tune_lr{lr:g}_st300_r10_qv.pt"


# LoRA 1m (adapter_sweep.LORA["1m"]): all 28 blocks, q/v, rank 7. The state file only holds {"state": ...}.
_LORA_CFG = _cfg(3e-4, rank=7)
MODELS: dict[str, Spec] = {s.name: s for s in [
    Spec("unmodified", None),
    _spec("rule-A", "rule", 12, "rule_k12_g48_lr0.001_st300_r10_qv_s0.pt", _cfg(1e-3), _heal("rule", 12, 1e-3),
          ("runs", "k12|g48|s0")),
    _spec("rule-B", "rule", 18, _heal("rule", 18, 3e-3), _cfg(3e-3), None, ("runs", "k18|g0|s0")),
    _spec("sensible-A", "sensible", 12, "sensible_k12_g64_lr0.001_st300_r10_qv_s0.pt", _cfg(1e-3),
          _heal("sensible", 12, 1e-3), ("runs", "k12|g64|s0")),
    _spec("rule-lora-1m", "rule", 0, "rule_recheck_lora_1m_lr0.0003_st300.pt", _LORA_CFG, None,
          ("recheck", "lora|1m")),
    _spec("sensible-lora-1m", "sensible", 0, "sensible_recheck_lora_1m_lr0.0003_st300.pt", _LORA_CFG, None,
          ("recheck", "lora|1m")),
    _spec("rule-k12", "rule", 12, _heal("rule", 12, 1e-3), _cfg(1e-3), None, ("runs", "k12|g0|s0")),
    _spec("sensible-k12", "sensible", 12, _heal("sensible", 12, 1e-3), _cfg(1e-3), None, ("runs", "k12|g0|s0")),
]}


def variants_of(name: str) -> list[str]:
    v = MODELS[name].variant
    return ["rule", "sensible"] if v is None else [v]


def alive_for(variant: str, k: int, m5: Path = M5) -> list[int]:
    """Surviving blocks (original indices) after the first k removals of the M5 excision path."""
    order = json.loads((Path(m5) / f"excise_{SHORT}_{variant}.json").read_text())["removed_order"]
    if len(order) < k:
        raise ValueError(f"M5 excision path for {variant} has {len(order)} blocks, need {k}")
    gone = set(order[:k])
    return [i for i in range(N_LAYERS) if i not in gone]


# ---------------------------------------------------------------- rebuild

def _load_strict(handle, state: dict, what: str) -> None:
    """`load_adapter_state` is strict=False: demand exact key and shape agreement first."""
    from surgery import adapters
    want = adapters.adapter_state(handle.modules)
    missing, extra = sorted(set(want) - set(state)), sorted(set(state) - set(want))
    bad = [k for k in want if k in state and tuple(state[k].shape) != tuple(want[k].shape)]
    if missing or extra or bad:
        raise ValueError(f"{what}: adapter state does not match the model "
                         f"(missing {missing[:3]}, unexpected {extra[:3]}, wrong shape {bad[:3]})")
    adapters.load_adapter_state(handle.modules, state)


def rebuild(model, alive, cfg, final_state, base_state=None, groups=(), means=None):
    """The M7 test path on a fresh fp32 CPU model (mutated and returned). `groups` are [(layer, group)] in the
    excised numbering; `means` is (n_layers, n_heads, head_dim) as saved with the checkpoint."""
    from surgery import adapters, structure
    alive, groups = list(alive), [tuple(g) for g in groups]
    structure.excise(model, alive)
    n = len(alive)
    if groups:
        if base_state is None or means is None:
            raise ValueError("a KV-pruned point needs base_state and means")
        h = adapters.add_lora(model, range(n), cfg["modules"], cfg["rank"])
        _load_strict(h, base_state, "base heal")
        adapters.merge_lora(h)
        structure.fold_kv_groups(model, means, groups)
    h = adapters.add_lora(model, range(n), cfg["modules"], cfg["rank"])
    _load_strict(h, final_state, "final adapter")
    adapters.merge_lora(h)
    return model


def load_point(name: str, variant: str | None = None, *, ckpt_dir: Path = CKPT, m5: Path = M5, pristine=None):
    """Rebuilt merged fp32 CPU model for one registry entry. `pristine` is a fresh unmodified model (default:
    loaded from the HF snapshot); it is mutated. `unmodified` needs no variant and no checkpoint."""
    spec = MODELS[name]
    if pristine is None:
        from surgery.eval import load_model
        pristine, _ = load_model(MODEL, device="cpu")
    if spec.final is None:
        return pristine
    alive = alive_for(spec.variant, spec.k, m5) if spec.k else list(range(N_LAYERS))
    final = torch.load(Path(ckpt_dir) / spec.final, map_location="cpu")
    groups, means, base_state = (), None, None
    if spec.base is not None:
        groups, means = final["groups"], final["means"]
        base_state = torch.load(Path(ckpt_dir) / spec.base, map_location="cpu")["state"]
    return rebuild(pristine, alive, spec.cfg, final["state"], base_state, groups, means)


# ---------------------------------------------------------------- export

_BLOCK = re.compile(r"(^|\.)layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|down|up)_proj)\.weight$")


def block_param_count(weights: dict) -> int:
    """Sum of the q/k/v/o and gate/down/up `.weight` sizes (shapes in, as `structure.param_count` counts them)."""
    return sum(math.prod(shape) for key, shape in weights.items() if _BLOCK.search(key))


_CONFIG_KEYS = ["model_type", "hidden_size", "num_hidden_layers", "intermediate_size", "num_attention_heads",
                "num_key_value_heads", "head_dim", "rms_norm_eps", "vocab_size", "max_position_embeddings",
                "tie_word_embeddings", "rope_parameters", "layer_types", "hidden_act", "attention_bias",
                "bos_token_id", "eos_token_id"]


def mlx_config(model, dtype: str) -> tuple[dict, bool]:
    """(config.json content, needs_model_file). Layer count and layer_types are the excised ones."""
    cfg = model.config
    full = cfg.to_dict()
    out = {k: full[k] for k in _CONFIG_KEYS if k in full}
    out["architectures"] = ["Qwen3ForCausalLM"]
    out["head_dim"] = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    out["num_hidden_layers"] = len(model.model.layers)
    out["dtype"] = {"fp32": "float32", "bf16": "bfloat16"}[dtype]
    hd, nkv = out["head_dim"], cfg.num_key_value_heads
    kv = [l.self_attn.k_proj.out_features // hd for l in model.model.layers]
    bias = [l.self_attn.o_proj.bias is not None for l in model.model.layers]
    needs = any(k != nkv for k in kv) or any(bias)
    if needs:
        out.update(model_file=MODEL_FILE, kv_heads_per_layer=kv, o_proj_bias=bias)
    return out, needs


def export_weights(model, dtype: str) -> dict:
    """Name -> cloned contiguous tensor in `dtype`; the tied lm_head is dropped."""
    tied = bool(model.config.tie_word_embeddings)
    dt = DTYPES[dtype]
    return {k: v.detach().to(dt).clone().contiguous() for k, v in model.state_dict().items()
            if not (tied and k == "lm_head.weight")}


def export_mlx(model, out_dir, dtype: str, tokenizer=None) -> dict:
    """Write config.json, model.safetensors (and the tokenizer, and the model file when needed) to `out_dir`,
    atomically (temp dir, then rename). Returns the manifest."""
    from safetensors.torch import save_file
    if dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {sorted(DTYPES)}")
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(out_dir)
    config, needs = mlx_config(model, dtype)
    weights = export_weights(model, dtype)
    tmp = out_dir.with_name(out_dir.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        (tmp / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        save_file(weights, str(tmp / "model.safetensors"), metadata={"format": "pt"})
        if needs:
            shutil.copyfile(MODEL_FILE_SRC, tmp / MODEL_FILE)
        if tokenizer is not None:
            tokenizer.save_pretrained(tmp)
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        tmp.rename(out_dir)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    shapes = {k: tuple(v.shape) for k, v in weights.items()}
    return {"dtype": dtype, "needs_model_file": needs, "n_layers": config["num_hidden_layers"],
            "weight_bytes": (out_dir / "model.safetensors").stat().st_size,
            "n_params": sum(math.prod(s) for s in shapes.values()), "block_params": block_param_count(shapes),
            "kv_heads_per_layer": config.get("kv_heads_per_layer"), "path": str(out_dir)}


# ---------------------------------------------------------------- parity

def parity_gate(pt_pred, mlx_pred, mlx_acc: float, m7_acc: float) -> dict:
    """Protocol gate: predictions agree on >= 99% of items and accuracy is within 0.5 points of M7's."""
    if len(pt_pred) != len(mlx_pred) or not len(pt_pred):
        raise ValueError("prediction lists must be the same non-zero length")
    agree = sum(int(a) == int(b) for a, b in zip(pt_pred, mlx_pred)) / len(pt_pred)
    delta = float(mlx_acc) - float(m7_acc)
    ok = agree >= 0.99 - 1e-12 and abs(delta) <= 0.005 + 1e-12
    return {"agreement": agree, "acc_delta": delta, "passed": bool(ok)}


def m7_accuracy(name: str, variant: str, m7_dir: Path = M7, baseline_dir: Path = BASELINE) -> float:
    """M7's selection accuracy for a registry model (unmodified: the M2 baseline pass)."""
    spec = MODELS[name]
    if spec.m7 is None:
        d = json.loads((Path(baseline_dir) / f"{SHORT}_{variant}_s0_selection.json").read_text())
        return d["metrics"]["accuracy"]["point"]
    d = json.loads((Path(m7_dir) / f"m7_{variant}.json").read_text())
    kind, key = spec.m7
    row = d["runs"][key] if kind == "runs" else d["recheck"]["selection"][key]
    return row["selection"]["accuracy"]
