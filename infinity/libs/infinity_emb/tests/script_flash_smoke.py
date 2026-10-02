#!/usr/bin/env python
"""GPU smoke harness for the ``attn_implementation`` knob (workstream 3).

Standalone script — NEVER imported or collected by the unit tests (the name
does not start with ``test_``) and never run by CI/twins: the architect runs
it after integration on the GPU box. Scenarios (``--mode``, default ``all``):

* ``fallback``   : ``resolve_attn_implementation("flash_attention_2", bf16)``
                   on a box without ``flash-attn`` -> ``"sdpa"`` + warning
                   (this mode also runs without CUDA).
* ``parity``     : tiny Qwen3 bf16 forward, ``eager`` vs ``sdpa`` on random
                   ids (B=2, L=1024, ~10% padding); max |logit diff| < 1e-2.
* ``padded``     : B=16, L=8192, 10% padding (the prod OOM shape); an OOM is
                   caught and the requested-alloc size is printed.
* ``long_prompt``: B=1, L=65536, maskless (all-ones), bf16, ``sdpa`` — shows
                   the O(L) memory profile of the maskless path (skipped when
                   the GPU has < 12 GiB).

Usage::

    python script_flash_smoke.py [--mode parity|fallback|long_prompt|padded|all]
"""

from __future__ import annotations

import argparse
import gc
import re
import time

import torch
from transformers import Qwen3Config, Qwen3ForSequenceClassification  # type: ignore

from infinity_emb.transformer.attention import CHECK_FLASH_ATTN, resolve_attn_implementation

#: tiny Qwen3 recipe shared with the docs: hidden=256, layers=2, heads=8,
#: kv=4, head_dim=32, max_position_embeddings=131072, sliding_window=4096.
#: vocab_size/intermediate_size/num_labels are structural necessities.
TINY_QWEN3 = {
    "vocab_size": 1024,
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_hidden_layers": 2,
    "num_attention_heads": 8,
    "num_key_value_heads": 4,
    "head_dim": 32,
    "max_position_embeddings": 131072,
    "sliding_window": 4096,
    "num_labels": 2,
    "pad_token_id": 1,
}


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(f"== {title}")
    print("=" * 78, flush=True)


def build_tiny_qwen3(attn_implementation: str) -> Qwen3ForSequenceClassification:
    """Random tiny Qwen3 classifier in bf16 on cuda with a pinned attention impl.

    ``config._attn_implementation`` is set before construction so
    ``Qwen3ForSequenceClassification`` (from config, no downloads) honors it.
    ``max_window_layers=1`` makes the last layer slide, the first full —
    mirroring jina-reranker-v3.5's mixed sliding/full attention, i.e. the
    two-mask-set layout ([B,1,L,L] full + sliding) that OOMs the prod card.
    """
    torch.manual_seed(0)
    config = Qwen3Config(
        **TINY_QWEN3,
        use_sliding_window=True,
        max_window_layers=1,
        torch_dtype=torch.bfloat16,
    )
    config._attn_implementation = attn_implementation
    model = Qwen3ForSequenceClassification(config)
    model.eval()
    model = model.to(device="cuda", dtype=torch.bfloat16)
    return model


def padded_ids_and_mask(B: int, L: int, pad_frac: float = 0.1):
    """Random ids with padding at the END of each row (collation-style)."""
    torch.manual_seed(0)
    ids = torch.randint(1, TINY_QWEN3["vocab_size"], (B, L), device="cuda")
    attention_mask = torch.ones((B, L), dtype=torch.long, device="cuda")
    pad_cols = max(1, int(L * pad_frac))
    ids[:, -pad_cols:] = 0
    attention_mask[:, -pad_cols:] = 0
    return ids, attention_mask


def forward(model, ids, attention_mask):
    """One forward under inference_mode; returns (logits, elapsed_s, peak_gib)."""
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.inference_mode():
        logits = model(input_ids=ids, attention_mask=attention_mask).logits
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak_gib = torch.cuda.max_memory_allocated() / 1024**3
    return logits, elapsed, peak_gib


def free_memory(*objects) -> None:
    for obj in objects:
        del obj
    gc.collect()
    torch.cuda.empty_cache()


def mode_fallback() -> None:
    section("mode=fallback  flash_attention_2 requested, flash-attn installed?  (CPU-safe)")
    installed = CHECK_FLASH_ATTN.is_available
    print(f"flash_attn installed (CHECK_FLASH_ATTN.is_available): {installed}")
    resolved = resolve_attn_implementation("flash_attention_2", torch.bfloat16)
    print(f"resolve_attn_implementation('flash_attention_2', torch.bfloat16) -> {resolved!r}")
    if installed:
        print("flash-attn present: 'flash_attention_2' passed through (FA2 kernels used).")
    else:
        print(
            "flash-attn missing: expected 'sdpa' fallback + the warning logged above "
            "(self-explanatory on the dev box)."
        )


def mode_parity() -> None:
    section("mode=parity  eager vs sdpa (bf16, B=2, L=1024, ~10% padding)")
    ids, mask = padded_ids_and_mask(B=2, L=1024)

    eager = build_tiny_qwen3("eager")
    assert eager.config._attn_implementation == "eager"
    logits_eager, time_eager, peak_eager = forward(eager, ids, mask)
    print(f"eager: peak {peak_eager:.3f} GiB   forward {time_eager * 1000:.1f} ms")

    sdpa = build_tiny_qwen3("sdpa")
    assert sdpa.config._attn_implementation == "sdpa"
    logits_sdpa, time_sdpa, peak_sdpa = forward(sdpa, ids, mask)
    print(f"sdpa : peak {peak_sdpa:.3f} GiB   forward {time_sdpa * 1000:.1f} ms")

    max_diff = (logits_eager - logits_sdpa).abs().max().item()
    print(f"max |logit(eager) - logit(sdpa)| = {max_diff:.3e}  (must be < 1e-2)")
    assert max_diff < 1e-2, f"eager/sdpa parity broken, max diff {max_diff:.3e}"

    free_memory(eager, sdpa, logits_eager, logits_sdpa, ids, mask)


def mode_padded() -> None:
    section("mode=padded  B=16 L=8192, 10% padding, bf16, sdpa (prod OOM shape)")
    model = build_tiny_qwen3("sdpa")
    ids, mask = padded_ids_and_mask(B=16, L=8192)
    try:
        logits, elapsed, peak = forward(model, ids, mask)
        print(f"forward {elapsed:.2f} s, peak allocation {peak:.3f} GiB (no OOM)")
        dense_bool = 2 * 8192**2  # 2 x [B,1,L,L] bool masks (full + sliding sets)
        print(
            f"the dense-mask pair for this shape alone is {dense_bool / 1024**3:.2f} GiB "
            "(bool) plus a same-size conversion in the model dtype by the sdpa path "
            "(x2 for float32) — this is the allocation that OOMs the 24 GiB prod "
            "card on listwise 32k-char docs."
        )
        free_memory(model, logits, ids, mask)
    except torch.cuda.OutOfMemoryError as exc:
        text = str(exc)
        match = re.search(r"Tried to allocate ([\d.]+)\s*(GiB|MiB)", text)
        if match:
            size, unit = match.groups()
            print(f"OOM as expected: tried to allocate {size} {unit}")
        else:
            print(f"OOM as expected: {text}")
        free_memory(model, ids, mask)


def mode_long_prompt() -> None:
    props = torch.cuda.get_device_properties(0)
    if props.total_memory < 12 * 1024**3:
        print(
            "SKIP long_prompt: device has "
            f"{props.total_memory / 1024**3:.1f} GiB (< 12 GiB guard for the L=65536 run)."
        )
        return
    section("mode=long_prompt  B=1 L=65536 maskless (all-ones), bf16, sdpa")
    model = build_tiny_qwen3("sdpa")
    ids = torch.randint(1, TINY_QWEN3["vocab_size"], (1, 65536), device="cuda")
    ones = torch.ones((1, 65536), dtype=torch.long, device="cuda")
    logits, elapsed, peak = forward(model, ids, ones)
    print(f"forward {elapsed:.2f} s, peak allocation {peak:.3f} GiB")
    dense_bool = 2 * 65536**2  # 2 x [B,1,L,L] bool masks (full + sliding sets)
    print(
        "peak is dominated by the SLIDING layer's dense [B,1,L,L] mask:",
        "with max_window_layers=1 the sliding layer cannot skip mask creation"
        " (kv_length 65536 > sliding_window 4096), so it materializes/expands"
        f" ~{dense_bool / 1024**3:.1f} GiB of [1,1,65536,65536] tensors even with"
        " a maskless input. FA2 computes windowed attention in-kernel (no dense"
        " mask), which is the point of this refactor. Full-attention (non-sliding)"\
        " layers stay O(L)."
    )
    free_memory(model, logits, ids, ones)


MODES = {
    "fallback": mode_fallback,
    "parity": mode_parity,
    "padded": mode_padded,
    "long_prompt": mode_long_prompt,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GPU smoke harness for the attention-implementation knob.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["all", *MODES.keys()],
        default="all",
        help="which scenario to run ('fallback' also runs without CUDA).",
    )
    args = parser.parse_args()

    print(
        "script_flash_smoke: "
        f"torch {torch.__version__} | cuda {torch.cuda.is_available()} | "
        + (
            f"{torch.cuda.get_device_name(0)} "
            f"({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB, "
            f"cc {torch.cuda.get_device_properties(0).major}."
            f"{torch.cuda.get_device_properties(0).minor})"
            if torch.cuda.is_available()
            else "no CUDA device"
        )
    )

    selected = list(MODES) if args.mode == "all" else [args.mode]
    try:
        for name in selected:
            if name != "fallback" and not torch.cuda.is_available():
                print(f"SKIP mode={name}: torch.cuda.is_available() is False (GPU harness).")
                continue
            try:
                MODES[name]()
            except torch.cuda.OutOfMemoryError as exc:  # outer guard (e.g. parity on a small card)
                print(f"mode={name} hit torch.cuda.OutOfMemoryError: {exc}")
            finally:
                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.empty_cache()
    except KeyboardInterrupt:
        print("\ninterrupted; releasing CUDA memory...")
    finally:
        gc.collect()
        torch.cuda.empty_cache()
    print("\nall requested modes finished.")


if __name__ == "__main__":
    main()
