# file: ml/etl/ev/build_ev_postflop_facing.py
from __future__ import annotations

import importlib
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

from ml.etl.ev.mc import EVMC
from ml.etl.ev.ranges import VillainRangeProvider
from ml.etl.ev.sampling import sample_random_hand_excluding, sample_random_flop_excluding
from ml.features.boards import load_board_clusterer
from ml.utils.board_mask import make_board_mask_52
from ml.utils.config import load_model_config
from ml.features.hands import hand169_id_from_hand_code
from ml.etl.ev.common import  write_outputs

def _default_pot_by_ctx(ctx: str) -> float:
    c = (ctx or "").upper()
    if c == "LIMPED_SINGLE": return 3.0
    if c == "VS_OPEN":       return 7.5
    if c == "VS_3BET":       return 22.5
    if c == "VS_4BET":       return 50.0
    return 7.5


def _action_seq_for_ctx(ctx: str) -> List[str]:
    ctx = (ctx or "").upper()
    if ctx == "VS_OPEN":        return ["RAISE", "CALL", ""]
    if ctx == "VS_3BET":        return ["RAISE", "3BET", "CALL"]
    if ctx == "VS_4BET":        return ["RAISE", "3BET", "4BET"]
    if ctx == "LIMPED_SINGLE":  return ["LIMP", "CHECK", ""]
    return ["", "", ""]


def _import_symbol(path: str):
    """Import 'pkg.mod:SYM' → object."""
    mod, sym = path.split(":")
    return getattr(importlib.import_module(mod), sym)

def _resolve_root_tokens(cfg: Dict[str, Any]) -> List[str]:
    """Resolve ROOT action vocab from cfg; prefer action_vocab_import."""
    m = cfg.get("model", {}) or {}
    if m.get("action_vocab_import"):
        toks = list(_import_symbol(m["action_vocab_import"]))
        if not toks:
            raise ValueError("action_vocab_import resolved to empty list")
        return toks
    if m.get("action_vocab"):
        toks = [str(x) for x in m["action_vocab"]]
        if not toks:
            raise ValueError("model.action_vocab is empty")
        return toks
    # last resort: canonical module
    try:
        from ml.models.vocab_actions import ROOT_ACTION_VOCAB
        return list(ROOT_ACTION_VOCAB)
    except Exception as e:
        raise ValueError("Cannot resolve ROOT action vocab; set model.action_vocab_import or model.action_vocab") from e


def build_ev_postflop_root(cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Postflop ROOT EV parquet builder.
    - Pairs are taken from build.pairs_ip_oop[ctx] (required).
    - hero_pos is the actual OOP seat (not a role literal).
    - ip_pos/oop_pos are role tokens ("IP"/"OOP") to match schema.
    - Emits 'illegal_mask_root' (1=illegal, 0=legal) so trainer can zero loss.
    """
    dataset = cfg.get("dataset", {}) or {}
    build    = cfg.get("build", {}) or {}
    compute  = cfg.get("compute", {}) or {}
    stake    = cfg.get("stake", {}) or {}

    # Repro
    seed = int(dataset.get("seed", 42))
    random.seed(seed); np.random.seed(seed)

    # Canonical root vocab (CHECK/BET_%/DONK_33, etc.)
    ACTION_TOKENS: List[str] = _resolve_root_tokens(cfg)

    # Knobs (replace your current block with this)
    stacks_cfg = build.get("stacks_bb", [25, 40, 60, 80, 100, 150])
    stacks_bb: List[float] = sorted({float(s) for s in stacks_cfg if float(s) > 0.0})
    if not stacks_bb:
        raise ValueError("build.stacks_bb is empty/invalid")

    samples_per_board: int = int(build.get("samples_per_board", 8))
    ctxs: List[str] = [str(x).upper() for x in
                       build.get("contexts", ["VS_OPEN", "VS_3BET", "VS_4BET", "LIMPED_SINGLE"])]
    pot_overrides: Dict[str, float] = {str(k).upper(): float(v) for k, v in (build.get("pot_by_ctx", {}) or {}).items()}

    # Required pairs per ctx from config (no hidden fallback)
    cfg_pairs: Dict[str, List[Tuple[str, str]]] = {
        str(k).upper(): [(str(ip).upper(), str(oop).upper()) for ip, oop in v]
        for k, v in (build.get("pairs_ip_oop", {}) or {}).items()
    }

    def _pairs_for_ctx(ctx: str) -> List[Tuple[str, str]]:
        pairs = cfg_pairs.get(ctx, [])
        if not pairs:
            raise ValueError(f"build.pairs_ip_oop has no pairs for ctx='{ctx}'")
        return pairs

    # Coverage guard: at least one ctx must have pairs
    if not any(cfg_pairs.get(c, []) for c in ctxs):
        raise ValueError("postflop root: build.pairs_ip_oop is empty for all contexts")

    def _illegal_mask_root(action_tokens: List[str]) -> List[int]:
        mask: List[int] = []
        for tok in action_tokens:
            T = tok.upper()
            if T == "CHECK" or T.startswith("BET_"):
                mask.append(0)
            else:
                mask.append(1)  # DONK_* and any other stray tokens illegal at root
        return mask

    # Stake artifacts
    stakes_token = str(stake.get("token", "NL10")).upper().replace(" ", "")
    stakes_id = str(stake.get("stakes_id", "2"))
    ranges_parquet = stake.get("ranges_parquet") or f"data/datasets/rangenet_preflop_from_flop_{stakes_token}.parquet"

    # Engines / artifacts
    vrp       = VillainRangeProvider(ranges_parquet)
    clusterer = load_board_clusterer(cfg)
    evmc      = EVMC(samples=int(compute.get("root_samples", 5000)), seed=seed)

    # Progress sizing (fixed: pass cfg_pairs, ctx)
    total_pairs = sum(len(_pairs_for_ctx(c)) for c in ctxs)
    total_iters = total_pairs * max(1, len(stacks_bb)) * samples_per_board

    rows: List[Dict[str, Any]] = []
    with tqdm(total=total_iters, desc=f"Building EV (postflop:root)[{stakes_token}]", unit="row", ncols=100) as pbar:
        for ctx in ctxs:
            pairs_ip_oop = _pairs_for_ctx(ctx)
            action_seq   = _action_seq_for_ctx(ctx)
            base_pot     = float(pot_overrides.get(ctx, _default_pot_by_ctx(ctx)))
            il_mask      = _illegal_mask_root(ACTION_TOKENS)

            for (ip_seat, oop_seat) in pairs_ip_oop:
                hero_pos_seat = oop_seat          # actual seat playing first at root
                ip_pos_role, oop_pos_role = "IP", "OOP"

                for stack in stacks_bb:
                    size_frac = 0.0
                    faced_bb  = 0.0

                    for _ in range(samples_per_board):
                        hero  = sample_random_hand_excluding(exclude=[])
                        board = sample_random_flop_excluding(exclude=[hero[:2], hero[2:]])
                        hand_id = hand169_id_from_hand_code(hero)
                        if hand_id is None:
                            hand_id = -1

                        bmask = make_board_mask_52(board)
                        try:
                            if hasattr(clusterer, "predict_one"):
                                cluster_id = int(clusterer.predict_one(board))
                            else:
                                cluster_id = int(clusterer.predict([board])[0])
                        except Exception:
                            cluster_id = 0

                        vvec = vrp.get_vector(
                            hero_pos=ip_seat,      # preflop seat mapping for ranges
                            villain_pos=oop_seat,
                            stack=stack,
                            action_seq=action_seq,
                        )

                        evs = evmc.compute_ev_vector(
                            ACTION_TOKENS,
                            hero_hand=hero,
                            board=board,
                            stack_bb=stack,
                            pot_bb=base_pot,
                            faced_size_bb=faced_bb,
                            villain_vec=vvec,
                        )

                        rows.append({
                            # Categoricals (match dataset.x_cols)
                            "hero_pos": hero_pos_seat,            # seat, NOT "OOP"
                            "ip_pos": ip_pos_role,                # role token
                            "oop_pos": oop_pos_role,              # role token
                            "ctx": ctx,
                            "street": 1,
                            "board_cluster_id": int(cluster_id),
                            "stakes_id": stakes_id,
                            "hand_id": int(hand_id),

                            # Continuous (match dataset.cont_cols)
                            "board_mask_52": bmask,
                            "pot_bb": float(base_pot),
                            "stack_bb": float(stack),
                            "size_frac": float(size_frac),        # root: 0.0

                            # Targets
                            **{f"ev_{tok}": float(val) for tok, val in zip(ACTION_TOKENS, evs)},

                            # Per-row controls
                            "illegal_mask_root": il_mask,         # 1=illegal, 0=legal
                            "weight": int(evmc.samples),

                            # Audit
                            "board": board,
                            "hero_hand": hero,
                            "ip_seat": ip_seat,
                            "oop_seat": oop_seat,
                        })
                        pbar.update(1)

    df = pd.DataFrame(rows)

    write_outputs(df, cfg, parquet_key="paths.parquet_path")
    return df

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="ml/config/evnet/postflop_root/base.yaml",
                    help="Path to YAML, or 'ev/postflop_root/base' triple.")
    args = ap.parse_args()

    cfg_arg = args.config.strip()
    if cfg_arg.endswith(".yaml"):
        cfg = load_model_config(path=cfg_arg)
    else:
        parts = cfg_arg.split("/")
        if len(parts) == 3:
            model, variant, profile = parts
            cfg = load_model_config(model=model, variant=variant, profile=profile)
        elif len(parts) == 2:
            model, variant = parts
            cfg = load_model_config(model=model, variant=variant, profile="base")
        else:
            cfg = load_model_config(path=cfg_arg)

    df = build_ev_postflop_root(cfg)
    print(f"✅ ev_postflop_root → {cfg.get('paths',{}).get('parquet_path','<unknown>')}  rows={len(df):,}")


if __name__ == "__main__":
    main()