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
from ml.etl.ev.common import write_outputs


def _default_pot_by_ctx(ctx: str) -> float:
    c = (ctx or "").upper()
    if c == "LIMPED_SINGLE": return 3.0
    if c == "VS_OPEN":       return 7.5
    if c == "VS_3BET":       return 22.5
    if c == "VS_4BET":       return 50.0
    return 7.5

def _import_symbol(path: str):
    """Import 'pkg.mod:SYM' → object."""
    mod, sym = path.split(":")
    return getattr(importlib.import_module(mod), sym)

def _resolve_facing_tokens(cfg: Dict[str, Any]) -> List[str]:
    """Resolve FACING action vocab from cfg; prefer model.action_vocab_import."""
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
    # fallback to canonical module
    try:
        from ml.models.vocab_actions import FACING_ACTION_VOCAB
        return list(FACING_ACTION_VOCAB)
    except Exception as e:
        raise ValueError("Cannot resolve FACING action vocab; set model.action_vocab_import or model.action_vocab") from e

def _action_seq_for_ctx(ctx: str) -> List[str]:
    c = (ctx or "").upper()
    if c == "VS_OPEN":       return ["RAISE", "CALL", ""]
    if c == "VS_3BET":       return ["RAISE", "3BET", "CALL"]
    if c == "VS_4BET":       return ["RAISE", "3BET", "4BET"]
    if c == "LIMPED_SINGLE": return ["LIMP", "CHECK", ""]
    return ["", "", ""]


# --- BUILDER PATCH: build_ev_postflop_facing (full function) ---
def build_ev_postflop_facing(cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Build EV parquet for POSTFLOP FACING (IP responds to a bet).
    - Uses pairs from build.pairs_ip_oop (required).
    - hero_pos is the IP seat (e.g., 'BTN'); ip_pos/oop_pos are role tokens.
    - faced_bb = size_frac * pot_bb (not stack).
    - Emits illegal_mask_facing (1=illegal, 0=legal) based on stack cap for raises.
    """
    # ---- Config ----
    dataset = cfg.get("dataset", {}) or {}
    build    = cfg.get("build", {}) or {}
    compute  = cfg.get("compute", {}) or {}
    stake    = cfg.get("stake", {}) or {}

    # ---- Repro ----
    seed = int(dataset.get("seed", 42))
    random.seed(seed); np.random.seed(seed)

    # ---- Facing vocab ----
    ACTION_TOKENS: List[str] = _resolve_facing_tokens(cfg)

    # ---- Knobs ----
    stacks_cfg = build.get("stacks_bb", [25, 40, 60, 80, 100, 150])
    stacks_bb: List[float] = sorted({float(s) for s in stacks_cfg if float(s) > 0.0})
    if not stacks_bb:
        raise ValueError("facing: build.stacks_bb is empty/invalid")

    size_fracs_cfg = build.get("size_fracs", [0.33, 0.66])
    # keep in (0,1]; round for stable strata
    size_fracs: List[float] = sorted({round(float(x), 2) for x in size_fracs_cfg if 0.0 < float(x) <= 1.0})
    if not size_fracs:
        raise ValueError("facing: build.size_fracs is empty/invalid (expect fractions of pot)")

    samples_per_board: int = max(1, int(build.get("samples_per_board", build.get("samples_per_combination", 64))))
    ctxs: List[str] = [str(x).upper() for x in
                       (build.get("contexts") or ["VS_OPEN", "VS_3BET", "VS_4BET", "LIMPED_SINGLE"])]
    pot_overrides: Dict[str, float] = {str(k).upper(): float(v) for k, v in (build.get("pot_by_ctx", {}) or {}).items()}

    # ---- Pairs (required) ----
    pairs_cfg: Dict[str, List[Tuple[str, str]]] = {
        str(k).upper(): [(str(ip).upper(), str(oop).upper()) for ip, oop in v]
        for k, v in (build.get("pairs_ip_oop") or {}).items()
    }

    def pairs_for_ctx(ctx: str) -> List[Tuple[str, str]]:
        pairs = pairs_cfg.get(ctx, [])
        if not pairs:
            raise ValueError(f"facing: no pairs provided for ctx='{ctx}' in build.pairs_ip_oop")
        return pairs

    # Global coverage guard
    if not any(pairs_cfg.get(c, []) for c in ctxs):
        raise ValueError("facing: build.pairs_ip_oop has no pairs for all contexts; please configure pairs")

    # ---- Stake artifacts ----
    stakes_token = str(stake.get("token", "NL10")).upper().replace(" ", "")
    stakes_id = str(stake.get("stakes_id", "2"))
    ranges_parquet = stake.get("ranges_parquet") or f"data/datasets/rangenet_preflop_from_flop_{stakes_token}.parquet"
    # ---- Engines ----
    vrp       = VillainRangeProvider(ranges_parquet)
    clusterer = load_board_clusterer(cfg)
    evmc      = EVMC(samples=int(compute.get("facing_samples", 5000)), seed=seed)

    # ---- Helper: illegal mask for facing (depends on faced size & stack) ----
    def _illegal_mask_facing(action_tokens: List[str], faced_bb: float, stack_bb: float) -> List[int]:
        mask: List[int] = []
        for tok in action_tokens:
            T = tok.upper()

            # CHECK is illegal when we are facing a bet
            if T == "CHECK":
                mask.append(1)
                continue

            if T in ("FOLD", "CALL"):
                mask.append(0)
                continue

            if T == "ALLIN":
                mask.append(0 if stack_bb > 0 else 1)
                continue

            if T.startswith("RAISE_"):
                try:
                    mult = int(T.split("_", 1)[1]) / 100.0  # e.g., 150 -> 1.5x (your convention)
                except Exception:
                    mask.append(1)
                    continue
                raise_to_bb = faced_bb * mult
                mask.append(1 if raise_to_bb > stack_bb + 1e-9 else 0)
                continue

            # Unknown tokens → illegal
            mask.append(1)

        return mask

    # ---- Progress ----
    total_iters = sum(len(pairs_for_ctx(ctx)) * len(stacks_bb) * len(size_fracs) * samples_per_board for ctx in ctxs)

    rows: List[Dict[str, Any]] = []
    with tqdm(total=total_iters, desc=f"Building EV (postflop:facing)[{stakes_token}]", unit="row", ncols=100) as pbar:
        for ctx in ctxs:
            pairs = pairs_for_ctx(ctx)
            base_pot = float(pot_overrides.get(ctx, _default_pot_by_ctx(ctx)))
            seq = _action_seq_for_ctx(ctx)

            for (ip_seat, oop_seat) in pairs:
                hero_pos_seat = ip_seat
                ip_pos_role, oop_pos_role = "IP", "OOP"

                for stack in stacks_bb:
                    for frac in size_fracs:
                        faced_bb = float(frac) * float(base_pot)  # bet is fraction of pot
                        for _ in range(samples_per_board):
                            # --- Sample hero & flop; avoid collisions
                            hero  = sample_random_hand_excluding(exclude=[])
                            board = sample_random_flop_excluding(exclude=[hero[:2], hero[2:]])
                            hand_id = hand169_id_from_hand_code(hero)
                            if hand_id is None:
                                pbar.update(1); continue

                            bmask = make_board_mask_52(board)
                            try:
                                if hasattr(clusterer, "predict_one"):
                                    cluster_id = int(clusterer.predict_one(board))
                                else:
                                    cluster_id = int(clusterer.predict([board])[0])
                            except Exception:
                                cluster_id = 0

                            # --- Villain preflop range vec (169-d) for this context
                            vvec = vrp.get_vector(
                                hero_pos=ip_seat,
                                villain_pos=oop_seat,
                                stack=stack,
                                action_seq=seq,
                            )

                            # --- EVs aligned to ACTION_TOKENS order
                            evs = evmc.compute_ev_vector(
                                ACTION_TOKENS,
                                hero_hand=hero,
                                board=board,
                                stack_bb=stack,
                                pot_bb=base_pot,
                                faced_size_bb=faced_bb,
                                villain_vec=vvec,
                            )

                            # --- Per-row illegal mask (exceeding stack raises)
                            il_mask = _illegal_mask_facing(ACTION_TOKENS, faced_bb=faced_bb, stack_bb=stack)

                            # --- Row
                            rows.append({
                                # Categoricals (match dataset.x_cols)
                                "hero_pos": hero_pos_seat,   # seat
                                "ip_pos": ip_pos_role,       # role
                                "oop_pos": oop_pos_role,     # role
                                "ctx": ctx,
                                "street": 1,
                                "board_cluster_id": int(cluster_id),
                                "stakes_id": stakes_id,
                                "hand_id": int(hand_id),

                                # Continuous (match dataset.cont_cols)
                                "board_mask_52": bmask,
                                "pot_bb": float(base_pot),
                                "stack_bb": float(stack),
                                "size_frac": float(frac),

                                # Targets aligned to vocab
                                **{f"ev_{tok}": float(val) for tok, val in zip(ACTION_TOKENS, evs)},

                                # Per-row controls
                                "illegal_mask_facing": il_mask,
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
    ap.add_argument(
        "--config",
        type=str,
        default="ml/config/evnet/postflop_facing/base.yaml",
        help="Path to YAML (recommended). Also accepts 'model/variant/profile' triple.",
    )
    args = ap.parse_args()

    # Resolve config
    cfg_arg = args.config.strip()
    if cfg_arg.endswith(".yaml"):
        cfg = load_model_config(path=cfg_arg)
    else:
        # supports: ev/postflop_facing/base
        parts = cfg_arg.split("/")
        if len(parts) == 3:
            model, variant, profile = parts
            cfg = load_model_config(model=model, variant=variant, profile=profile)
        elif len(parts) == 2:
            model, variant = parts
            cfg = load_model_config(model=model, variant=variant, profile="base")
        else:
            cfg = load_model_config(path=cfg_arg)  # last-ditch

    df = build_ev_postflop_facing(cfg)

    # Persist using your standard helper
    write_outputs(
        df,
        cfg,
        manifest_key="paths.manifest_path",
        parquet_key="paths.parquet_path",
    )
    print(f"✅ Wrote {len(df):,} rows to {cfg.get('paths',{}).get('parquet_path','<unknown>')}")


if __name__ == "__main__":
    main()