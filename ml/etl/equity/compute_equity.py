import math, time, random, os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple
import pandas as pd
import concurrent.futures as cf
import eval7

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

from ml.config.types_hands import SUITS
from ml.etl.utils.hand import RANKS, hand_id_to_combo
from ml.features.boards import load_board_clusterer  # returns RuleBasedBoardClusterer or KMeansBoardClusterer
from ml.utils.config import load_model_config
from ml.utils.board_mask import make_board_mask_52


try:
    from ml.features.boards.representatives import (
        discover_representative_flops,
        discover_representative_turns,
        discover_representative_rivers,
    )
    HAS_DISCOVERY = True
except Exception:
    HAS_DISCOVERY = False


ALL_CARDS: List[eval7.Card] = [eval7.Card(r + s) for r in RANKS for s in SUITS]

def _equity_triplet_vs_random_1op_exact_river(
    hero: Tuple[eval7.Card, eval7.Card],
    board5: List[eval7.Card],
) -> Tuple[int, int, int]:
    """
    Exact showdown enumeration on the river (5-card board).
    Iterates all legal villain hole-card combos from remaining deck.
    Returns (wins, ties, losses).
    """
    assert len(board5) == 5, "exact river requires a full 5-card board"
    used = {hero[0], hero[1], *board5}
    deck = [c for c in ALL_CARDS if c not in used]

    wins = ties = losses = 0
    h_val = eval7.evaluate([hero[0], hero[1], *board5])

    # enumerate all villain 2-card combos
    n = len(deck)
    for i in range(n):
        for j in range(i + 1, n):
            v1, v2 = deck[i], deck[j]
            v_val = eval7.evaluate([v1, v2, *board5])
            if h_val > v_val: wins += 1
            elif h_val == v_val: ties += 1
            else: losses += 1

    return wins, ties, losses

def _norm_cluster_id(x) -> Optional[int]:
    if x is None:
        return None
    try:
        if isinstance(x, float) and math.isnan(x):
            return None
        return int(x)
    except Exception:
        return None


class BoardSamplers:
    """
    Holds per-street (1/2/3) samplers: street -> {cluster_id -> [boards]}
    Each board is a list[eval7.Card] of the correct length for the street.
    Fallbacks to random sampling if cluster not found.
    """
    def __init__(self, samplers: Dict[int, Dict[int, List[List[eval7.Card]]]]):
        self.samplers = samplers  # {street: {cluster_id: [boards...] }}

    def sample(
        self,
        street: int,
        cluster_id: Optional[int],
        rng: random.Random,
        used: Optional[Set[eval7.Card]] = None,   # NEW: cards to avoid (hero, previous streets)
    ) -> List[eval7.Card]:
        if street == 0:
            return []

        used = used or set()

        if cluster_id is not None:
            by_cluster = self.samplers.get(street, {})
            pool = by_cluster.get(int(cluster_id)) or []
            if pool:
                # Try a few draws that don't intersect with `used`
                for _ in range(16):
                    cand = pool[rng.randrange(len(pool))]
                    if not (set(cand) & used):
                        return list(cand)  # copy
                # If all candidates intersect, fall back to random below

        # fallback (now also respects `used`)
        return _sample_board_random(street, used=used, rng=rng)


def _to_eval7_board_strs(raw: Iterable[str]) -> List[eval7.Card]:
    return [eval7.Card(s) for s in raw]


def _sample_board_random(street: int, used: Set[eval7.Card], rng: random.Random) -> List[eval7.Card]:
    """Random board consistent with street. 1=flop(3), 2=turn(4), 3=river(5)."""
    if street == 0:
        return []
    need = {1: 3, 2: 4, 3: 5}[street]
    deck = [c for c in ALL_CARDS if c not in used]
    rng.shuffle(deck)
    return deck[:need]


def _build_board_samplers_from_clusterer(
    cfg: Mapping[str, Any],
    n_clusters_limit: Optional[int],
    boards_per_cluster: int,
    seed: int,
) -> BoardSamplers:
    """
    Use your configured clusterer + (optional) representative discovery utilities
    to create per-street {cluster_id -> [boards]} pools.
    Falls back to empty mapping (→ random sampling) when discovery functions are missing.
    """
    rng = random.Random(seed)
    clusterer = load_board_clusterer(cfg)  # respects board_clustering.type/artifact/n_clusters

    samplers: Dict[int, Dict[int, List[List[eval7.Card]]]] = {}

    def _mk_pool(raw_boards: Dict[int, List[Sequence[str]]]) -> Dict[int, List[List[eval7.Card]]]:
        # raw_boards: cid -> list of ['9d','Ac','6h'] / length matches street
        out: Dict[int, List[List[eval7.Card]]] = {}
        for cid, boards in raw_boards.items():
            if n_clusters_limit is not None and int(cid) >= int(n_clusters_limit):
                continue
            # cap boards per cluster
            if boards_per_cluster > 0 and len(boards) > boards_per_cluster:
                boards = rng.sample(boards, boards_per_cluster)
            out[int(cid)] = [ _to_eval7_board_strs(b) for b in boards ]
        return out

    # Try discovery helpers if available
    if HAS_DISCOVERY:
        # discover flop sets
        try:
            raw_flops = discover_representative_flops(
                clusterer=clusterer,
                n_clusters_limit=n_clusters_limit,
                boards_per_cluster=boards_per_cluster,
                seed=seed,
                sample_pool=None,
            )
            samplers[1] = _mk_pool(raw_flops)
        except Exception:
            samplers[1] = {}

        # discover turns
        try:
            raw_turns = discover_representative_turns(
                clusterer=clusterer,
                n_clusters_limit=n_clusters_limit,
                boards_per_cluster=boards_per_cluster,
                seed=seed + 1,
                sample_pool=None,
            )
            samplers[2] = _mk_pool(raw_turns)
        except Exception:
            samplers[2] = {}

        # discover rivers
        try:
            raw_rivers = discover_representative_rivers(
                clusterer=clusterer,
                n_clusters_limit=n_clusters_limit,
                boards_per_cluster=boards_per_cluster,
                seed=seed + 2,
                sample_pool=None,
            )
            samplers[3] = _mk_pool(raw_rivers)
        except Exception:
            samplers[3] = {}
    else:
        # No discovery util available → empty maps (we’ll fall back to random)
        samplers[1] = {}
        samplers[2] = {}
        samplers[3] = {}

    return BoardSamplers(samplers)

def _equity_triplet_vs_random_1op(
    hero: Tuple[eval7.Card, eval7.Card],
    base_board: List[eval7.Card],   # can be [], or 3/4/5 cards
    n_sims: int,
    rng: random.Random,
) -> Tuple[int, int, int]:
    """
    Monte Carlo outcomes vs one random opponent given a fixed partial board.
    Fills the remaining community cards to river each sim.
    Returns counts: (wins, ties, losses).
    """
    assert len(hero) == 2
    assert len(base_board) in (0, 3, 4, 5)

    wins = ties = losses = 0
    used0 = set([hero[0], hero[1], *base_board])

    for _ in range(n_sims):
        # sample villain hand avoiding used
        deck0 = [c for c in ALL_CARDS if c not in used0]
        i = rng.randrange(len(deck0))
        j = rng.randrange(len(deck0) - 1)
        if j >= i: j += 1
        v1, v2 = deck0[i], deck0[j]

        need = 5 - len(base_board)
        deck = [c for c in deck0 if c not in (v1, v2)]
        rng.shuffle(deck)
        full_board = base_board if need == 0 else (base_board + deck[:need])

        h_val = eval7.evaluate([hero[0], hero[1], *full_board])
        v_val = eval7.evaluate([v1, v2, *full_board])

        if h_val > v_val: wins += 1
        elif h_val == v_val: ties += 1
        else: losses += 1

    return wins, ties, losses


def _row_worker(
    row: Tuple[int, int, int, int],   # (street, cluster_id or -1, hand_id, sims)
    seed: int,
    board_sampler_blob: Dict[str, Any],  # JSON-serializable sampler spec
    river_exact: bool = False,           # whether to use exact enumeration on river
) -> Tuple[int, Optional[int], int, int, float, float, float, float, str, List[float]]:
    """
    Returns:
      (street, cluster_id, hand_id, sims, p_win, p_tie, p_lose, weight, board_str, board_mask_52)
    """
    street, cl_raw, hand_id, sims = row
    cluster_id = None if cl_raw < 0 else int(cl_raw)

    # Deterministic RNG
    row_seed = (
        int(seed) * 1_000_003
        + int(street) * 10_007
        + int(hand_id) * 101
        + (cluster_id or 0)
    )
    rng = random.Random(row_seed)

    # Rehydrate board sampler
    sampler = _rehydrate_board_sampler(board_sampler_blob)

    # Hero and board
    hero = hand_id_to_combo(hand_id)  # -> (eval7.Card, eval7.Card)
    base_board = sampler.sample(street, cluster_id, rng) if street > 0 else []

    # ---- Choose MC or Exact (for river) ----
    if river_exact and street == 3 and len(base_board) == 5:
        w, t, l = _equity_triplet_vs_random_1op_exact_river(hero, base_board)
        sims_used = (len(ALL_CARDS) - 2 - 5)
        sims_used = sims_used * (sims_used - 1) // 2  # all possible villain combos
    else:
        w, t, l = _equity_triplet_vs_random_1op(hero, base_board, max(1, sims), rng)
        sims_used = max(1, sims)

    total = float(w + t + l) if (w + t + l) > 0 else 1.0
    p_win, p_tie, p_lose = w / total, t / total, l / total

    # Serialize board and compute 52-bit mask
    board_str = "".join(str(c) for c in base_board) if base_board else ""
    mask_52 = make_board_mask_52(board_str)

    return (
        street,
        cluster_id,
        hand_id,
        sims_used,
        p_win,
        p_tie,
        p_lose,
        float(sims_used),
        board_str,
        mask_52,
    )


def _rehydrate_board_sampler(blob: Dict[str, Any]) -> BoardSamplers:
    """
    Make a BoardSamplers instance from a JSON-safe blob (cards as strings).
    """
    samplers: Dict[int, Dict[int, List[List[eval7.Card]]]] = {}
    for street_str, by_cluster in blob.items():
        st = int(street_str)
        samplers[st] = {}
        for cid_str, boards in by_cluster.items():
            cid = int(cid_str)
            samplers[st][cid] = [
                [eval7.Card(s) for s in board_strs] for board_strs in boards
            ]
    return BoardSamplers(samplers)


def _serialize_board_samplers(samplers: BoardSamplers) -> Dict[str, Any]:
    """
    Convert BoardSamplers → JSON-safe dict so we can ship to worker processes.
    """
    out: Dict[str, Any] = {}
    for st, by_cluster in samplers.samplers.items():
        out[str(st)] = {}
        for cid, boards in by_cluster.items():
            out[str(st)][str(cid)] = [[str(c) for c in board] for board in boards]
    return out

def _safe_int(x, default: int = 0) -> int:
    try:
        if x is None:
            return default
        if isinstance(x, float) and (x != x):  # NaN check
            return default
        return int(x)
    except Exception:
        return default


def compute_equity_from_manifest_cfg(cfg: Mapping[str, Any]) -> Path:
    """
    Read manifest & config → write equity parquet.
    Honors:
      - compute.threads, chunk_size, log_every_n
      - compute.mc.min_samples / max_samples
      - compute.preflop_samples
      - board_clustering.* (type/artifact/n_clusters)
    """
    manifest_path = Path(cfg["paths"]["manifest_path"])
    out_parquet   = Path(cfg["paths"]["parquet_path"])

    comp = cfg.get("compute", {})
    threads    = comp.get("threads", "auto")
    chunk_size = int(comp.get("chunk_size", 8192))
    log_every  = int(comp.get("log_every_n", 50000))
    seed       = int(comp.get("seed", 42))
    mc_min     = int(comp.get("mc", {}).get("min_samples", 0))
    mc_max     = int(comp.get("mc", {}).get("max_samples", 0))
    preflop_sims = int(comp.get("preflop_samples", 20000))
    river_exact = bool(comp.get("river_exact", False))

    # Optional: limit clusters if your artifact has more than manifest expects; else None
    n_clusters_limit = cfg.get("build", {}).get("clusters", {})
    # Per-street limits are OK to pass None because discovery already takes explicit limits.
    # We’ll pass None here and let the representative discovery cap by what’s requested there.

    # Build cluster-aware samplers (or empty maps → random fallback)
    boards_per_cluster = int(cfg.get("build", {}).get("boards_per_cluster", 64))  # optional; default 64
    samplers = _build_board_samplers_from_clusterer(
        cfg=cfg,
        n_clusters_limit=None,
        boards_per_cluster=boards_per_cluster,
        seed=seed,
    )
    sampler_blob = _serialize_board_samplers(samplers)

    df = pd.read_parquet(str(manifest_path)).copy()

    # Normalize inputs
    need_cols = {"street", "hand_id", "board_cluster_id", "samples"}
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")

    # Determine #workers
    if threads == "auto":
        workers = max(1, os.cpu_count() or 1)
    else:
        workers = max(1, int(threads))

    # Build row tuples
    rows: List[Tuple[int, int, int, int]] = []
    for _, r in df.iterrows():
        street = _safe_int(r.get("street"), default=0)  # <-- handles NaN → 0
        hand_id = _safe_int(r.get("hand_id"))
        cluster_id = _norm_cluster_id(r.get("board_cluster_id"))
        sims = _safe_int(r.get("samples"), default=1)

        if street == 0:
            sims = preflop_sims
            cluster_compact = -1
        else:
            # apply min/max MC guards if set
            if mc_min > 0: sims = max(sims, mc_min)
            if mc_max > 0: sims = min(sims, mc_max)
            cluster_compact = -1 if cluster_id is None else int(cluster_id)

        rows.append((street, cluster_compact, hand_id, sims))

    t0 = time.time()
    results: List[Tuple[int, Optional[int], int, int, float, float, float, float]] = []

    # Chunked parallel processing
    def chunks(it: List[Any], n: int) -> Iterable[List[Any]]:
        for i in range(0, len(it), n):
            yield it[i:i+n]

    processed = 0
    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        for batch in chunks(rows, chunk_size):
            futs = [ex.submit(_row_worker, row, seed, sampler_blob, river_exact) for row in batch]
            for fut in cf.as_completed(futs):
                results.append(fut.result())
                processed += 1
                if log_every > 0 and processed % log_every == 0:
                    dt = time.time() - t0
                    print(f"[equity] processed {processed:,} rows in {dt:.1f}s …")

    # Assemble DataFrame
    out_rows = []
    for (street, cluster_id, hand_id, sims,
         p_win, p_tie, p_lose, weight,
         board_str, board_mask_52) in results:
        out_rows.append((
            int(street),
            (float('nan') if cluster_id is None else int(cluster_id)),
            int(hand_id),
            int(sims),
            float(p_win), float(p_tie), float(p_lose),
            float(weight),
            str(board_str or ""),
            list(board_mask_52 or [])
        ))

    out_df = pd.DataFrame(
        out_rows,
        columns=[
            "street", "board_cluster_id", "hand_id", "samples",
            "p_win", "p_tie", "p_lose", "weight",
            "board", "board_mask_52"
        ]
    )
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_parquet, index=False)

    dt = time.time() - t0
    print(f"✅ wrote equity parquet → {out_parquet}  rows={len(out_df):,}  in {dt:.1f}s  (workers={workers})")
    return out_parquet


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Compute EquityNet parquet from unified equity manifest (cluster-aware).")
    ap.add_argument("--config", type=str, default="equitynet",
                    help="Model name or YAML path, resolved by load_model_config (e.g. equity/base or configs/equity_base.yaml)")
    args = ap.parse_args()

    cfg = load_model_config(args.config)
    compute_equity_from_manifest_cfg(cfg)