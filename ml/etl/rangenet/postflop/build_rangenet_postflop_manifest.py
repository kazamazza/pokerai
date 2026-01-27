from __future__ import annotations

import argparse
import json
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

ROOT_DIR = Path(__file__).resolve().parents[4]
sys.path.append(str(ROOT_DIR))

# Allowed dependency (you explicitly trust this)
from ml.etl.rangenet.preflop.range_lookup import PreflopRangeLookup
from infra.storage.s3_client import S3Client
from ml.features.boards.board_clusterers.kmeans import KMeansBoardClusterer

# You already have this in your repo (you said this part is ok)
# NOTE: we do NOT use the old helpers_topology/_menu_for/_ctx_for_lookup/sanitize_pairs etc.
from ml.features.boards.representatives import (
    discover_representative_flops,
    discover_representative_turns,
    discover_representative_rivers,
)

# -----------------------
# Small local utilities
# -----------------------
POS_SET = {"UTG", "HJ", "MP", "CO", "BTN", "SB", "BB"}

def canon_pos(p: str) -> str:
    s = str(p or "").strip().upper()
    if s == "MP":
        return "MP"
    if s not in POS_SET:
        raise ValueError(f"Illegal position: {p}")
    return s

def street_name(street: int) -> str:
    if street == 1:
        return "flop"
    if street == 2:
        return "turn"
    if street == 3:
        return "river"
    raise ValueError(f"Illegal postflop street={street}")

def _board_to_cards(board: str) -> List[str]:
    # "Ts5cKd2h" -> ["Ts","5c","Kd","2h"]
    b = (board or "").strip()
    if len(b) % 2 != 0:
        raise ValueError(f"Bad board string: {board}")
    return [b[i:i+2] for i in range(0, len(b), 2)]

def solve_sha1(params: Dict[str, Any]) -> str:
    # minimal canonicalization: only fields that define a unique solve output
    canon_fields = [
        "street", "ctx", "topology", "role",
        "ip_pos", "oop_pos",
        "effective_stack_bb", "pot_bb",
        "bet_sizing_id", "bet_sizes",
        "board", "board_cluster_id",
        "range_ip", "range_oop",
        "accuracy", "max_iter", "allin_threshold",
        "stake",
    ]
    payload = {k: params.get(k) for k in canon_fields if k in params}
    txt = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(txt.encode("utf-8")).hexdigest()

def s3_key_base(params: Dict[str, Any], sha: str, prefix: str) -> str:
    st = int(params["street"])
    pos = f"{params['ip_pos']}v{params['oop_pos']}"
    stack = float(params["effective_stack_bb"])
    pot = float(params["pot_bb"])
    acc = float(params["accuracy"])

    board = str(params.get("board") or "")
    if not board:
        board = f"cluster_{int(params.get('board_cluster_id', -1))}"

    shard = sha[:2]

    stack_str = f"{stack:.2f}".rstrip("0").rstrip(".")
    pot_str   = f"{pot:.2f}".rstrip("0").rstrip(".")
    acc_str   = f"{acc:.3f}".rstrip("0").rstrip(".")

    return (
        f"{prefix}"
        f"/street={st}"
        f"/pos={pos}"
        f"/stack={stack_str}"
        f"/pot={pot_str}"
        f"/board={board}"
        f"/acc={acc_str}"
        f"/sizes={params['bet_sizing_id']}"
        f"/{shard}/{sha}"
    )

def compute_pot_bb(*, ctx: str, opener: Optional[str], threebettor: Optional[str], stake_cfg: Dict[str, Any]) -> float:
    """
    Deterministic pot size model for manifest only.
    Uses ONLY solver.yaml stake_cfg values (open_x/threebet_x/fourbet_x + pot_adj).
    """
    c = str(ctx).upper()
    padj = stake_cfg.get("pot_adj", {}) or {}
    open_x = stake_cfg.get("open_x", {}) or {}
    three_x = stake_cfg.get("threebet_x", {}) or {}
    four_x = float(stake_cfg.get("fourbet_x", 24.0))

    def pot_formula(final_raise_to: float, base: float = 1.5) -> float:
        return base + 2.0 * float(final_raise_to)

    if c in {"LIMPED_SINGLE", "LIMPED_MULTI"}:
        return 1.5

    if c in {"VS_OPEN", "SRP"}:
        op = canon_pos(opener or "BTN")
        x = float(open_x.get(op, 2.5))
        return float(padj.get("srp", 1.0)) * pot_formula(x)

    if c == "BLIND_VS_STEAL":
        op = canon_pos(opener or "BTN")
        x = float(open_x.get(op, 2.5))
        # treat as SRP pot baseline (steal is still an open)
        return float(padj.get("srp", 1.0)) * pot_formula(x)

    if c == "VS_3BET":
        # if threebettor is IP vs OOP controls sizing x
        # We infer IP/OOP via positions (below); here we only need "IP or OOP" label.
        # We pass threebettor as "IP" or "OOP" from caller.
        label = str(threebettor or "OOP").upper()  # "IP" or "OOP"
        x = float(three_x.get(label, three_x.get("OOP", 9.0)))
        return float(padj.get("threebet", 1.0)) * pot_formula(x)

    if c == "VS_4BET":
        return float(padj.get("fourbet", 1.0)) * pot_formula(four_x)

    # fallback
    return 6.0

def load_yaml(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def load_clusterer_kmeans(artifact_path: str):
    """
    Street-aware clusterer loader.
    This assumes your KMeansBoardClusterer has .load(path) API.
    If your actual class lives elsewhere, change this import here only.
    """

    return KMeansBoardClusterer.load(artifact_path)

# -----------------------
# Builder
# -----------------------
@dataclass(frozen=True)
class SolverProfile:
    accuracy: float
    max_iter: int
    allin_threshold: float

def profile_for(menu_id: str, solver_yaml: Dict[str, Any]) -> SolverProfile:
    table = (solver_yaml.get("solver_profiles") or {})
    default = (table.get("default") or {"accuracy": 0.02, "max_iter": 4000, "allin_threshold": 0.67})
    row = table.get(menu_id) or default
    return SolverProfile(
        accuracy=float(row["accuracy"]),
        max_iter=int(row["max_iter"]),
        allin_threshold=float(row["allin_threshold"]),
    )

def bet_sizes_for(menu_id: str, street: int, stake_cfg: Dict[str, Any]) -> List[float]:
    menus = (stake_cfg.get("bet_menus") or {})
    if menu_id not in menus:
        raise KeyError(f"bet_menus missing menu_id={menu_id}")
    sname = street_name(street)
    sizes = menus[menu_id].get(sname)
    if not sizes:
        raise KeyError(f"bet_menus[{menu_id}].{sname} missing")
    return [float(x) for x in sizes]

def stacks_for_ctx(ctx: str, stake_cfg: Dict[str, Any]) -> List[float]:
    table = (stake_cfg.get("stacks_by_ctx") or {})
    xs = table.get(str(ctx).upper())
    if not xs:
        raise KeyError(f"stacks_by_ctx missing ctx={ctx}")
    return [float(x) for x in xs]

def ip_oop_from_pair(ctx: str, a: str, b: str) -> Tuple[str, str]:
    """
    IMPORTANT: This is the ONLY place we interpret scenario position_pairs.
    We define it explicitly (no legacy imports):

    - For VS_OPEN / BLIND_VS_STEAL:
        pairs are [OPENER, DEFENDER] and postflop IP/OOP is usually opener IP vs blind OOP.
        Special case: SB opened vs BB called -> IP=BB, OOP=SB.
    - For VS_3BET / VS_4BET:
        pairs are [OPENER, OTHER] (commonly [BTN, BB]).
        Postflop IP/OOP is still based on seating order (postflop order),
        but for your current pair lists it's always late-pos vs blind -> IP=late-pos.
        If you later add [BB, BTN], you'll get IP=BTN (since BTN acts last postflop).
    - For LIMPED_SINGLE:
        pairs are [BB, SB] (you explicitly wrote that), so IP=BB, OOP=SB.
    """
    c = str(ctx).upper()
    A, B = canon_pos(a), canon_pos(b)

    if c == "LIMPED_SINGLE":
        # your scenario explicitly uses ["BB","SB"]
        return ("BB", "SB")

    # postflop action order: SB, BB, UTG, MP, HJ, CO, BTN
    order_post = ["SB", "BB", "UTG", "MP", "HJ", "CO", "BTN"]

    def later(p1: str, p2: str) -> str:
        return p1 if order_post.index(p1) > order_post.index(p2) else p2

    def earlier(p1: str, p2: str) -> str:
        return p1 if order_post.index(p1) < order_post.index(p2) else p2

    ip = later(A, B)
    oop = earlier(A, B)

    # SB open vs BB call in SRP: postflop IP=BB, OOP=SB (already handled by order_post)
    return (ip, oop)

def build_manifest(*, postflop_yaml_path: str, solver_yaml_path: str, stake_key: str, out_path: str) -> pd.DataFrame:
    cfg = load_yaml(postflop_yaml_path)
    solver_yaml = load_yaml(solver_yaml_path)

    # stake cfg lives under solver yaml key "Stakes.NL10" (exactly as you showed)
    stake_cfg = solver_yaml.get(stake_key)
    if not stake_cfg:
        raise KeyError(f"solver.yaml missing stake section '{stake_key}'")

    mb = cfg.get("manifest_build") or {}
    inputs = cfg.get("inputs") or {}
    board_cfg = cfg.get("board_clustering") or {}

    include_missing = bool(mb.get("include_missing", False))
    allow_pair_subs = bool(mb.get("allow_pair_subs", True))
    max_stack_delta = mb.get("lookup_max_stack_delta", None)
    max_stack_delta = int(max_stack_delta) if max_stack_delta is not None else None

    sample_pool = int(mb.get("sample_pool", 50000))
    seed = int(mb.get("seed", 42))

    boards_per_cluster_cfg = mb.get("boards_per_cluster") or {}
    if not isinstance(boards_per_cluster_cfg, dict):
        raise ValueError("manifest_build.boards_per_cluster must be a dict {flop:..,turn:..,river:..}")

    artifacts = (board_cfg.get("artifacts") or {})
    if not artifacts:
        raise ValueError("board_clustering.artifacts missing")

    scenarios = mb.get("scenarios") or []
    if not scenarios:
        raise ValueError("manifest_build.scenarios missing/empty")

    # Preflop ranges lookup (trusted)
    lookup = PreflopRangeLookup(
        monker_manifest_parquet=inputs.get("monker_manifest", "data/artifacts/monker_manifest.parquet"),
        sph_manifest_parquet=inputs.get("sph_manifest", "data/artifacts/sph_manifest.parquet"),
        s3_client=S3Client(),
        s3_vendor=inputs.get("vendor_s3_prefix", "data/vendor"),
        cache_dir=(cfg.get("solver") or {}).get("local_cache_dir", "data/vendor_cache"),
        allow_pair_subs=allow_pair_subs,
        max_stack_delta=max_stack_delta,
    )

    rows: List[Dict[str, Any]] = []
    prefix = (cfg.get("solver") or {}).get("s3_prefix", "solver/outputs/v1")

    # ---- street loop (1/2/3) ----
    for street in (1, 2, 3):
        sname = street_name(street)

        artifact_path = artifacts.get(sname)
        if not artifact_path:
            raise KeyError(f"board_clustering.artifacts missing '{sname}'")

        clusterer = load_clusterer_kmeans(artifact_path)

        n_clusters_limit = int((board_cfg.get("n_clusters") or {}).get(sname, 0) or 0)
        if n_clusters_limit <= 0:
            raise ValueError(f"board_clustering.n_clusters.{sname} missing/invalid")

        boards_per_cluster = int(boards_per_cluster_cfg.get(sname, 1))
        if boards_per_cluster <= 0:
            raise ValueError(f"boards_per_cluster.{sname} must be >=1")

        # discover representative boards for this street
        if street == 1:
            boards_by_cluster = discover_representative_flops(
                clusterer=clusterer,
                n_clusters_limit=n_clusters_limit,
                boards_per_cluster=boards_per_cluster,
                seed=seed,
                sample_pool=sample_pool,
            )
        elif street == 2:
            boards_by_cluster = discover_representative_turns(
                clusterer=clusterer,
                n_clusters_limit=n_clusters_limit,
                boards_per_cluster=boards_per_cluster,
                seed=seed,
                sample_pool=max(sample_pool, 80000),
            )
        else:
            boards_by_cluster = discover_representative_rivers(
                clusterer=clusterer,
                n_clusters_limit=n_clusters_limit,
                boards_per_cluster=boards_per_cluster,
                seed=seed,
                sample_pool=max(sample_pool, 120000),
            )

        cluster_ids = sorted(boards_by_cluster.keys())

        # ---- scenario loop ----
        for sc in scenarios:
            name = str(sc.get("name") or "").strip()
            if not name:
                raise ValueError("Scenario missing name")
            ctx = str(sc.get("ctx") or "").upper()
            topo = str(sc.get("topology") or "").upper()
            role = str(sc.get("role") or "").upper()
            pairs = sc.get("position_pairs") or []
            menus = sc.get("bet_menus") or []
            if not ctx or not topo or not role:
                raise ValueError(f"Scenario '{name}' missing ctx/topology/role")
            if not pairs:
                raise ValueError(f"Scenario '{name}' missing position_pairs")
            if not menus or len(menus) != 1:
                raise ValueError(f"Scenario '{name}' must have exactly 1 bet_menu in v1")

            bet_sizing_id = str(menus[0])

            # stack grid from solver stake cfg
            stack_grid = stacks_for_ctx(ctx, stake_cfg)

            # solver knobs from solver_profiles
            prof = profile_for(bet_sizing_id, solver_yaml)

            for stack_bb in stack_grid:
                for (a, b) in pairs:
                    ip_pos, oop_pos = ip_oop_from_pair(ctx, a, b)

                    # minimal opener/threebettor tags for pot sizing:
                    opener = canon_pos(a) if ctx in {"VS_OPEN", "BLIND_VS_STEAL"} else canon_pos(a)
                    threebettor_label: Optional[str] = None
                    if ctx == "VS_3BET":
                        # infer whether the 3bettor is IP or OOP from seating order
                        # with your current pairs, this will usually be OOP (blind)
                        threebettor_seat = canon_pos(b)
                        threebettor_label = "IP" if threebettor_seat == ip_pos else "OOP"

                    pot_bb = compute_pot_bb(
                        ctx=ctx,
                        opener=opener,
                        threebettor=threebettor_label,
                        stake_cfg=stake_cfg,
                    )

                    # ranges (trusted component)
                    try:
                        range_ip, range_oop, meta_rng = lookup.ranges_for_pair(
                            stack_bb=stack_bb,
                            ip=ip_pos,
                            oop=oop_pos,
                            ctx=ctx,
                            strict=False,
                        )
                    except Exception as ex:
                        range_ip, range_oop = None, None
                        meta_rng = {"error": str(ex)}

                    if (range_ip is None or range_oop is None) and not include_missing:
                        continue

                    # street-aware bet sizes
                    bet_sizes = bet_sizes_for(bet_sizing_id, street, stake_cfg)

                    for cid in cluster_ids:
                        for board in boards_by_cluster[cid]:
                            params = {
                                "stake": stake_key,  # e.g. "Stakes.NL10"
                                "scenario": name,

                                "street": street,
                                "ctx": ctx,
                                "topology": topo,
                                "role": role,

                                "ip_pos": ip_pos,
                                "oop_pos": oop_pos,

                                "board_cluster_id": int(cid),
                                "board": str(board),

                                "pot_bb": float(pot_bb),
                                "effective_stack_bb": float(stack_bb),

                                "bet_sizing_id": bet_sizing_id,
                                "bet_sizes": list(bet_sizes),

                                "range_ip": range_ip or "",
                                "range_oop": range_oop or "",

                                "accuracy": float(prof.accuracy),
                                "max_iter": int(prof.max_iter),
                                "allin_threshold": float(prof.allin_threshold),
                                "solver_version": "v1",
                            }

                            sha = solve_sha1(params)
                            s3_key = s3_key_base(params, sha=sha, prefix=prefix)

                            rows.append({
                                **params,
                                "sha1": sha,
                                "s3_key": s3_key,
                                "node_key": "root",
                                "weight": 1.0,
                                # range lookup metadata (keep for debugging/auditing)
                                "range_source": meta_rng.get("source", "unknown"),
                                "range_ip_source_stack": meta_rng.get("range_ip_source_stack"),
                                "range_oop_source_stack": meta_rng.get("range_oop_source_stack"),
                                "range_ip_stack_delta": meta_rng.get("range_ip_stack_delta"),
                                "range_oop_stack_delta": meta_rng.get("range_oop_stack_delta"),
                                "range_pair_substituted": bool(meta_rng.get("range_pair_substituted", False)),
                                "range_ip_source_pair": meta_rng.get("range_ip_source_pair"),
                                "range_oop_source_pair": meta_rng.get("range_oop_source_pair"),
                                "range_error": meta_rng.get("error"),
                            })

    df = pd.DataFrame(rows)

    # dedup (prefer sha1)
    before = len(df)
    if "sha1" in df.columns:
        df = df.drop_duplicates(subset=["sha1"]).reset_index(drop=True)
    removed = before - len(df)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    print(f"✅ wrote manifest: {out} rows={len(df):,} (removed dups={removed:,})")
    return df


def main() -> None:
    ap = argparse.ArgumentParser("Build street-aware RangeNet postflop manifest (fresh v1).")
    ap.add_argument("--postflop_yaml", required=True, help="Path to postflop policy YAML")
    ap.add_argument("--solver_yaml", required=True, help="Path to solver YAML (stake cfg + solver_profiles)")
    ap.add_argument("--stake_key", default="Stakes.NL10", help="Stake section key inside solver.yaml")
    ap.add_argument("--out", default="data/artifacts/rangenet_postflop_manifest.parquet")
    args = ap.parse_args()

    build_manifest(
        postflop_yaml_path=args.postflop_yaml,
        solver_yaml_path=args.solver_yaml,
        stake_key=args.stake_key,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()