from __future__ import annotations


def main():
    import argparse, sys, json
    from pathlib import Path
    import pandas as pd
    import yaml

    # --- imports from your codebase (with safe fallbacks if not present) ---
    try:
        from ml.etl.utils.positions import sanitize_position_pairs, CTX_ALIASES
    except Exception:
        # Minimal local fallbacks
        from typing import List, Tuple, Set
        Pair = Tuple[str, str]
        def _canon_pair(ip: str, oop: str) -> Pair:
            return str(ip).strip().upper(), str(oop).strip().upper()
        CTX_ALIASES = {
            "SRP": "SRP", "OPEN": "SRP", "VS_OPEN": "SRP", "VS_OPEN_RFI": "SRP",
            "BLIND_VS_STEAL": "BLIND_VS_STEAL", "BVS": "BLIND_VS_STEAL",
            "VS_3BET": "VS_3BET", "3BET": "VS_3BET",
            "VS_4BET": "VS_4BET", "4BET": "VS_4BET",
            "LIMPED_SINGLE": "LIMP_SINGLE", "LIMP_SINGLE": "LIMP_SINGLE",
            "LIMPED_MULTI": "LIMP_MULTI",   "LIMP_MULTI": "LIMP_MULTI",
        }
        def sanitize_position_pairs(pairs_in: List[Pair], ctx: str) -> List[Pair]:
            # Accept all pairs if we don't have your curated VALID_* sets available.
            seen: Set[Pair] = set()
            out: List[Pair] = []
            for ip, oop in pairs_in:
                ip2, oop2 = _canon_pair(ip, oop)
                if ip2 == oop2:
                    continue
                if (ip2, oop2) not in seen:
                    seen.add((ip2, oop2)); out.append((ip2, oop2))
            return out

    # ---------- args ----------
    ap = argparse.ArgumentParser(description="Detailed coverage report for preflop range availability (Monker/SPH) per scenario")
    ap.add_argument("--manifest-build", type=Path, required=True,
                    help="YAML with manifest_build.scenarios (ctx, stacks, pairs, etc.)")
    ap.add_argument("--monker", type=Path, default=Path("data/artifacts/monker_manifest.parquet"),
                    help="Path to Monker manifest parquet")
    ap.add_argument("--sph", type=Path, default=Path("data/artifacts/sph_manifest.parquet"),
                    help="Path to SPH manifest parquet")
    ap.add_argument("--limit-missing", type=int, default=40, help="Max missing items to print per scenario")
    ap.add_argument("--show-found", action="store_true", help="Also print covered items (verbose)")
    args = ap.parse_args()

    # ---------- load manifests ----------
    monker_df = pd.read_parquet(args.monker) if args.monker.exists() else pd.DataFrame()
    sph_df    = pd.read_parquet(args.sph)    if args.sph.exists()    else pd.DataFrame()

    # Normalize columns we rely on
    def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty: return df
        for c in ("ctx","ip_pos","oop_pos","hero_pos"):
            if c in df.columns:
                df[c] = df[c].astype(str).str.upper()
        if "stack_bb" in df.columns:
            df["stack_bb"] = pd.to_numeric(df["stack_bb"], errors="coerce").astype("Int64")
        return df

    monker_df = _norm_cols(monker_df)
    sph_df    = _norm_cols(sph_df)

    # Monker availability keyed by (ctx, stack, ip, oop, hero)
    def _mk_key_rows(df: pd.DataFrame):
        if df.empty: return set()
        need = {"ctx","stack_bb","ip_pos","oop_pos","hero_pos"}
        if not need.issubset(df.columns):
            # Back-compat: try to synthesize from older schema if present
            # (skip if truly missing)
            return set()
        rows = set()
        for _, r in df.iterrows():
            if pd.isna(r["stack_bb"]) or not r["ctx"] or not r["ip_pos"] or not r["oop_pos"] or not r["hero_pos"]:
                continue
            rows.add((str(r["ctx"]),
                      int(r["stack_bb"]),
                      str(r["ip_pos"]),
                      str(r["oop_pos"]),
                      str(r["hero_pos"])))
        return rows

    monker_rows = _mk_key_rows(monker_df)
    sph_rows    = _mk_key_rows(sph_df)

    # ---------- load scenarios ----------
    cfg = yaml.safe_load(args.manifest_build.read_text(encoding="utf-8"))
    mb  = (cfg.get("manifest_build") or {})
    scenarios = mb.get("scenarios") or []
    if not scenarios:
        print("No scenarios found under manifest_build.scenarios in YAML.", file=sys.stderr)
        sys.exit(2)

    # Global defaults (used if a scenario omits its own)
    stacks_default = [float(x) for x in mb.get("stacks_bb", [100])]
    pairs_default  = [tuple(x) for x in mb.get("position_pairs", [("BTN","BB")])]

    # ---------- helpers ----------
    def _canon_ctx(ctx: str) -> str:
        return CTX_ALIASES.get(str(ctx).upper(), str(ctx).upper())

    def _nearest_stack(target: float, candidates: list[int]) -> int | None:
        if not candidates: return None
        t = int(round(target))
        return sorted(candidates, key=lambda s: (abs(s - t), s))[0]

    # Build a per-scenario report
    grand_total = 0
    grand_ok = grand_monker = grand_sph = 0
    any_missing = False

    for sc in scenarios:
        name = str(sc.get("name") or sc.get("ctx") or "SCENARIO").upper()
        ctx  = _canon_ctx(sc.get("ctx") or "SRP")
        stacks = [int(round(float(x))) for x in sc.get("stacks_bb", stacks_default)]
        raw_pairs = [tuple(x) for x in sc.get("position_pairs", pairs_default)]

        pairs: List[tuple[str,str]] = sanitize_position_pairs(raw_pairs, ctx)
        if not pairs:
            print(f"\n[warn] {name}: no legal (IP,OOP) pairs for ctx={ctx}; skipping.")
            continue

        # Candidate stacks available in each source for nearest-fallback diagnostics
        monker_stacks = sorted({s for (_, s, _, _, _) in monker_rows if _ == ctx})
        sph_stacks    = sorted({s for (_, s, _, _, _) in sph_rows if _ == ctx})

        total = covered = n_monker = n_sph = 0
        missing_details = []
        found_details   = []

        for stack in stacks:
            for (ip, oop) in pairs:
                total += 1
                # We require both sides (hero = IP and hero = OOP)
                m_ip  = (ctx, stack, ip, oop, "IP")  in monker_rows
                m_oop = (ctx, stack, ip, oop, "OOP") in monker_rows
                s_ip  = (ctx, stack, ip, oop, "IP")  in sph_rows
                s_oop = (ctx, stack, ip, oop, "OOP") in sph_rows

                use_monker = m_ip and m_oop
                use_sph    = (not use_monker) and (s_ip and s_oop)

                if use_monker:
                    covered += 1; n_monker += 1
                    if args.show_found:
                        found_details.append(f"{ctx}:{ip}v{oop}@{stack} → MONKER")
                elif use_sph:
                    covered += 1; n_sph += 1
                    if args.show_found:
                        found_details.append(f"{ctx}:{ip}v{oop}@{stack} → SPH")
                else:
                    # missing; compute nearest stack hints
                    mn = _nearest_stack(stack, monker_stacks)
                    sn = _nearest_stack(stack, sph_stacks)
                    hint = []
                    if mn is not None:
                        have_m_ip  = (ctx, mn, ip, oop, "IP")  in monker_rows
                        have_m_oop = (ctx, mn, ip, oop, "OOP") in monker_rows
                        if have_m_ip or have_m_oop:
                            hint.append(f"nearest_monker={mn}({int(have_m_ip)}IP/{int(have_m_oop)}OOP)")
                    if sn is not None:
                        have_s_ip  = (ctx, sn, ip, oop, "IP")  in sph_rows
                        have_s_oop = (ctx, sn, ip, oop, "OOP") in sph_rows
                        if have_s_ip or have_s_oop:
                            hint.append(f"nearest_sph={sn}({int(have_s_ip)}IP/{int(have_s_oop)}OOP)")
                    missing_details.append(f"{ctx}:{ip}v{oop}@{stack}" + (f"  [{' | '.join(hint)}]" if hint else ""))

        # print scenario block
        pct = (covered / total * 100.0) if total else 0.0
        print(f"\n=== {name} (ctx={ctx}) ===")
        print(f"pairs={len(pairs)}  stacks={len(stacks)}  combos={total}")
        print(f"covered={covered} ({pct:.1f}%)  • monker={n_monker}  • sph={n_sph}  • missing={total - covered}")

        if args.show_found and found_details:
            print("  -- covered examples --")
            for line in found_details[:min(40, len(found_details))]:
                print("   ", line)
            if len(found_details) > 40:
                print(f"   … and {len(found_details)-40} more")

        if missing_details:
            any_missing = True
            cap = args.limit_missing or len(missing_details)
            print("  -- missing (top) --")
            for line in missing_details[:cap]:
                print("   ", line)
            if len(missing_details) > cap:
                print(f"   … and {len(missing_details)-cap} more")

        # accumulate global
        grand_total += total
        grand_ok    += covered
        grand_monker += n_monker
        grand_sph    += n_sph

    # ---------- global footer ----------
    if grand_total:
        g_pct = grand_ok / grand_total * 100.0
        print(f"\n=== GLOBAL ===")
        print(f"total={grand_total}  covered={grand_ok} ({g_pct:.1f}%)  • monker={grand_monker}  • sph={grand_sph}  • missing={grand_total - grand_ok}")
    else:
        print("\nNo scenario combos to evaluate (check your YAML).")

    # Exit non-zero if anything missing (useful in CI)
    if any_missing:
        sys.exit(3)
        print(line)


if __name__ == "__main__":
    main()