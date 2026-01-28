from __future__ import annotations
import pandas as pd
from ml.etl.rangenet.postflop.build_postflop_policy import build_postflop_policy
from ml.etl.rangenet.postflop.texas_solver_extractor import SolverExtraction
from ml.models.vocab_actions import ROOT_ACTION_VOCAB, FACING_ACTION_VOCAB

def test_build_postflop_policy_happy_path(tmp_path, monkeypatch):
    # --- create a tiny manifest parquet matching your schema columns used in builder ---
    manifest = pd.DataFrame([{
        "sha1": "abc123",
        "s3_key": "solver/outputs/v1/street=1/pos=BTNvBB/xxx",
        "street": 1,
        "board": "Ah9hAs",
        "ip_pos": "BTN",
        "oop_pos": "BB",
        "ctx": "VS_3BET",
        "bet_sizing_id": "3bet_hu.Aggressor_IP",
        "pot_bb": 10.0,
        "effective_stack_bb": 100.0,
        "bet_sizes": [{"element": 0.33}, {"element": 0.66}],
        "stake": "Stakes.NL10",
        "scenario": "unit",
        "board_cluster_id": 0,
        "solver_version": "v1",
    }])
    manifest_path = tmp_path / "manifest.parquet"
    manifest.to_parquet(manifest_path, index=False)

    # --- stub solver yaml file ---
    solver_yaml_path = tmp_path / "solver.yaml"
    solver_yaml_path.write_text(
        "Stakes.NL10:\n  raise_mult: [2.0, 3.0, 4.5]\n",
        encoding="utf-8"
    )

    # --- monkeypatch all IO / external deps ---
    # parse_bet_sizes_cell should turn [{"element":0.33},..] -> [33,66]
    def parse_bet_sizes_cell(cell):
        return [33, 66]
    monkeypatch.setattr("ml.etl.rangenet.postflop.build_postflop_policy.parse_bet_sizes_cell", parse_bet_sizes_cell)

    # size_key_for should append size suffix
    def size_key_for(base_key: str, size_pct: int) -> str:
        return f"{base_key}/size={size_pct}p/output_result.json.gz"
    monkeypatch.setattr("ml.etl.rangenet.postflop.build_postflop_policy.size_key_for", size_key_for)

    # no s3, just pretend file exists after “download”
    def s3_exists(*args, **kwargs): return True
    monkeypatch.setattr("ml.etl.rangenet.postflop.build_postflop_policy.s3_exists", s3_exists)

    def s3_download_to(s3, bucket, key, local_path):
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr("ml.etl.rangenet.postflop.build_postflop_policy.s3_download_to", s3_download_to)

    # extractor mock returns fixed mixes
    class FakeExtractor:
        def extract(self, *args, **kwargs):
            return SolverExtraction(
                ctx=kwargs["ctx"],
                ip_pos=kwargs["ip_pos"],
                oop_pos=kwargs["oop_pos"],
                board=kwargs["board"],
                pot_bb=kwargs["pot_bb"],
                stack_bb=kwargs["stack_bb"],
                bet_sizing_id=kwargs["bet_sizing_id"],
                root_mix={"CHECK": 0.2, "BET 33%": 0.8},
                facing_mix={"FOLD": 0.1, "CALL": 0.6, "RAISE 3X": 0.3},
                ok=True,
            )

    monkeypatch.setattr(
        "ml.etl.rangenet.postflop.texas_solver_extractor.TexasSolverExtractor",
        FakeExtractor,
    )

    # invariants accept mock
    monkeypatch.setattr("ml.etl.rangenet.postflop.build_postflop_policy.validate_extractor_output", lambda ex: None)

    # capture writes instead of writing parquet parts
    written = {"root": [], "facing": []}

    def write_part(df: pd.DataFrame, path):
        if "size_pct" in df.columns:
            written["root"].append(df.copy())
        else:
            written["facing"].append(df.copy())

    monkeypatch.setattr("ml.etl.rangenet.postflop.build_postflop_policy.write_part", write_part)

    # --- run builder ---

    build_postflop_policy(
        manifest_path=str(manifest_path),
        solver_yaml_path=str(solver_yaml_path),
        stake_key="Stakes.NL10",
        s3_bucket="bucket",
        s3_region="eu-central-1",
        out_root_dir=str(tmp_path / "out_root"),
        out_facing_dir=str(tmp_path / "out_facing"),
        part_rows=1,  # force flush quickly
        strict="emit_sentinel",
        local_cache_dir=str(tmp_path / "cache")
    )

    assert len(written["root"]) > 0
    assert len(written["facing"]) > 0

    root_df = pd.concat(written["root"], ignore_index=True)
    facing_df = pd.concat(written["facing"], ignore_index=True)

    # should have one row per size
    assert set(root_df["size_pct"].tolist()) == {33, 66}
    assert set(facing_df["faced_size_pct"].tolist()) == {33, 66}

    # vocab columns exist
    for a in ROOT_ACTION_VOCAB:
        assert a in root_df.columns
    for a in FACING_ACTION_VOCAB:
        assert a in facing_df.columns