from pathlib import Path

from openvad_train.export_smoke import main


def test_export_matches_committed_fixture(tmp_path):
    committed = Path(__file__).resolve().parents[3] / "inferencing/fixtures/smoke/linear.safetensors"
    main([str(tmp_path / "linear.safetensors")])
    assert (tmp_path / "linear.safetensors").read_bytes() == committed.read_bytes()
