"""Audited, one-time migration of stopping policy; preserve weights and optimizer."""

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
import torch
from runtime_io import atomic_json
from publish_model_status import alive

ROOT = Path(__file__).resolve().parents[1]


def main():
    run = ROOT / "work_dirs/imdb_full_v1"
    live = json.loads((run / "status_live.json").read_text())
    assert live["stage"] == "paused", live
    assert alive(live.get("pid")) is False, "Wait until old process fully exits."
    config_file = ROOT / "configs/full_imdb.json"
    old = json.loads(config_file.read_text())
    assert old["max_epochs"] == 15 and old.get("early_stopping", True)
    new = {**old, "max_epochs": 50, "early_stopping": False}
    manifest = (ROOT / old["dataset"] / "manifest.json").read_text(encoding="utf-8")
    sign = lambda cfg: hashlib.sha256(
        (json.dumps(cfg, sort_keys=True) + manifest).encode()
    ).hexdigest()
    oldsig, newsig = sign(old), sign(new)
    stored = json.loads((run / "run_config.json").read_text())
    assert stored["signature"] == oldsig and stored["config"] == old
    assert not list(
        run.glob("*/test_metrics.json")
    ), "Do not silently extend previously test-evaluated runs."
    checkpoint_paths = sorted(list(run.glob("*/latest.pt")) + list(run.glob("*/best.pt")))
    assert checkpoint_paths
    loaded = []
    for path in checkpoint_paths:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        assert ck["signature"] == oldsig
        loaded.append((path, ck))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = ROOT / "reports" / f"epoch_extension_50_{stamp}"
    backup.mkdir()
    shutil.copy2(config_file, backup / "full_imdb_before.json")
    shutil.copy2(run / "run_config.json", backup / "run_config_before.json")
    details = []
    for path, ck in loaded:
        dest = backup / path.parent.name / path.name
        dest.parent.mkdir(exist_ok=True)
        shutil.copy2(path, dest)
        ck["signature"] = newsig
        temp = path.with_suffix(".extension.tmp")
        torch.save(ck, temp)
        temp.replace(path)
        verified = torch.load(path, map_location="cpu", weights_only=False)
        original = torch.load(dest, map_location="cpu", weights_only=False)
        assert verified["epoch"] == original["epoch"]
        assert all(torch.equal(v, verified["head"][k]) for k, v in original["head"].items())
        if "optimizer" in original:
            assert original["optimizer"]["param_groups"] == verified["optimizer"]["param_groups"]
            for key, state in original["optimizer"]["state"].items():
                for name, value in state.items():
                    other = verified["optimizer"]["state"][key][name]
                    assert torch.equal(value, other) if torch.is_tensor(value) else value == other
        details.append(
            {
                "path": str(path),
                "epoch": verified["epoch"],
                "backup": str(dest),
                "weights_optimizer_preserved": True,
            }
        )
    atomic_json(config_file, new)
    atomic_json(run / "run_config.json", {**stored, "config": new, "signature": newsig})
    report = {
        "reason": "User requested 50 epochs for every model; early stopping disabled.",
        "old_max_epochs": 15,
        "new_max_epochs": 50,
        "early_stopping": False,
        "old_signature": oldsig,
        "new_signature": newsig,
        "changed_config_fields": ["max_epochs", "early_stopping"],
        "checkpoints": details,
        "backup_directory": str(backup),
    }
    atomic_json(ROOT / "reports/epoch_extension_50.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
