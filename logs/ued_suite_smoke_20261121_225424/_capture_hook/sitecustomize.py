import json
import os
import sys
try:
    import wandb
    _original_init = wandb.init
    def _init_and_capture(*args, **kwargs):
        run = _original_init(*args, **kwargs)
        original_log = wandb.log
        metrics_file = os.environ.get("UED_METRICS_FILE")
        def _capture_log(data, *log_args, **log_kwargs):
            if isinstance(data, dict) and "solve_rate/mean" in data and metrics_file:
                try:
                    def scalar(value):
                        return float(value)
                    item = {
                        "num_updates": int(data.get("num_updates", -1)),
                        "solve_rate_mean": scalar(data["solve_rate/mean"]),
                        "per_level": {
                            k[len("solve_rate/"):]: scalar(v)
                            for k, v in data.items()
                            if k.startswith("solve_rate/") and k != "solve_rate/mean"
                        },
                    }
                    with open(metrics_file, "a", encoding="utf-8") as handle:
                        handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                except Exception as exc:
                    print("[UED METRIC CAPTURE WARNING]", repr(exc), file=sys.stderr)
            return original_log(data, *log_args, **log_kwargs)
        wandb.log = _capture_log
        return run
    wandb.init = _init_and_capture
except ImportError:
    pass
except Exception as exc:
    print("[UED CAPTURE SETUP WARNING]", repr(exc), file=sys.stderr)
