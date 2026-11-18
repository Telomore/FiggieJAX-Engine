"""Train a small agent with collection and repeated updates compiled in JAX."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

import jax
import numpy as np

from figgie_jax import __version__
from figgie_jax.learning import collect, game_metrics, initialize, make_train
from figgie_jax.types import Config


def main() -> None:
    """Verify the backend, train, evaluate held-out deals, and save parameters."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--players", type=int, choices=(4, 5), default=4)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--updates", type=int, default=64)
    parser.add_argument("--eval-games", type=int, default=256)
    parser.add_argument("--duration-ms", type=int, default=10_000)
    parser.add_argument("--max-events", type=int, default=50_000)
    parser.add_argument("--max-price", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if jax.default_backend() != args.backend:
        parser.error(f"Requested {args.backend}; actual backend is {jax.default_backend()}")
    if args.eval_games < 1:
        parser.error("eval-games must be positive")
    config = Config(
        num_players=args.players,
        duration_ms=args.duration_ms,
        max_events=args.max_events,
        max_price=args.max_price,
    )
    root_key = jax.random.key(args.seed)
    initial = initialize(jax.random.fold_in(root_key, 0), config)
    train_key = jax.random.fold_in(root_key, 10)
    eval_key = jax.random.fold_in(root_key, 20)
    train = make_train(
        config, args.batch, args.updates, args.learning_rate, args.entropy_coefficient
    )
    evaluate = jax.jit(lambda w: game_metrics(collect(w, eval_key, config, args.eval_games)))
    print(f"Compiling on {jax.devices()}...", flush=True)
    start = time.perf_counter()
    executable = train.lower(initial, train_key).compile()
    compile_seconds = time.perf_counter() - start
    # Warm the exact executable, but discard its weights and reuse the same keys.
    jax.block_until_ready(executable(initial, train_key))
    before = jax.block_until_ready(evaluate(initial))
    start = time.perf_counter()
    weights, history = jax.block_until_ready(executable(initial, train_key))
    train_seconds = time.perf_counter() - start
    after = jax.block_until_ready(evaluate(weights))
    if not np.isfinite(np.asarray(weights)).all():
        raise RuntimeError("Training produced nonfinite parameters")
    if not all(np.isfinite(np.asarray(value)).all() for value in history.values()):
        raise RuntimeError("Training produced nonfinite metrics")
    if (
        np.asarray(history["truncated_games"]).sum()
        or np.asarray(history["unfinished_games"]).sum()
    ):
        raise RuntimeError("Training contains incomplete games; no success report written")
    if int(after["completed_games"]) != args.eval_games:
        raise RuntimeError("Evaluation contains incomplete games")
    change = float(np.linalg.norm(np.asarray(weights - initial)))
    if not change > 0 or not np.asarray(history["update_applied"]).all():
        raise RuntimeError("The policy did not update on every training batch")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output.with_suffix(".npz")
    np.savez(checkpoint, weights=np.asarray(weights), initial_weights=np.asarray(initial))
    root = Path(__file__).resolve().parents[1]
    sources = sorted((root / "src").rglob("*.py")) + [
        Path(__file__).resolve(),
        root / "pyproject.toml",
        root / "uv.lock",
        root / "Dockerfile",
    ]
    report = {
        "results_are_simulated": True,
        "claim": "Compiled policy updates verified; strategic strength is not established.",
        "version": __version__,
        "configuration": asdict(config),
        "training": {
            "batch": args.batch,
            "updates": args.updates,
            "completed_games": int(np.asarray(history["completed_games"]).sum()),
            "learner_decisions": int(np.asarray(history["learner_decisions"]).sum()),
            "truncated_games": int(np.asarray(history["truncated_games"]).sum()),
            "seed": args.seed,
            "key_domains": {"initialization": 0, "training": 10, "evaluation": 20},
            "learning_rate": args.learning_rate,
            "entropy_coefficient": args.entropy_coefficient,
            "reward_scale_chips": 100,
            "gradient_clip_norm": 1,
            "baseline": "leave-one-out completed batch mean, zero for singleton",
            "learner_seat": "uniform per table and round",
            "opponents": "fixed heuristic_policy",
            "parameter_count": weights.size,
            "parameter_l2_change": change,
            "compile_seconds": compile_seconds,
            "synchronized_train_seconds_excluding_compile_and_warmup": train_seconds,
            "history": {name: np.asarray(value).tolist() for name, value in history.items()},
        },
        "evaluation": {
            "games": args.eval_games,
            "initial": {name: np.asarray(value).item() for name, value in before.items()},
            "final": {name: np.asarray(value).item() for name, value in after.items()},
            "protocol": "same held-out deal/policy keys and seats before and after training",
        },
        "runtime": {
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "device_kind": jax.devices()[0].device_kind,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("jax", "jaxlib", "numpy", "pydantic")
            },
        },
        "source_sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        },
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(args.output),
                "checkpoint": str(checkpoint),
                "completed_training_games": report["training"]["completed_games"],
                "parameter_l2_change": change,
                "evaluation": report["evaluation"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
