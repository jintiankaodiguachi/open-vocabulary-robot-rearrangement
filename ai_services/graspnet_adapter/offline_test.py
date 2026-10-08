#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

from graspnet_engine import GraspNetEngine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--output", default="outputs/latest")
    args = ap.parse_args()

    engine = GraspNetEngine(args.config)

    _, result = engine.predict_npz(
        args.npz,
        save_dir=args.output,
    )

    print("=" * 88)
    print(
        "GraspNet V8 — Target-aware Stratified Sampling + "
        "Scene Collision + Original Score"
    )
    print("=" * 88)

    print(json.dumps({
        "status": result["status"],
        "version": result["version"],
        "label": result["label"],
        "proposal_source": result["proposal_source"],
        "collision_source": result["collision_source"],
        "ranking_policy": result["ranking_policy"],
        "sampling": result["sampling"],
        "counts": result["counts"],
        "raw_width_diagnostics": result["raw_width_diagnostics"],
        "score_summary": result["score_summary"],
        "latency_ms": result["latency_ms"],
        "filters": result["filters"],
    }, ensure_ascii=False, indent=2))

    if result["grasps"]:
        print("\nTop-1:")
        print(json.dumps(
            result["grasps"][0],
            ensure_ascii=False,
            indent=2,
        ))

        print("\nTop candidates:")
        for g in result["grasps"][:10]:
            print(
                f"rank={g['rank']:02d} "
                f"score={g['model_score']:.6f} "
                f"width={g['width_m']:.4f}m "
                f"target_dist={g['target_nearest_distance_m']:.4f}m "
                f"approach_down={g['approach_angle_from_down_deg']:.1f}deg"
            )
    else:
        print("\nNO grasp survived V8.")

    print("\nSaved:", Path(args.output).resolve())
    print("=" * 88)


if __name__ == "__main__":
    main()
