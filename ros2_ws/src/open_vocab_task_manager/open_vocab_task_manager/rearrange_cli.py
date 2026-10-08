#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from open_vocab_interfaces.action import Rearrange


class RearrangeCli(Node):
    def __init__(self):
        super().__init__("open_vocab_rearrange_cli")
        self.client = ActionClient(self, Rearrange, "/open_vocab/rearrange")


def main(args=None) -> int:
    parser = argparse.ArgumentParser(
        description="Send one complete open-vocabulary rearrangement task."
    )
    parser.add_argument("source", help="Source object query, e.g. banana or 'medicine box'")
    parser.add_argument("destination", help="Destination query/profile, e.g. 'glass bowl' or organizer")
    parser.add_argument("--verify", action="store_true", help="Run optional coarse post-place verification")
    parser.add_argument(
        "--no-pick-dry-run",
        action="store_true",
        help="Skip the validated approach_dry_run stage. Default keeps it enabled to mirror the proven manual pipeline.",
    )
    ns = parser.parse_args(args)

    rclpy.init()
    node = RearrangeCli()
    try:
        if not node.client.wait_for_server(timeout_sec=10.0):
            print("ERROR: /open_vocab/rearrange action server not available", file=sys.stderr)
            return 2

        goal = Rearrange.Goal()
        goal.source_query = ns.source
        goal.destination_query = ns.destination
        goal.verify_place = bool(ns.verify)
        goal.use_pick_dry_run = not bool(ns.no_pick_dry_run)

        def feedback_cb(msg):
            f = msg.feedback
            print(
                f"[{f.progress*100:5.1f}%] {f.state:<24} {f.message} "
                f"(elapsed={f.elapsed_s:.1f}s)",
                flush=True,
            )

        send_future = node.client.send_goal_async(goal, feedback_callback=feedback_cb)
        rclpy.spin_until_future_complete(node, send_future)
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            print("ERROR: goal rejected", file=sys.stderr)
            return 3

        print("Goal accepted. Ctrl-C requests client shutdown; use robot E-stop for unsafe motion.")
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(node, result_future)
        wrapped = result_future.result()
        if wrapped is None:
            print("ERROR: no action result", file=sys.stderr)
            return 4
        result = wrapped.result
        print("\n=== Rearrangement result ===")
        print(f"success       : {result.success}")
        print(f"task_id       : {result.task_id}")
        print(f"final_state   : {result.final_state}")
        print(f"failure_stage : {result.failure_stage}")
        print(f"total_time_s  : {result.total_time_s:.2f}")
        print(f"message       : {result.message}")
        return 0 if result.success else 5
    except KeyboardInterrupt:
        print("\nClient interrupted. If robot is moving, use the physical E-stop if necessary.")
        return 130
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
