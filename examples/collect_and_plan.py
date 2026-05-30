# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Intuition Labs LLC
"""End to end: stable-worldmodel collects from the OS-world, and coherence-R ranks
config changes.

    pip install 'stable-worldmodel[env]' torch
    python examples/collect_and_plan.py
"""
import os
import tempfile

import numpy as np
import stable_worldmodel as swm
import torch
from stable_worldmodel.policy import RandomPolicy

import terminals_worldmodel as twm


def main() -> None:
    print("registered:", twm.register())

    # 1) swm collects trajectories from the OS-world — the same harness it uses for
    #    its physical control envs, pointed at a non-physical world.
    world = swm.World(
        twm.ENV_ID, num_envs=2, add_pixels=False, goal_conditioned=False, max_episode_steps=8
    )
    world.set_policy(RandomPolicy())
    path = os.path.join(tempfile.mkdtemp(), "rollout.lance")
    world.collect(path, episodes=4, seed=0)
    world.close()

    # 2) the collected trajectories load back as a swm dataset — the training corpus.
    ds = swm.data.load_dataset(path)
    print(f"collected {len(ds)} transitions from {twm.ENV_ID} -> swm {type(ds).__name__}")

    # 3) coherence-R as the planning cost. Rank candidate config changes by
    #    1 - goal_progress * coherence_R. `predict_fn` is the latent predictor —
    #    a transparent goal-seeking stand-in here; in production, a swm-trained model.
    goal = np.zeros(8)
    goal[0] = 1.0

    def predict_fn(z, a):
        z = np.asarray(z, dtype=float).copy()
        z[0] += 0.4 * (float(a[0]) - z[0])  # contract the goal-coordinate toward action[0]
        return z

    cost = twm.CoherenceCost(predict_fn, horizon=6)
    candidates = {
        "toward goal": [1, 0, 0, 0, 0, 0, 0, 0],
        "no-op": [0, 0, 0, 0, 0, 0, 0, 0],
        "away": [-1, 0, 0, 0, 0, 0, 0, 0],
    }
    costs = cost.get_cost(
        {"state": np.zeros(8), "goal": goal},
        torch.tensor(list(candidates.values()), dtype=torch.float32),
    )
    for name, c in zip(candidates, costs):
        print(f"  {name:>12}: cost {float(c):.3f}")
    print("planner picks:", list(candidates)[int(costs.argmin())])


if __name__ == "__main__":
    main()
