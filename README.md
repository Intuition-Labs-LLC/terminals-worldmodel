# terminals-worldmodel

A [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) extension
for a non-physical world: the configuration state of an agent or OS.

Most world-model work models physical dynamics — control suites, robot arms, video.
This models the state an agent acts *on*: which screen it's on, what's selected, the
settings it can change. The same collect → train → plan loop applies, with one
addition: an externally-measured stability signal used as the planning cost.

## The idea

**A world with no simulator.** The config state is a structured vector, not pixels.
It registers as a Gymnasium env (`terminals-os/Config-v0`), goal-conditioned, that
`swm.World(...)` wraps like any control env.

**Coherence as the cost.** A world model rolls a latent forward under a candidate
action. We score how much that rollout *settles* —

```
R = exp(-d_tail / scale)      # d_tail = mean step displacement over the final third
```

`R → 1` when the prediction converges to a fixed point, `R → 0` when it keeps
wandering. R is read off the rollout from the outside; it does not use the model's
own confidence head. The planning cost is then `1 − goal_progress × R`: prefer
actions whose predicted outcome both moves toward the goal and lands somewhere the
model is *stable* about. Any swm solver (CEM, MPPI, iCEM) minimizes it unchanged.

**The agent's own usage is the corpus.** Its trace — `(state, action, reward)` rows
— registers as a swm dataset format (`terminals-os-trace`), loadable by
`swm.data.load_dataset`, so the world model trains on real interaction.

## Install

```bash
pip install 'stable-worldmodel[env]'   # the [env] extra brings cv2/imageio swm imports need
pip install git+https://github.com/Intuition-Labs-LLC/terminals-worldmodel
```

## Use

```python
import terminals_worldmodel as twm
import stable_worldmodel as swm

twm.register()                                  # → {'env': 'terminals-os/Config-v0', 'format': 'terminals-os-trace'}
world = swm.World("terminals-os/Config-v0", num_envs=1, add_pixels=False)

# coherence-R as the swm planning cost (predict_fn is your trained latent predictor)
from stable_worldmodel.solver import CEMSolver
cost = twm.CoherenceCost(predict_fn, horizon=8)
solver = CEMSolver(model=cost, num_samples=256, device="cuda")
```

## Example

[`examples/collect_and_plan.py`](examples/collect_and_plan.py) runs the loop end to
end: swm collects trajectories from the OS-world, they load back as a swm dataset,
and coherence-R ranks candidate changes (picking the goal-reaching one).

## Status

The env, the dataset format, and the coherence cost are implemented and tested
against `stable-worldmodel` 0.1.0. The env ships a transparent stand-in dynamics so
the harness runs end to end; the production dynamics is a swm world model trained on
collected traces — `CoherenceCost` takes any latent predictor as `predict_fn`.

## Relation to prior work

Built on stable-worldmodel and the JEPA latent-prediction line. LeWM contributes
end-to-end *training* stability (next-embedding prediction + a Gaussian-latent
regularizer). The stability here is measured at *inference*, per rollout, by a
signal the model does not control — and it is used directly as the MPC cost. The two
are complementary axes of "stable." Object-centric latents (DINO-WM, C-JEPA) are a
natural encoder for structured config state and a clean next step.

## License

AGPL-3.0-or-later · Copyright (c) 2026 Tej Desai / Intuition Labs LLC. See `LICENSE` and `NOTICE`.
