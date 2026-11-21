"""Host-side trace of ACCEL state and observations."""

from hashlib import sha256
import json

import jax
import numpy as np


BRANCHES = ("new", "replay", "mutation")


def base_sampler(sampler):
    return {key: value for key, value in sampler.items() if key not in ("qd_archive", "qd_max_age")}


def digest(array):
    array = np.asarray(array)
    return {
        "dtype": str(array.dtype), "shape": list(array.shape),
        "sha256": sha256(array.tobytes()).hexdigest(),
    }


def tree_digests(tree):
    return {jax.tree_util.keystr(path): digest(value)
            for path, value in jax.tree_util.tree_flatten_with_path(tree)[0]}


def encode_levels(levels):
    fields = ("wall_map", "goal_pos", "agent_pos", "agent_dir", "width", "height")
    levels = {key: np.asarray(getattr(levels, key)) for key in fields}
    walls = levels["wall_map"]
    encoded = {key: value.tolist() for key, value in levels.items() if key != "wall_map"}
    encoded["wall_shape"] = list(walls.shape)
    encoded["walls_hex"] = [np.packbits(row).tobytes().hex() for row in walls]
    encoded["identities"] = [sha256(b"".join(levels[field][i].tobytes() for field in fields)).hexdigest()
                             for i in range(len(walls))]
    return encoded


def json_scalar(value):
    value = np.asarray(value).item()
    return value if np.isfinite(value) else str(value)


class Trace:
    """Ordered JIT callbacks write private trace data and return no values."""

    def __init__(self, path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8"):
            pass
        self.last_replay_indices = None

    def snapshot(self, state, rng, round_index, stage, losses=()):
        self.write(
            stage, round_index, rng=np.asarray(rng).tolist(), optimizer_steps=int(state.step),
            params=tree_digests(state.params), optimizer=tree_digests(state.opt_state),
            sampler=tree_digests(base_sampler(state.sampler)),
            last_batches=tree_digests((state.dr_last_level_batch, state.replay_last_level_batch,
                                       state.mutation_last_level_batch)),
            curriculum=[int(state.update_state), int(state.num_dr_updates),
                        int(state.num_replay_updates), int(state.num_mutation_updates)],
            losses=[json_scalar(loss) for loss in jax.tree_util.tree_leaves(losses)],
        )

    def round_start(self, round_index, branch, rng, decision_rng):
        self.write("start", round_index, branch=BRANCHES[int(branch)],
                   rng=np.asarray(rng).tolist(), decision_rng=np.asarray(decision_rng).tolist())

    def selection(self, round_index, branch, rng, levels, indices, parents):
        replay_indices = np.asarray(indices).tolist() if branch == "replay" else []
        if branch == "replay":
            self.last_replay_indices = replay_indices
        parent_indices = self.last_replay_indices if branch == "mutation" else None
        if branch == "mutation" and parent_indices is None:
            # Resume can start immediately after replay. Recover current slots by identity.
            parent_indices = np.asarray(indices).tolist()
        self.write("selection", round_index, rng=np.asarray(rng).tolist(), levels=encode_levels(levels),
                   replay_indices=replay_indices, mutation_parent_indices=parent_indices,
                   parents=encode_levels(parents) if parents is not None else None)

    def rollout(self, round_index, rng, observations, actions, rewards, dones, log_probs, values, last_value):
        self.write("rollout", round_index, rng=np.asarray(rng).tolist(), observations=tree_digests(observations),
                   **{key: digest(value) for key, value in (
                       ("actions", actions), ("rewards", rewards), ("dones", dones),
                       ("log_probs", log_probs), ("values", values), ("last_value", last_value))})

    def buffer(self, round_index, sampler):
        size = int(sampler["size"])
        self.write("sampler", round_index, sampler=tree_digests(sampler),
                   active_scores=[json_scalar(x) for x in np.asarray(sampler["scores"])[:size]])

    def write(self, stage, round_index, **data):
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps({"round": int(round_index), "stage": stage, **data}, allow_nan=False) + "\n")
