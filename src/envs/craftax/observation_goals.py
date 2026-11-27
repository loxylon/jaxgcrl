"""Feature-level goal sets from Appendix B.1 of the Dual-CRL-PPO paper."""

import numpy as np
import jax
import jax.numpy as jnp

from craftax.craftax_classic.constants import BlockType
from src.envs.craftax.craftax_goals import (
    CRAFTAX_CLASSIC_OBS_SHAPE,
    CRAFTAX_OBS_SHAPE,
    goal_achieved,
    sample_goal,
)


def observation_shape(env_name):
    if env_name == "Craftax-Classic-Symbolic-v1":
        return CRAFTAX_CLASSIC_OBS_SHAPE
    if env_name == "Craftax-Symbolic-v1":
        return CRAFTAX_OBS_SHAPE
    raise ValueError(f"Unsupported observation-level goal environment {env_name}")


def flatten_observation(obs, env_name, batched):
    """Flatten in the source project's explicit symbolic feature order."""
    keys = observation_shape(env_name).key_order
    if batched:
        return jnp.concatenate([obs[key].reshape((obs[key].shape[0], -1)) for key in keys], axis=-1)
    return jnp.concatenate([obs[key].reshape((-1,)) for key in keys], axis=-1)


def get_all_observation_goals(env_name, _):
    shape = observation_shape(env_name)
    sizes = {key: tuple(int(value) for value in shape.dict_shape[key]) for key in shape.key_order}

    # Paper Appendix B.1: keep all 9,637 Full Craftax features. Classic has
    # 1,562 features before excluding impossible predicates. INVALID and WOOD
    # map tiles never occur, the player cannot occupy a solid, out-of-bounds,
    # or terminal lava tile, cannot share its cell with a mob, and an observed
    # living player cannot have zero health. The remaining count is 1,419.
    active = np.ones(sum(int(np.prod(size)) for size in sizes.values()), dtype=bool)
    if env_name == "Craftax-Classic-Symbolic-v1":
        offset = 0
        for key in shape.key_order:
            size = sizes[key]
            section = active[offset : offset + int(np.prod(size))].reshape(size)
            if key == "block_map":
                section[:, :, [BlockType.INVALID.value, BlockType.WOOD.value]] = False
                walkable_center = {
                    BlockType.GRASS.value,
                    BlockType.PATH.value,
                    BlockType.SAND.value,
                }
                for block in BlockType:
                    if block.value not in walkable_center:
                        section[size[0] // 2, size[1] // 2, block.value] = False
            elif key == "mob_map":
                section[size[0] // 2, size[1] // 2, :4] = False
            elif key == "intrinsics":
                section[0, 0] = False  # health == 0 is followed by auto-reset
            offset += int(np.prod(size))

    indices = np.flatnonzero(active).astype(np.int32)
    expected = 1419 if env_name == "Craftax-Classic-Symbolic-v1" else 9637
    assert len(indices) == expected

    names = []
    offset = 0
    for key in shape.key_order:
        size = sizes[key]
        for coordinates in np.ndindex(size):
            feature = offset + np.ravel_multi_index(coordinates, size)
            if active[feature]:
                names.append(f"{key}/{','.join(map(str, coordinates))}")
        offset += int(np.prod(size))
    return tuple(names), jnp.asarray(indices)


def goal_indexes_to_goals(all_goals, goal_indexes, env_name):
    # Feature-level change: recreate a one-hot goal only for commanded goals.
    # Dense (all_goals, all_features) masks would be quadratic in goal count.
    shape = observation_shape(env_name)
    prefix = goal_indexes.shape
    total_features = sum(int(np.prod(shape.dict_shape[key])) for key in shape.key_order)
    flat_goals = jax.nn.one_hot(all_goals[goal_indexes], total_features)
    goals = {}
    offset = 0
    for key in shape.key_order:
        feature_shape = tuple(int(value) for value in shape.dict_shape[key])
        length = int(np.prod(feature_shape))
        goals[key] = flat_goals[..., offset : offset + length].reshape(prefix + feature_shape)
        offset += length
    return goals


def achieved_all(obs_batch, all_goals, env_name):
    # Each observation feature is itself a predicate. Gather active features
    # directly instead of vmapping a dictionary-mask comparison over goals.
    return flatten_observation(obs_batch, env_name, batched=True)[:, all_goals] > 0


def get_goals_seen(obs_batch, all_goals, env_name):
    return achieved_all(obs_batch, all_goals, env_name).any(axis=0)


def sample_positive_goal_index_from_obs(rng, obs, all_goals, env_name):
    positive = flatten_observation(obs, env_name, batched=False)[all_goals] > 0
    return jax.random.choice(rng, len(all_goals), p=positive)


def goal_to_goal_index(all_goals, goal, env_name):
    flat_goal = flatten_observation(goal, env_name, batched=False)
    return jnp.argmax(flat_goal[all_goals])
