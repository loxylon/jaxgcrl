"""Rao-Blackwellized goal targets for Dual-CRL-PPO (paper, equations 6–8)."""

import jax
import jax.numpy as jnp


def make_crl_targets(achieved, done, gamma, offset_cap, termination_discount=None):
    """Return goal plus absorbing-class targets of shape (time, env, goals + 1).

    ``achieved[t]`` describes the observation returned after action t. A terminal
    observation is unavailable after an automatic reset, so ``done[t]`` masks it.
    """
    time_steps, num_envs, num_goals = achieved.shape
    max_offset = min(time_steps, offset_cap)
    dtype = jnp.float32

    # CRL change: split each visited state's mass over every semantic goal it
    # satisfies (equation 4), instead of selecting one positive future goal.
    counts = achieved.sum(axis=-1, keepdims=True)
    soft_goals = achieved.astype(dtype) / jnp.maximum(counts, 1)
    padded_goals = jnp.pad(soft_goals, ((0, max_offset), (0, 0), (0, 0)))
    padded_done = jnp.pad(done, ((0, max_offset), (0, 0)), constant_values=True)
    valid_time = jnp.arange(time_steps)[:, None]

    def add_offset(offset, carry):
        goal_mass, weight_sum, alive = carry
        next_done = jax.lax.dynamic_slice_in_dim(padded_done, offset, time_steps)
        alive = alive & ~next_done & (valid_time + offset < time_steps)
        weight = (1.0 - gamma) * gamma**offset * alive
        future_goals = jax.lax.dynamic_slice_in_dim(padded_goals, offset, time_steps)
        goal_mass = goal_mass + weight[:, :, None] * future_goals
        weight_sum = weight_sum + weight
        return goal_mass, weight_sum, alive

    # CRL change: sum all available discounted future offsets up to k_max,
    # renormalizing their weights as in equation 6.
    initial = (
        jnp.zeros((time_steps, num_envs, num_goals), dtype),
        jnp.zeros((time_steps, num_envs), dtype),
        jnp.ones((time_steps, num_envs), dtype=bool),
    )
    goal_mass, weight_sum, _ = jax.lax.fori_loop(
        0, max_offset, add_offset, initial
    )
    goal_targets = goal_mass / jnp.maximum(weight_sum[:, :, None], 1e-8)

    if termination_discount is not None:
        # CRL change: equation 8 uses distance to the episode's observed end.
        # Scan the full rollout even when k_max is shorter than that distance.
        def distance_to_end(next_distance, ended):
            distance = jnp.where(ended, 1, jnp.where(next_distance > 0, next_distance + 1, 0))
            return distance, distance

        _, distances = jax.lax.scan(
            distance_to_end,
            jnp.zeros((num_envs,), dtype=jnp.int32),
            done,
            reverse=True,
        )
        survival_factor = jnp.where(
            distances > 0, 1.0 - termination_discount**distances, 1.0
        )
        goal_targets = goal_targets * survival_factor[:, :, None]

    # CRL change: a fixed-zero-logit absorbing class receives all mass absent
    # from real goals, including fatal transitions and states satisfying no goal.
    absorbing = 1.0 - goal_targets.sum(axis=-1, keepdims=True)
    return jnp.concatenate((goal_targets, absorbing), axis=-1)
