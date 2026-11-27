import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from flax.linen.initializers import constant, orthogonal
from typing import Sequence

from src.models.batch_renorm import BatchRenorm


class CRLNetworkConvSymbolicCraftax(nn.Module):
    action_dim: int
    num_goals: int

    dense_hidden_size: int = 512
    dense_layers: int = 4

    conv_layers: int = 2
    conv_features: int = 32
    conv_kernel_size: int = 2

    norm_type: str = "batch_norm"
    norm_input: bool = False

    representation_dim: int = 64

    @nn.compact
    def __call__(self, obs, train: bool, action=None, goal_index=None):
        batch_size = obs["block_map"].shape[0]

        def _split_obs(obs):

            image_obs = jnp.concatenate([obs[k] for k in obs if "map" in k], axis=-1)

            flat_obs = jnp.concatenate(
                [obs[k].reshape((batch_size, -1)) for k in obs if "map" not in k],
                axis=-1,
            )

            return image_obs, flat_obs

        image_embedding, flat_embedding = _split_obs(obs)

        if self.norm_input:
            image_embedding = BatchRenorm(use_running_average=not train)(
                image_embedding
            )
            flat_embedding = BatchRenorm(use_running_average=not train)(flat_embedding)
        else:
            raise ValueError

        if self.norm_type == "layer_norm":

            def normalize(x):
                return nn.LayerNorm()(x)
        elif self.norm_type == "batch_norm":

            def normalize(x):
                return BatchRenorm(use_running_average=not train)(x)
        else:

            def normalize(x):
                return x

        # Convolutions on map
        for _ in range(self.conv_layers):
            image_embedding = nn.Conv(
                features=self.conv_features,
                kernel_size=(self.conv_kernel_size, self.conv_kernel_size),
            )(image_embedding)
            image_embedding = normalize(image_embedding)
            image_embedding = nn.relu(image_embedding)

        image_embedding = image_embedding.reshape(image_embedding.shape[0], -1)

        # Combine embeddings
        embedding = jnp.concatenate([image_embedding, flat_embedding], axis=-1)

        for _ in range(self.dense_layers):
            embedding = nn.Dense(
                self.dense_hidden_size,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(embedding)
            embedding = normalize(embedding)
            embedding = nn.relu(embedding)

        # CRL change: replace the per-goal Q head with state-action features and a
        # learned goal embedding. Their dot product is the paper's f(s, a, g).
        state_action_features = nn.Dense(
            self.action_dim * self.representation_dim,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )(embedding).reshape((batch_size, self.action_dim, self.representation_dim))
        goal_features = self.param(
            "goal_features", nn.initializers.normal(0.02),
            (self.num_goals, self.representation_dim),
        )
        # CRL scaling change: score only the taken action during critic updates
        # or the commanded goal during rollout. This avoids storing a dense
        # (time, environments, goals, actions) tensor for large goal sets.
        if action is not None:
            chosen_features = state_action_features[jnp.arange(batch_size), action]
            return jnp.einsum("br,gr->bg", chosen_features, goal_features)
        if goal_index is not None:
            chosen_goals = goal_features[goal_index]
            return jnp.einsum("bar,br->ba", state_action_features, chosen_goals)
        return jnp.einsum("bar,gr->bga", state_action_features, goal_features)


class CRLNetworkFlat(nn.Module):
    action_dim: int
    num_goals: int

    dense_hidden_size: int = 512
    dense_layers: int = 4

    norm_type: str = "batch_norm"
    norm_input: bool = False

    representation_dim: int = 64

    @nn.compact
    def __call__(self, obs, train: bool, action=None, goal_index=None):
        batch_size = obs.shape[0]

        # Flatten and concatenate obs and goal
        obs = jax.tree.map(lambda x: jnp.reshape(x, (batch_size, -1)), obs)
        embedding = jax.tree.reduce(
            lambda x, y: jnp.concatenate([x, y], axis=1),
            obs,
            jnp.zeros((batch_size, 0)),
        )

        if self.norm_input:
            embedding = BatchRenorm(use_running_average=not train)(embedding)
        else:
            raise ValueError

        if self.norm_type == "layer_norm":

            def normalize(x):
                return nn.LayerNorm()(x)
        elif self.norm_type == "batch_norm":

            def normalize(x):
                return BatchRenorm(use_running_average=not train)(x)
        else:

            def normalize(x):
                return x

        for _ in range(self.dense_layers):
            embedding = nn.Dense(
                self.dense_hidden_size,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(embedding)
            embedding = normalize(embedding)
            embedding = nn.relu(embedding)

        # CRL change: replace the per-goal Q head with state-action features and a
        # learned goal embedding. Their dot product is the paper's f(s, a, g).
        state_action_features = nn.Dense(
            self.action_dim * self.representation_dim,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )(embedding).reshape((batch_size, self.action_dim, self.representation_dim))
        goal_features = self.param(
            "goal_features", nn.initializers.normal(0.02),
            (self.num_goals, self.representation_dim),
        )
        # CRL scaling change: score only the taken action during critic updates
        # or the commanded goal during rollout. This avoids storing a dense
        # (time, environments, goals, actions) tensor for large goal sets.
        if action is not None:
            chosen_features = state_action_features[jnp.arange(batch_size), action]
            return jnp.einsum("br,gr->bg", chosen_features, goal_features)
        if goal_index is not None:
            chosen_goals = goal_features[goal_index]
            return jnp.einsum("bar,br->ba", state_action_features, chosen_goals)
        return jnp.einsum("bar,gr->bga", state_action_features, goal_features)
