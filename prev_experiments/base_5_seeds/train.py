from src.make_trains import make_train_several_seeds
import argparse
import numpy as np
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="Craftax-Symbolic-v1")
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument(
        "--total_timesteps", type=lambda x: int(float(x)), default=int(1e9)
    )
    parser.add_argument("--num_steps", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--no_use_wandb", dest="use_wandb", action="store_false")
    parser.add_argument("--save_policy", action="store_true")
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument(
        "--no_use_optimistic_resets", dest="use_optimistic_resets", action="store_false"
    )
    parser.add_argument("--optimistic_reset_ratio", type=int, default=16)

    # GC
    # Observation-level change: the feature-index sets are separate from the
    # original 136/512 semantic goals, with the same Craftax environments.
    parser.add_argument(
        "--goal_set", choices=["semantic", "observation"], default="semantic"
    )
    parser.add_argument("--goal_reward_scale", type=float, default=1.0)
    parser.add_argument(
        "--no_only_sample_from_seen_goals",
        dest="only_sample_from_seen_goals",
        action="store_false",
    )
    parser.add_argument("--live_success_rate_decay", type=float, default=0.999)

    # PPO
    parser.add_argument("--ppo_lr", type=float, default=2e-4)
    parser.add_argument("--ppo_num_epochs", type=int, default=1)
    parser.add_argument("--ppo_minibatch_size", type=int, default=2048)
    parser.add_argument("--ppo_clip_eps", type=float, default=0.2)
    parser.add_argument("--ppo_ent_coef", type=float, default=0.005)
    parser.add_argument("--ppo_vf_coef", type=float, default=0.5)
    parser.add_argument("--ppo_gae_lambda", type=float, default=0.95)
    parser.add_argument(
        "--no_ppo_anneal_lr", dest="ppo_anneal_lr", action="store_false"
    )
    parser.add_argument("--seed_spread_num", type=int, default=1)

    # PPO network
    parser.add_argument(
        "--network_type",
        type=str,
        default="symbolic_conv",
        choices=["symbolic_conv", "symbolic_flat"],
    )
    parser.add_argument("--network_layer_width", type=int, default=1024)
    parser.add_argument("--ppo_conv_layers", type=int, default=1)
    parser.add_argument("--ppo_conv_features", type=int, default=32)
    parser.add_argument("--ppo_conv_kernel_size", type=int, default=3)
    parser.add_argument("--ppo_num_embedding_layers", type=int, default=1)
    parser.add_argument("--ppo_num_actor_layers", type=int, default=2)
    parser.add_argument("--ppo_num_critic_layers", type=int, default=4)
    parser.add_argument(
        "--no_ppo_network_sigmoid_value",
        dest="ppo_network_sigmoid_value",
        action="store_false",
    )

    # CRL
    parser.add_argument("--crl_lr", type=float, default=2e-4)
    parser.add_argument("--crl_num_epochs", type=int, default=1)
    parser.add_argument("--crl_minibatch_size", type=int, default=2048)
    parser.add_argument(
        "--no_crl_anneal_lr", dest="crl_anneal_lr", action="store_false"
    )
    parser.add_argument("--crl_representation_dim", type=int, default=64)
    parser.add_argument("--crl_offset_cap", type=int, default=None)
    parser.add_argument("--crl_termination_discount", type=float, default=None)

    # CRL network
    parser.add_argument("--crl_dense_layers", type=int, default=4)
    parser.add_argument("--crl_conv_layers", type=int, default=1)
    parser.add_argument("--crl_conv_features", type=int, default=16)
    parser.add_argument("--crl_conv_kernel_size", type=int, default=3)
    parser.add_argument(
        "--no_crl_norm_input", dest="crl_norm_input", action="store_false"
    )
    parser.add_argument("--crl_norm_type", type=str, default="layer_norm")

    # BC
    parser.add_argument("--bc_coef_policy", type=float, default=0.1)
    parser.add_argument(
        "--bc_policy_target", type=str, default="argmax", choices=["argmax", "softmax"]
    )
    parser.add_argument("--bc_policy_temp", type=float, default=1.0)
    # CRL change: equation 9 anneals the policy guidance to zero.
    parser.add_argument("--no_anneal_bc", dest="anneal_bc", action="store_false")

    # Gridworld
    parser.add_argument("--gridworld_map_size", type=int, default=16)
    parser.add_argument("--gridworld_map_rng_seed", type=int, default=0)
    parser.add_argument("--gridworld_wall_threshold", type=float, default=0.3)
    parser.add_argument("--run_name", type=str)

    args, rest_args = parser.parse_known_args(sys.argv[1:])
    if rest_args:
        raise ValueError(f"Unknown args {rest_args}")

    if args.seed is None:
        args.seed = np.random.randint(2**31)

    #TODO: Сделать сохранение конфига джобы - какие модели, какой версии, какой make_train и какой версии
    #TODO: Сделать скрипт для автоматиеского переноса имеющегося train.py, awl-job-preset, awl_run в отдельную папку с названием <run_name> внутрь prev_experiments

    make_train_several_seeds.run_dual_crl_ppo(args)
