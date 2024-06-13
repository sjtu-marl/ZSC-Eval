import argparse
import getpass


def scientific_notation(value):
    return int(float(value))


def get_config() -> argparse.ArgumentParser:
    """
    The configuration parser for common hyperparameters of all environment.
    Please reach each `scripts/train/<env>_runner.py` file to find private hyperparameters
    """
    parser = argparse.ArgumentParser(description="zsceval", formatter_class=argparse.RawDescriptionHelpFormatter)

    # prepare parameters
    # MARK: algo name
    parser.add_argument(
        "--algorithm_name",
        type=str,
        default="mappo",
        choices=[
            "rmappo",
            "mappo",
            "population",  # Qs: what is this?
            "mep",
            "traj",
            "adaptive",
            "cole",
            "e3t",
        ],
    )

    parser.add_argument(
        "--experiment_name",
        type=str,
        default="check",
        help="an identifier to distinguish different experiment.",
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed for numpy/torch")
    parser.add_argument(
        "--cuda",
        action="store_false",
        default=True,
        help="by default True, will use GPU to train; or else will use CPU;",
    )
    parser.add_argument(
        "--cuda_deterministic",
        action="store_false",
        default=True,
        help="by default, make sure random seed effective. if set, bypass such function.",
    )
    parser.add_argument(
        "--n_training_threads",
        type=int,
        default=1,
        help="Number of torch threads for training",
    )
    parser.add_argument(
        "--n_rollout_threads",
        type=int,
        default=32,
        help="Number of parallel envs for training rollout",
    )
    parser.add_argument(
        "--dummy_batch_size",
        type=int,
        default=5,
        help="Number of parallel envs in a dummy batch",
    )
    parser.add_argument(
        "--n_eval_rollout_threads",
        type=int,
        default=1,
        help="Number of parallel envs for evaluating rollout",
    )
    parser.add_argument(
        "--n_render_rollout_threads",
        type=int,
        default=1,
        help="Number of parallel envs for rendering rollout",
    )
    parser.add_argument(
        "--num_env_steps",
        type=scientific_notation,
        default=10e6,
        help="Number of environment steps to train (default: 10e6)",
    )
    parser.add_argument(
        "--user_name",
        type=str,
        default=getpass.getuser(),
        help="[for wandb usage], to specify user's name for simply collecting training data.",
    )
    parser.add_argument(
        "--wandb_name",
        type=str,
        default="wandb_name",
        help="[for wandb usage], to specify user's name for simply collecting training data.",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_false",
        default=True,
        help="[for wandb usage], by default True, will log date to wandb server. or else will use tensorboard to log data.",
    )

    # env parameters
    parser.add_argument(
        "--env_name",
        type=str,
        default="Overcooked",
        help="specify the name of environment",
        choices=["Overcooked", "GRF"],
    )
    parser.add_argument(
        "--use_obs_instead_of_state",
        action="store_true",
        default=False,
        help="Whether to use global state or concatenated obs",
    )
    parser.add_argument(
        "--use_available_actions",
        action="store_false",
        default=True,
        help="Whether to use available actions",
    )

    # replay buffer parameters
    parser.add_argument("--episode_length", type=int, default=200, help="Max length for any episode")

    # network parameters
    parser.add_argument(
        "--share_policy",
        action="store_false",
        default=True,
        help="Whether agent share the same policy",
    )
    parser.add_argument(
        "--use_centralized_V",
        action="store_false",
        default=True,
        help="Whether to use centralized V function",
    )
    parser.add_argument("--use_conv1d", action="store_true", default=False, help="Whether to use conv1d")
    parser.add_argument(
        "--stacked_frames",
        type=int,
        default=1,
        help="Dimension of hidden layers for actor/critic networks",
    )
    parser.add_argument(
        "--use_stacked_frames",
        action="store_true",
        default=False,
        help="Whether to use stacked_frames",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=64,
        help="Dimension of hidden layers for actor/critic networks",
    )
    parser.add_argument(
        "--mlp_hidden_size",
        type=int,
        default=64,
        help="Dimension of mlp hidden layers for actor/critic networks",
    )
    parser.add_argument(
        "--layer_N",
        type=int,
        default=1,
        help="Number of layers for actor/critic networks",
    )
    parser.add_argument(
        "--layer_after_N",
        type=int,
        default=0,
        help="Number of layers for actor/critic networks after rnn",
    )
    parser.add_argument(
        "--activation_id",
        type=int,
        default=1,
        help="choose 0 to use tanh, 1 to use relu, 2 to use leaky relu, 3 to use elu",
    )
    parser.add_argument(
        "--use_popart",
        action="store_true",
        default=False,
        help="by default False, use PopArt to normalize rewards.",
    )
    parser.add_argument(
        "--use_valuenorm",
        action="store_false",
        default=True,
        help="by default True, use running mean and std to normalize rewards.",
    )
    parser.add_argument(
        "--use_feature_normalization",
        action="store_false",
        default=True,
        help="Whether to apply layernorm to the inputs",
    )
    parser.add_argument(
        "--use_orthogonal",
        action="store_false",
        default=True,
        help="Whether to use Orthogonal initialization for weights and 0 initialization for biases",
    )
    parser.add_argument("--gain", type=float, default=0.01, help="The gain # of last action layer")
    parser.add_argument(
        "--cnn_layers_params",
        type=str,
        default=None,
        help="The parameters of cnn layer",
    )
    parser.add_argument(
        "--use_maxpool2d",
        action="store_true",
        default=False,
        help="Whether to apply layernorm to the inputs",
    )

    # recurrent parameters
    parser.add_argument(
        "--use_naive_recurrent_policy",
        action="store_true",
        default=False,
        help="Whether to use a naive recurrent policy",
    )
    parser.add_argument(
        "--use_recurrent_policy",
        action="store_false",
        default=True,
        help="use a recurrent policy",
    )
    parser.add_argument("--recurrent_N", type=int, default=1, help="The number of recurrent layers.")
    parser.add_argument(
        "--data_chunk_length",
        type=int,
        default=10,
        help="Time length of chunks used to train a recurrent_policy",
    )
    parser.add_argument(
        "--use_influence_policy",
        action="store_true",
        default=False,
        help="use a influence policy",
    )
    parser.add_argument(
        "--influence_layer_N",
        type=int,
        default=1,
        help="Number of layers for actor/critic networks",
    )

    # attn parameters
    parser.add_argument(
        "--use_attn",
        action="store_true",
        default=False,
        help=" by default False, use attention tactics.",
    )
    parser.add_argument("--attn_N", type=int, default=1, help="the number of attn layers, by default 1")
    parser.add_argument(
        "--attn_size",
        type=int,
        default=64,
        help="by default, the hidden size of attn layer",
    )
    parser.add_argument("--attn_heads", type=int, default=4, help="by default, the # of multiply heads")
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="by default 0, the dropout ratio of attn layer.",
    )
    parser.add_argument(
        "--use_average_pool",
        action="store_false",
        default=True,
        help="by default True, use average pooling for attn model.",
    )
    parser.add_argument(
        "--use_attn_internal",
        action="store_false",
        default=True,
        help="by default True, whether to strengthen own characteristics",
    )
    parser.add_argument(
        "--use_cat_self",
        action="store_false",
        default=True,
        help="by default True, whether to strengthen own characteristics",
    )

    # optimizer parameters
    parser.add_argument("--lr", type=float, default=5e-4, help="learning rate (default: 5e-4)")
    parser.add_argument("--tau", type=float, default=0.995, help="soft update polyak (default: 0.995)")
    parser.add_argument(
        "--critic_lr",
        type=float,
        default=5e-4,
        help="critic learning rate (default: 5e-4)",
    )
    parser.add_argument(
        "--opti_eps",
        type=float,
        default=1e-5,
        help="RMSprop optimizer epsilon (default: 1e-5)",
    )
    parser.add_argument("--weight_decay", type=float, default=0)

    # ppo parameters
    parser.add_argument("--ppo_epoch", type=int, default=15, help="number of ppo epochs (default: 15)")
    parser.add_argument(
        "--use_policy_vhead",
        action="store_true",
        default=False,
        help="by default, do not use policy vhead. if set, use policy vhead.",
    )
    parser.add_argument(
        "--use_clipped_value_loss",
        action="store_false",
        default=True,
        help="by default, clip loss value. If set, do not clip loss value.",
    )
    parser.add_argument(
        "--clip_param",
        type=float,
        default=0.2,
        help="ppo clip parameter (default: 0.2)",
    )
    parser.add_argument(
        "--num_mini_batch",
        type=int,
        default=1,
        help="number of batches for ppo (default: 1)",
    )
    parser.add_argument(
        "--policy_value_loss_coef",
        type=float,
        default=1,
        help="policy value loss coefficient (default: 0.5)",
    )

    parser.add_argument(
        "--entropy_coefs",
        type=float,
        nargs="+",
        default=[0.01, 0.01],
        help="entropy term coefficient (default: 0.01)",
    )
    parser.add_argument(
        "--entropy_coef_horizons",
        type=scientific_notation,
        nargs="+",
        default=[0, 1e7],
    )
    parser.add_argument(
        "--value_loss_coef",
        type=float,
        default=1,
        help="value loss coefficient (default: 0.5)",
    )
    parser.add_argument(
        "--use_max_grad_norm",
        action="store_false",
        default=True,
        help="by default, use max norm of gradients. If set, do not use.",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=10.0,
        help="max norm of gradients (default: 0.5)",
    )
    parser.add_argument(
        "--use_gae",
        action="store_false",
        default=True,
        help="use generalized advantage estimation",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="discount factor for rewards (default: 0.99)",
    )
    parser.add_argument(
        "--gae_lambda",
        type=float,
        default=0.95,
        help="gae lambda parameter (default: 0.95)",
    )
    parser.add_argument(
        "--use_proper_time_limits",
        action="store_true",
        default=False,
        help="compute returns taking into account time limits",
    )
    parser.add_argument(
        "--use_peb",
        action="store_true",
        default=False,
        help="partial episode bootstrapping",
    )
    parser.add_argument(
        "--use_huber_loss",
        action="store_false",
        default=True,
        help="by default, use huber loss. If set, do not use huber loss.",
    )
    parser.add_argument(
        "--use_value_active_masks",
        action="store_false",
        default=True,
        help="by default True, whether to mask useless data in value loss.",
    )
    parser.add_argument(
        "--use_policy_active_masks",
        action="store_false",
        default=True,
        help="by default True, whether to mask useless data in policy loss.",
    )
    parser.add_argument("--huber_delta", type=float, default=10.0, help=" coefficience of huber loss.")
    parser.add_argument("--num_v_out", default=1, type=int, help="number of value heads in critic")

    parser.add_argument(
        "--use_single_network",
        action="store_true",
        default=False,
        help="Whether to use centralized V function",
    )

    # run parameters
    parser.add_argument(
        "--use_linear_lr_decay",
        action="store_true",
        default=False,
        help="use a linear schedule on the learning rate",
    )
    # save parameters
    parser.add_argument(
        "--save_interval",
        type=int,
        default=1,
        help="time duration between contiunous twice models saving.",
    )

    # log parameters
    parser.add_argument(
        "--log_interval",
        type=int,
        default=5,
        help="time duration between contiunous twice log printing.",
    )

    # eval parameters
    parser.add_argument(
        "--use_eval",
        action="store_true",
        default=False,
        help="by default, do not start evaluation. If set`, start evaluation alongside with training.",
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=25,
        help="time duration between contiunous twice evaluation progress.",
    )
    parser.add_argument(
        "--eval_episodes",
        type=int,
        default=32,
        help="number of episodes of a single evaluation.",
    )
    parser.add_argument(
        "--eval_stochastic",
        action="store_true",
        default=False,
        help="use stochastic policy when eval",
    )

    # render parameters
    parser.add_argument(
        "--save_gifs",
        action="store_true",
        default=False,
        help="by default, do not save render video. If set, save video.",
    )
    parser.add_argument(
        "--use_render",
        action="store_true",
        default=False,
        help="by default, do not render the env during training. If set, start render. Note: something, the environment has internal render process which is not controlled by this hyperparam.",
    )
    parser.add_argument(
        "--render_episodes",
        type=int,
        default=5,
        help="the number of episodes to render a given env",
    )
    parser.add_argument(
        "--ifi",
        type=float,
        default=0.1,
        help="the play interval of each rendered image in saved video.",
    )

    # pretrained parameters
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="by default None. set the path to pretrained model.",
    )
    parser.add_argument(
        "--critic_warmup_horizon",
        type=int,
        default=0,
        help="by default 0. The horizon to warm up critic.",
    )

    # wandb
    parser.add_argument("--wandb_tags", nargs="+", help="wandb tags to your experiment", default=[])

    # data parallel
    parser.add_argument(
        "--data_parallel",
        action="store_true",
        default=False,
        help="Use dataparallel in pytorch",
    )

    # method type
    parser.add_argument(
        "--algorithm_type",
        type=str,
        choices=["co-play", "evolution"],
        default="co-play",
    )

    return parser
