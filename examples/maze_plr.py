import json
import time
from pathlib import Path
from typing import Sequence, Tuple
import numpy as np
import jax
import jax.numpy as jnp
from flax import core, struct
from flax.training.train_state import TrainState as BaseTrainState
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import optax
import distrax
import os
import orbax.checkpoint as ocp
import wandb
from jaxued.environments.underspecified_env import EnvParams, EnvState, Observation, UnderspecifiedEnv
from jaxued.linen import ResetRNN
from jaxued.environments import Maze, MazeRenderer
from jaxued.environments.maze import Level, make_level_generator, make_level_mutator_minimax
from jaxued import adaptive_mutation, hybrid_teacher, learnability, map_elites, mutation_checkpoint
from jaxued import pace_score, teacher_experiments, teacher_parity
from jaxued.level_sampler import LevelSampler
from jaxued.utils import compute_max_returns, max_mc, positive_value_loss
from jaxued.wrappers import AutoReplayWrapper
import chex
from enum import IntEnum

class UpdateState(IntEnum):
    DR = 0
    REPLAY = 1

class TrainState(BaseTrainState):
    sampler: core.FrozenDict[str, chex.ArrayTree] = struct.field(pytree_node=True)
    update_state: UpdateState = struct.field(pytree_node=True)
    # === Below is used for logging ===
    num_dr_updates: int
    num_replay_updates: int
    num_mutation_updates: int
    dr_last_level_batch: chex.ArrayTree = struct.field(pytree_node=True)
    replay_last_level_batch: chex.ArrayTree = struct.field(pytree_node=True)
    mutation_last_level_batch: chex.ArrayTree = struct.field(pytree_node=True)

# region PPO helper functions
def compute_gae(
    gamma: float,
    lambd: float,
    last_value: chex.Array,
    values: chex.Array,
    rewards: chex.Array,
    dones: chex.Array,
) -> Tuple[chex.Array, chex.Array]:
    """This takes in arrays of shape (NUM_STEPS, NUM_ENVS) and returns the advantages and targets.

    Args:
        gamma (float): 
        lambd (float): 
        last_value (chex.Array):  Shape (NUM_ENVS)
        values (chex.Array): Shape (NUM_STEPS, NUM_ENVS)
        rewards (chex.Array): Shape (NUM_STEPS, NUM_ENVS)
        dones (chex.Array): Shape (NUM_STEPS, NUM_ENVS)

    Returns:
        Tuple[chex.Array, chex.Array]: advantages, targets; each of shape (NUM_STEPS, NUM_ENVS)
    """
    def compute_gae_at_timestep(carry, x):
        gae, next_value = carry
        value, reward, done = x
        delta = reward + gamma * next_value * (1 - done) - value
        gae = delta + gamma * lambd * (1 - done) * gae
        return (gae, value), gae

    _, advantages = jax.lax.scan(
        compute_gae_at_timestep,
        (jnp.zeros_like(last_value), last_value),
        (values, rewards, dones),
        reverse=True,
        unroll=16,
    )
    return advantages, advantages + values

def sample_trajectories_rnn(
    rng: chex.PRNGKey,
    env: UnderspecifiedEnv,
    env_params: EnvParams,
    train_state: TrainState,
    init_hstate: chex.ArrayTree,
    init_obs: Observation,
    init_env_state: EnvState,
    num_envs: int,
    max_episode_length: int,
) -> Tuple[Tuple[chex.PRNGKey, TrainState, chex.ArrayTree, Observation, EnvState, chex.Array], Tuple[Observation, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, dict]]:
    """This samples trajectories from the environment using the agent specified by the `train_state`.

    Args:

        rng (chex.PRNGKey): Singleton 
        env (UnderspecifiedEnv): 
        env_params (EnvParams): 
        train_state (TrainState): Singleton
        init_hstate (chex.ArrayTree): This is the init RNN hidden state, has to have shape (NUM_ENVS, ...)
        init_obs (Observation): The initial observation, shape (NUM_ENVS, ...)
        init_env_state (EnvState): The initial env state (NUM_ENVS, ...)
        num_envs (int): The number of envs that are vmapped over.
        max_episode_length (int): The maximum episode length, i.e., the number of steps to do the rollouts for.

    Returns:
        Tuple[Tuple[chex.PRNGKey, TrainState, chex.ArrayTree, Observation, EnvState, chex.Array], Tuple[Observation, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, dict]]: (rng, train_state, hstate, last_obs, last_env_state, last_value), traj, where traj is (obs, action, reward, done, log_prob, value, info). The first element in the tuple consists of arrays that have shapes (NUM_ENVS, ...) (except `rng` and and `train_state` which are singleton). The second element in the tuple is of shape (NUM_STEPS, NUM_ENVS, ...), and it contains the trajectory.
    """
    def sample_step(carry, _):
        rng, train_state, hstate, obs, env_state, last_done = carry
        rng, rng_action, rng_step = jax.random.split(rng, 3)

        x = jax.tree_util.tree_map(lambda x: x[None, ...], (obs, last_done))
        hstate, pi, value = train_state.apply_fn(train_state.params, x, hstate)
        action = pi.sample(seed=rng_action)
        log_prob = pi.log_prob(action)
        value, action, log_prob = (
            value.squeeze(0),
            action.squeeze(0),
            log_prob.squeeze(0),
        )

        next_obs, env_state, reward, done, info = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(jax.random.split(rng_step, num_envs), env_state, action, env_params)

        carry = (rng, train_state, hstate, next_obs, env_state, done)
        return carry, (obs, action, reward, done, log_prob, value, info)

    (rng, train_state, hstate, last_obs, last_env_state, last_done), traj = jax.lax.scan(
        sample_step,
        (
            rng,
            train_state,
            init_hstate,
            init_obs,
            init_env_state,
            jnp.zeros(num_envs, dtype=bool),
        ),
        None,
        length=max_episode_length,
    )

    x = jax.tree_util.tree_map(lambda x: x[None, ...], (last_obs, last_done))
    _, _, last_value = train_state.apply_fn(train_state.params, x, hstate)

    return (rng, train_state, hstate, last_obs, last_env_state, last_value.squeeze(0)), traj

def evaluate_rnn(
    rng: chex.PRNGKey,
    env: UnderspecifiedEnv,
    env_params: EnvParams,
    train_state: TrainState,
    init_hstate: chex.ArrayTree,
    init_obs: Observation,
    init_env_state: EnvState,
    max_episode_length: int,
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """This runs the RNN on the environment, given an initial state and observation, and returns (states, rewards, episode_lengths)

    Args:
        rng (chex.PRNGKey): 
        env (UnderspecifiedEnv): 
        env_params (EnvParams): 
        train_state (TrainState): 
        init_hstate (chex.ArrayTree): Shape (num_levels, )
        init_obs (Observation): Shape (num_levels, )
        init_env_state (EnvState): Shape (num_levels, )
        max_episode_length (int): 

    Returns:
        Tuple[chex.Array, chex.Array, chex.Array]: (States, rewards, episode lengths) ((NUM_STEPS, NUM_LEVELS), (NUM_STEPS, NUM_LEVELS), (NUM_LEVELS,)
    """
    num_levels = jax.tree_util.tree_flatten(init_obs)[0][0].shape[0]
    
    def step(carry, _):
        rng, hstate, obs, state, done, mask, episode_length = carry
        rng, rng_action, rng_step = jax.random.split(rng, 3)

        x = jax.tree_util.tree_map(lambda x: x[None, ...], (obs, done))
        hstate, pi, _ = train_state.apply_fn(train_state.params, x, hstate)
        action = pi.sample(seed=rng_action).squeeze(0)

        obs, next_state, reward, done, _ = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(jax.random.split(rng_step, num_levels), state, action, env_params)
        
        next_mask = mask & ~done
        episode_length += mask

        return (rng, hstate, obs, next_state, done, next_mask, episode_length), (state, reward)
    
    (_, _, _, _, _, _, episode_lengths), (states, rewards) = jax.lax.scan(
        step,
        (
            rng,
            init_hstate,
            init_obs,
            init_env_state,
            jnp.zeros(num_levels, dtype=bool),
            jnp.ones(num_levels, dtype=bool),
            jnp.zeros(num_levels, dtype=jnp.int32),
        ),
        None,
        length=max_episode_length,
    )

    return states, rewards, episode_lengths

def update_actor_critic_rnn(
    rng: chex.PRNGKey,
    train_state: TrainState,
    init_hstate: chex.ArrayTree,
    batch: chex.ArrayTree,
    num_envs: int,
    n_steps: int,
    n_minibatch: int,
    n_epochs: int,
    clip_eps: float,
    entropy_coeff: float,
    critic_coeff: float,
    update_grad: bool=True,
) -> Tuple[Tuple[chex.PRNGKey, TrainState], chex.ArrayTree]:
    """This function takes in a rollout, and PPO hyperparameters, and updates the train state.

    Args:
        rng (chex.PRNGKey): 
        train_state (TrainState): 
        init_hstate (chex.ArrayTree): 
        batch (chex.ArrayTree): obs, actions, dones, log_probs, values, targets, advantages
        num_envs (int): 
        n_steps (int): 
        n_minibatch (int): 
        n_epochs (int): 
        clip_eps (float): 
        entropy_coeff (float): 
        critic_coeff (float): 
        update_grad (bool, optional): If False, the train state does not actually get updated. Defaults to True.

    Returns:
        Tuple[Tuple[chex.PRNGKey, TrainState], chex.ArrayTree]: It returns a new rng, the updated train_state, and the losses. The losses have structure (loss, (l_vf, l_clip, entropy))
    """
    obs, actions, dones, log_probs, values, targets, advantages = batch
    last_dones = jnp.roll(dones, 1, axis=0).at[0].set(False)
    batch = obs, actions, last_dones, log_probs, values, targets, advantages
    
    def update_epoch(carry, _):
        def update_minibatch(train_state, minibatch):
            init_hstate, obs, actions, last_dones, log_probs, values, targets, advantages = minibatch
            
            def loss_fn(params):
                _, pi, values_pred = train_state.apply_fn(params, (obs, last_dones), init_hstate)
                log_probs_pred = pi.log_prob(actions)
                entropy = pi.entropy().mean()

                ratio = jnp.exp(log_probs_pred - log_probs)
                A = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
                l_clip = (-jnp.minimum(ratio * A, jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * A)).mean()

                values_pred_clipped = values + (values_pred - values).clip(-clip_eps, clip_eps)
                l_vf = 0.5 * jnp.maximum((values_pred - targets) ** 2, (values_pred_clipped - targets) ** 2).mean()

                loss = l_clip + critic_coeff * l_vf - entropy_coeff * entropy

                return loss, (l_vf, l_clip, entropy)

            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            loss, grads = grad_fn(train_state.params)
            if update_grad:
                train_state = train_state.apply_gradients(grads=grads)
            return train_state, loss

        rng, train_state = carry
        rng, rng_perm = jax.random.split(rng)
        permutation = jax.random.permutation(rng_perm, num_envs)
        minibatches = (
            jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=0)
                .reshape(n_minibatch, -1, *x.shape[1:]),
                init_hstate,
            ),
            *jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=1)
                .reshape(x.shape[0], n_minibatch, -1, *x.shape[2:])
                .swapaxes(0, 1),
                batch,
            ),
        )
        train_state, losses = jax.lax.scan(update_minibatch, train_state, minibatches)
        return (rng, train_state), losses

    return jax.lax.scan(update_epoch, (rng, train_state), None, n_epochs)

class ActorCritic(nn.Module):
    """This is an actor critic class that uses an LSTM
    """
    action_dim: Sequence[int]
    
    @nn.compact
    def __call__(self, inputs, hidden):
        obs, dones = inputs
        
        img_embed = nn.Conv(16, kernel_size=(3, 3), strides=(1, 1), padding="VALID")(obs.image)
        img_embed = img_embed.reshape(*img_embed.shape[:-3], -1)
        img_embed = nn.relu(img_embed)
        
        dir_embed = jax.nn.one_hot(obs.agent_dir, 4)
        dir_embed = nn.Dense(5, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name="scalar_embed")(dir_embed)
        
        embedding = jnp.append(img_embed, dir_embed, axis=-1)

        hidden, embedding = ResetRNN(nn.OptimizedLSTMCell(features=256))((embedding, dones), initial_carry=hidden)

        actor_mean = nn.Dense(32, kernel_init=orthogonal(2), bias_init=constant(0.0), name="actor0")(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0), name="actor1")(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(32, kernel_init=orthogonal(2), bias_init=constant(0.0), name="critic0")(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0), name="critic1")(critic)

        return hidden, pi, jnp.squeeze(critic, axis=-1)
    
    @staticmethod
    def initialize_carry(batch_dims):
        return nn.OptimizedLSTMCell(features=256).initialize_carry(jax.random.PRNGKey(0), (*batch_dims, 256))
# endregion

# region checkpointing
def setup_checkpointing(config: dict, train_state: TrainState, env: UnderspecifiedEnv, env_params: EnvParams) -> ocp.CheckpointManager:
    """This takes in the train state and config, and returns an orbax checkpoint manager.
        It also saves the config in `checkpoints/run_name/seed/config.json`

    Args:
        config (dict): 
        train_state (TrainState): 
        env (UnderspecifiedEnv): 
        env_params (EnvParams): 

    Returns:
        ocp.CheckpointManager: 
    """
    overall_save_dir = os.path.join(os.getcwd(), "checkpoints", f"{config['run_name']}", str(config['seed']))
    os.makedirs(overall_save_dir, exist_ok=True)
    
    # save the config
    with open(os.path.join(overall_save_dir, 'config.json'), 'w+') as f:
        f.write(json.dumps(config.as_dict(), indent=True))
    
    checkpoint_manager = ocp.CheckpointManager(
        os.path.join(overall_save_dir, 'models'),
        options=ocp.CheckpointManagerOptions(
            save_interval_steps=config['checkpoint_save_interval'],
            max_to_keep=config['max_number_of_checkpoints'],
        )
    )
    return checkpoint_manager
#endregion

def train_state_to_log_dict(train_state: TrainState, level_sampler: LevelSampler, *, weight_sampler=None) -> dict:
    """To prevent the entire (large) train_state to be copied to the CPU when doing logging, this function returns all of the important information in a dictionary format.

        Anything in the `log` key will be logged to wandb.
    
    Args:
        train_state (TrainState): 
        level_sampler (LevelSampler): 

    Returns:
        dict: 
    """
    sampler = train_state.sampler
    idx = jnp.arange(level_sampler.capacity) < sampler["size"]
    s = jnp.maximum(idx.sum(), 1)
    valid_scores = idx & jnp.isfinite(sampler["scores"])
    scores = jnp.where(valid_scores, sampler["scores"], 0)
    weights = level_sampler.level_weights(sampler if weight_sampler is None else weight_sampler)
    valid_weights = idx & jnp.isfinite(weights)
    safe_weights = jnp.where(valid_weights, weights, 0)
    frontier = teacher_experiments.frontier_mask(sampler)
    log = {
        "level_sampler/size": sampler["size"],
        "level_sampler/episode_count": sampler["episode_count"],
        "level_sampler/max_score": jnp.where(
            valid_scores.any(), jnp.where(valid_scores, scores, -jnp.inf).max(), 0
        ),
        "level_sampler/weighted_score": (scores * safe_weights).sum(),
        "level_sampler/mean_score": scores.sum() / s,
        "level_sampler/nonfinite_active_scores": (idx & ~valid_scores).sum(),
        "level_sampler/nonfinite_active_weights": (idx & ~valid_weights).sum(),
        "level_sampler/frontier_fraction": frontier.sum() / s,
        "level_sampler/frontier_weight": jnp.where(frontier, safe_weights, 0).sum(),
        "level_sampler/weight_ess": jnp.where(
            idx.any(), 1 / jnp.maximum((safe_weights ** 2).sum(), 1e-20), 0
        ),
    }
    extra = sampler["levels_extra"]
    if "initialized" in extra:
        observed = idx & extra["initialized"]
        n = jnp.maximum(observed.sum(), 1)
        progress = jnp.maximum(0, extra["fast_ema"] - extra["slow_ema"])
        log.update({
            "har/levels_with_history": observed.sum(),
            "har/mean_fast_ema": jnp.where(observed, extra["fast_ema"], 0).sum() / n,
            "har/mean_slow_ema": jnp.where(observed, extra["slow_ema"], 0).sum() / n,
            "har/mean_learning_progress": jnp.where(observed, progress, 0).sum() / n,
        })
        bins = jnp.clip((scores * 10).astype(jnp.int32), 0, 9)
        histogram = jnp.bincount(bins, weights=valid_scores.astype(jnp.int32), length=10)
        log.update({f"har/score_bin_{i}": histogram[i] for i in range(10)})
    if "qd_archive" in sampler:
        last_round = train_state.num_dr_updates + train_state.num_replay_updates + train_state.num_mutation_updates - 1
        log.update(map_elites.archive_statistics(sampler["qd_archive"], last_round, sampler["qd_max_age"]))
    return {
        "log": log,
        "info": {
            "num_dr_updates": train_state.num_dr_updates,
            "num_replay_updates": train_state.num_replay_updates,
            "num_mutation_updates": train_state.num_mutation_updates,
        }
    }

def compute_score(config, dones, values, max_returns, advantages):
    if config['score_function'] == "MaxMC":
        return max_mc(dones, values, max_returns)
    elif config['score_function'] == "pvl":
        return positive_value_loss(dones, advantages)
    else:
        raise ValueError(f"Unknown score function: {config['score_function']}")

def main(config=None, project="JAXUED_TEST"):
    if config.get("debug_teacher_parity"):
        try:
            jax.local_devices(backend="cpu")
        except RuntimeError as error:
            raise ValueError("parity tracing needs the CPU backend for host callbacks (JAX_PLATFORMS=cuda,cpu on GPU)") from error
    resume_dir = config.get("resume_checkpoint_directory")
    if resume_dir:
        destination = Path("checkpoints") / str(config["run_name"]) / str(config["seed"])
        if config["mode"] != "train" or not config["run_name"] or destination.exists():
            raise ValueError("resume requires train mode and a new --run_name")
    tags = []
    if not config["exploratory_grad_updates"]:
        tags.append("robust")
    if config["use_accel"]:
        tags.append("ACCEL")
    else:
        tags.append("PLR")
    use_pace = config["score_function"] == "pace_pg"
    log_pace = use_pace and config.get("pace_log_diagnostics", False)
    if use_pace:
        if config["num_minibatches"] != 1:
            raise ValueError("pace_pg requires num_minibatches=1 for PPO advantage normalization")
        tags.append("PACE-PG")
    run = wandb.init(config=config, project=project, group=config["run_name"], tags=tags)
    config = wandb.config
    
    wandb.define_metric("num_updates")
    wandb.define_metric("num_env_steps")
    wandb.define_metric("solve_rate/*", step_metric="num_updates")
    wandb.define_metric("level_sampler/*", step_metric="num_updates")
    wandb.define_metric("replay/*", step_metric="num_updates")
    wandb.define_metric("har/*", step_metric="num_updates")
    wandb.define_metric("adaptive_mutation/*", step_metric="num_updates")
    wandb.define_metric("qd/*", step_metric="num_updates")
    wandb.define_metric("hybrid/*", step_metric="num_updates")
    if log_pace:
        wandb.define_metric("pace/*", step_metric="num_updates")
    wandb.define_metric("agent/*", step_metric="num_updates")
    wandb.define_metric("return/*", step_metric="num_updates")
    wandb.define_metric("eval_ep_lengths/*", step_metric="num_updates")

    def log_eval(stats, train_state_info):
        print(f"Logging update: {stats['update_count']}")
        
        # generic stats
        env_steps = stats["update_count"] * config["num_train_envs"] * config["num_steps"]
        log_dict = {
            "num_updates": stats["update_count"],
            "num_env_steps": env_steps,
            "sps": env_steps / stats['time_delta'],
            "num_mutation_updates": train_state_info["info"]["num_mutation_updates"],
        }
        
        # evaluation performance
        solve_rates = stats['eval_solve_rates']
        returns     = stats["eval_returns"]
        log_dict.update({f"solve_rate/{name}": solve_rate for name, solve_rate in zip(config["eval_levels"], solve_rates)})
        log_dict.update({"solve_rate/mean": solve_rates.mean()})
        log_dict.update({f"return/{name}": ret for name, ret in zip(config["eval_levels"], returns)})
        log_dict.update({"return/mean": returns.mean()})
        log_dict.update({"eval_ep_lengths/mean": stats['eval_ep_lengths'].mean()})
        
        # level sampler
        diagnostics = {**train_state_info["log"], **stats["replay"]}
        diagnostics.update(stats.get("adaptive_mutation", {}))
        diagnostics.update(stats.get("qd", {}))
        diagnostics.update(stats.get("pace", {}))
        log_dict.update(diagnostics)

        # images
        log_dict.update({"images/highest_scoring_level": wandb.Image(np.array(stats["highest_scoring_level"]), caption="Highest scoring level")})
        log_dict.update({"images/highest_weighted_level": wandb.Image(np.array(stats["highest_weighted_level"]), caption="Highest weighted level")})

        for s in ['dr', 'replay', 'mutation']:
            if train_state_info['info'][f'num_{s}_updates'] > 0:
                log_dict.update({f"images/{s}_levels": [wandb.Image(np.array(image)) for image in stats[f"{s}_levels"]]})

        # animations
        for i, level_name in enumerate(config["eval_levels"]):
            frames, episode_length = stats["eval_animation"][0][:, i], stats["eval_animation"][1][i]
            frames = np.array(frames[:episode_length])
            log_dict.update({f"animations/{level_name}": wandb.Video(frames, fps=4, format="gif")})
        
        wandb.log(log_dict)
        return diagnostics
    
    # Setup the environment
    env = Maze(max_height=13, max_width=13, agent_view_size=config["agent_view_size"], normalize_obs=True)
    eval_env = env
    sample_random_level = make_level_generator(env.max_height, env.max_width, config["n_walls"])
    env_renderer = MazeRenderer(env, tile_size=8)
    env = AutoReplayWrapper(env)
    env_params = env.default_params
    mutate_level = make_level_mutator_minimax(adaptive_mutation.MAX_EDITS)
    selection = config.get("mutation_selection", "fixed")
    arms = tuple(config.get("mutation_arms", adaptive_mutation.DEFAULT_ARMS))
    num_arms = len(arms)
    empty_mutation_metrics = jnp.zeros(5 + num_arms, dtype=jnp.float32)
    qd_mode = config.get("qd_mode", "off")
    parity_path = config.get("debug_teacher_parity")
    parity = teacher_parity.Trace(Path(parity_path)) if parity_path else None
    if qd_mode != "off":
        qd_bins = tuple(config["qd_bins"])
        wall_edges = jnp.asarray(config["qd_wall_edges"], dtype=jnp.float32)
        deadend_edges = jnp.asarray(config["qd_deadend_edges"], dtype=jnp.float32)
        qd_size = qd_bins[0] * qd_bins[1]
        empty_qd_metrics = jnp.zeros(6 + qd_size, dtype=jnp.float32)
        qd_max_age = config["qd_max_age"]

    # And the level sampler    
    level_sampler = LevelSampler(
        capacity=config["level_buffer_capacity"],
        replay_prob=config["replay_prob"],
        staleness_coeff=config["staleness_coeff"],
        minimum_fill_ratio=config["minimum_fill_ratio"],
        prioritization=config["prioritization"],
        prioritization_params={"temperature": config["temperature"], "k": config['topk_k']},
        duplicate_check=config['buffer_duplicate_check'],
        replay_max_copies=config.get("replay_max_copies", 0),
        replay_preserve_frontier_mass=config.get("replay_preserve_frontier_mass", False),
    )
    use_har = config["score_function"] == "history_aware_regret"
    dual_scores = config.get("dual_score_tracking", False)
    use_learnability = dual_scores or use_har or config["score_function"] == "recent_learnability"
    history_weight = config.get("har_history_weight", 0.2) if use_har else None
    rank_priorities = config["prioritization"] == "rank"
    
    @jax.jit
    def create_train_state(rng) -> TrainState:
        # Creates the train state
        def linear_schedule(count):
            frac = (
                1.0
                - (count // (config["num_minibatches"] * config["epoch_ppo"]))
                / config["num_updates"]
            )
            return config["lr"] * frac
        obs, _ = env.reset_to_level(rng, sample_random_level(rng), env_params)
        obs = jax.tree_util.tree_map(
            lambda x: jnp.repeat(jnp.repeat(x[None, ...], config["num_train_envs"], axis=0)[None, ...], 256, axis=0),
            obs,
        )
        init_x = (obs, jnp.zeros((256, config["num_train_envs"])))
        network = ActorCritic(env.action_space(env_params).n)
        network_params = network.init(rng, init_x, ActorCritic.initialize_carry((config["num_train_envs"],)))
        tx = optax.chain(
            optax.clip_by_global_norm(config["max_grad_norm"]),
            optax.adam(learning_rate=linear_schedule, eps=1e-5),
            # optax.adam(learning_rate=config["lr"], eps=1e-5),
        )
        pholder_level = sample_random_level(jax.random.PRNGKey(0))
        level_extra = learnability.empty_history(track_progress=use_har) if use_learnability else {"max_return": -jnp.inf}
        if dual_scores:
            level_extra = hybrid_teacher.empty_history()
        sampler = level_sampler.initialize(pholder_level, level_extra)
        if selection == "bandit":
            sampler["mutation_bandit"] = adaptive_mutation.initialize(num_arms)
        if qd_mode != "off":
            sampler["qd_archive"] = map_elites.initialize_archive(pholder_level, qd_bins)
            sampler["qd_max_age"] = jnp.int32(qd_max_age)
        pholder_level_batch = jax.tree_util.tree_map(lambda x: jnp.array([x]).repeat(config["num_train_envs"], axis=0), pholder_level)
        return TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
            sampler=sampler,
            update_state=0,
            num_dr_updates=0,
            num_replay_updates=0,
            num_mutation_updates=0,
            dr_last_level_batch=pholder_level_batch,
            replay_last_level_batch=pholder_level_batch,
            mutation_last_level_batch=pholder_level_batch,
        )

    def train_step(carry: Tuple[chex.PRNGKey, TrainState], _):
        """
            This is the main training loop. It basically calls either `on_new_levels`, `on_replay_levels`, or `on_mutate_levels` at every step.
        """
        rng, train_state = carry
        round_index = jnp.asarray(
            train_state.num_dr_updates + train_state.num_replay_updates + train_state.num_mutation_updates,
            dtype=jnp.int32,
        )
        if dual_scores:
            hybrid_weight = hybrid_teacher.config_coefficient(config, round_index)
            train_state = train_state.replace(sampler=hybrid_teacher.reweight(
                train_state.sampler, hybrid_weight, raw_rank=rank_priorities,
            ))

        def score_pace_rollout(train_state, obs, actions, dones, rewards, advantages):
            if not use_pace:
                return None, {}
            scores = pace_score.compute_pace_pg_scores(
                train_state.apply_fn, train_state.params, obs, actions, dones, advantages,
                ActorCritic.initialize_carry((actions.shape[1],)),
                chunk_size=config.get("pace_score_chunk_size", 1),
                entropy_coeff=config["entropy_coeff"] if config.get("pace_score_entropy", False) else 0.0,
            )
            metrics = {"pace": pace_score.rollout_statistics(scores, advantages, dones, rewards)} if log_pace else {}
            return jnp.where(jnp.isfinite(scores), scores, 0), metrics

        def insert_levels(sampler, levels, dones, rewards, values, advantages, pace_priority=None):
            max_returns = compute_max_returns(dones, rewards)
            if use_learnability:
                evidence = learnability.rollout_evidence(dones, rewards, values, max_returns)
                if dual_scores:
                    return hybrid_teacher.insert_batch(
                        level_sampler, sampler, levels, evidence, round_index, hybrid_weight, raw_rank=rank_priorities
                    )
                return learnability.insert_batch(
                    level_sampler, sampler, levels, evidence, round_index, history_weight=history_weight
                )
            # The singleton time axis lets adaptive insertion slice this alongside the rollout.
            scores = (compute_score(config, dones, values, max_returns, advantages)
                      if pace_priority is None else pace_priority[0])
            return level_sampler.insert_batch(sampler, levels, scores, {"max_return": max_returns})[0]

        def candidate_scores(dones, rewards, values, advantages):
            max_returns = compute_max_returns(dones, rewards)
            if not use_learnability:
                return compute_score(config, dones, values, max_returns, advantages)
            evidence = learnability.rollout_evidence(dones, rewards, values, max_returns)
            history = learnability.empty_history(track_progress=use_har)
            return jax.vmap(lambda row: learnability.record_evidence(
                history, -jnp.inf, row, round_index, history_weight=history_weight
            )[0])(evidence)

        def observe_qd(sampler, levels, dones, rewards, values, key):
            if qd_mode == "off":
                return sampler
            evidence = learnability.rollout_evidence(dones, rewards, values, compute_max_returns(dones, rewards))
            quality, observations = map_elites.rollout_quality(levels, evidence, round_index)
            archive_key = jax.random.fold_in(key, 74021) if qd_mode == "random_archive" else None
            archive = map_elites.update_archive_batch(
                sampler["qd_archive"], levels, quality, observations, round_index,
                wall_edges, deadend_edges, qd_max_age, random_key=archive_key,
            )
            return {**sampler, "qd_archive": archive}

        def trace_selection(branch, rng, levels, indices=None, parents=None):
            if parity is not None:
                def record(round_index, key, selected, slots, parent_levels):
                    parity.selection(round_index, branch, key, selected, slots, parent_levels)
                jax.debug.callback(record, round_index, rng, levels, indices, parents, ordered=True)

        def trace_rollout(rng, obs, actions, rewards, dones, log_probs, values, last_value):
            if parity is not None:
                jax.debug.callback(parity.rollout, round_index, rng, obs, actions, rewards, dones,
                                   log_probs, values, last_value, ordered=True)

        def trace_sampler(sampler):
            if parity is not None:
                jax.debug.callback(parity.buffer, round_index, teacher_parity.base_sampler(sampler), ordered=True)

        def on_new_levels(rng: chex.PRNGKey, train_state: TrainState):
            """
                Samples new (randomly-generated) levels and evaluates the policy on these. It also then adds the levels to the level buffer if they have high-enough scores.
                The agent is updated on these trajectories iff `config["exploratory_grad_updates"]` is True.
            """
            sampler = train_state.sampler
            
            # Reset
            rng, rng_levels, rng_reset = jax.random.split(rng, 3)
            new_levels = jax.vmap(sample_random_level)(jax.random.split(rng_levels, config["num_train_envs"]))
            trace_selection("new", rng, new_levels)
            init_obs, init_env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(jax.random.split(rng_reset, config["num_train_envs"]), new_levels, env_params)
            # Rollout
            (
                (rng, train_state, hstate, last_obs, last_env_state, last_value),
                (obs, actions, rewards, dones, log_probs, values, info),
            ) = sample_trajectories_rnn(
                rng,
                env,
                env_params,
                train_state,
                ActorCritic.initialize_carry((config["num_train_envs"],)),
                init_obs,
                init_env_state,
                config["num_train_envs"],
                config["num_steps"],
            )
            trace_rollout(rng, obs, actions, rewards, dones, log_probs, values, last_value)
            advantages, targets = compute_gae(config["gamma"], config["gae_lambda"], last_value, values, rewards, dones)
            pace_scores, pace_metrics = score_pace_rollout(train_state, obs, actions, dones, rewards, advantages)
            pace_priority = None if pace_scores is None else pace_scores[None]
            sampler = insert_levels(sampler, new_levels, dones, rewards, values, advantages, pace_priority)
            sampler = observe_qd(sampler, new_levels, dones, rewards, values, rng)
            trace_sampler(sampler)
            
            # Update: train_state only modified if exploratory_grad_updates is on
            (rng, train_state), losses = update_actor_critic_rnn(
                rng,
                train_state,
                ActorCritic.initialize_carry((config["num_train_envs"],)),
                (obs, actions, dones, log_probs, values, targets, advantages),
                config["num_train_envs"],
                config["num_steps"],
                config["num_minibatches"],
                config["epoch_ppo"],
                config["clip_eps"],
                config["entropy_coeff"],
                config["critic_coeff"],
                update_grad=config["exploratory_grad_updates"],
            )
            
            metrics = {
                "losses": jax.tree_util.tree_map(lambda x: x.mean(), losses),
                "mean_num_blocks": new_levels.wall_map.sum() / config["num_train_envs"],
                "replay": jnp.zeros(5, dtype=jnp.float32),
                **pace_metrics,
            }
            if selection != "fixed":
                metrics["mutation"] = empty_mutation_metrics
            if qd_mode != "off":
                metrics["qd"] = empty_qd_metrics
            
            train_state = train_state.replace(
                sampler=sampler,
                update_state=UpdateState.DR,
                num_dr_updates=train_state.num_dr_updates + 1,
                dr_last_level_batch=new_levels,
            )
            return (rng, train_state), metrics
        
        def on_replay_levels(rng: chex.PRNGKey, train_state: TrainState):
            """
                This samples levels from the level buffer, and updates the policy on them.
            """
            sampler = train_state.sampler
            
            # Collect trajectories on replay levels
            rng, rng_levels, rng_reset = jax.random.split(rng, 3)
            sampler, (level_inds, levels) = level_sampler.sample_replay_levels(sampler, rng_levels, config["num_train_envs"])
            trace_selection("replay", rng, levels, level_inds)
            replay_stats = teacher_experiments.replay_batch_stats(level_inds, teacher_experiments.frontier_mask(sampler))
            init_obs, init_env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(jax.random.split(rng_reset, config["num_train_envs"]), levels, env_params)
            (
                (rng, train_state, hstate, last_obs, last_env_state, last_value),
                (obs, actions, rewards, dones, log_probs, values, info),
            ) = sample_trajectories_rnn(
                rng,
                env,
                env_params,
                train_state,
                ActorCritic.initialize_carry((config["num_train_envs"],)),
                init_obs,
                init_env_state,
                config["num_train_envs"],
                config["num_steps"],
            )
            trace_rollout(rng, obs, actions, rewards, dones, log_probs, values, last_value)
            advantages, targets = compute_gae(config["gamma"], config["gae_lambda"], last_value, values, rewards, dones)
            pace_scores, pace_metrics = score_pace_rollout(train_state, obs, actions, dones, rewards, advantages)
            max_returns = compute_max_returns(dones, rewards)
            if use_learnability:
                evidence = learnability.rollout_evidence(dones, rewards, values, max_returns)
                if dual_scores:
                    sampler = hybrid_teacher.update_batch(
                        level_sampler, sampler, level_inds, evidence, dones, values, round_index, hybrid_weight,
                        raw_rank=rank_priorities,
                    )
                else:
                    sampler = learnability.update_batch(
                        level_sampler, sampler, level_inds, evidence, round_index, history_weight=history_weight
                    )
            else:
                max_returns = jnp.maximum(level_sampler.get_levels_extra(sampler, level_inds)["max_return"], max_returns)
                scores = (compute_score(config, dones, values, max_returns, advantages)
                          if pace_scores is None else pace_scores)
                sampler = level_sampler.update_batch(sampler, level_inds, scores, {"max_return": max_returns})
            sampler = observe_qd(sampler, levels, dones, rewards, values, rng)
            trace_sampler(sampler)
            
            # Update the policy using trajectories collected from replay levels
            (rng, train_state), losses = update_actor_critic_rnn(
                rng,
                train_state,
                ActorCritic.initialize_carry((config["num_train_envs"],)),
                (obs, actions, dones, log_probs, values, targets, advantages),
                config["num_train_envs"],
                config["num_steps"],
                config["num_minibatches"],
                config["epoch_ppo"],
                config["clip_eps"],
                config["entropy_coeff"],
                config["critic_coeff"],
                update_grad=True,
            )
                            
            metrics = {
                "losses": jax.tree_util.tree_map(lambda x: x.mean(), losses),
                "mean_num_blocks": levels.wall_map.sum() / config["num_train_envs"],
                **pace_metrics,
                "replay": replay_stats,
            }
            if selection != "fixed":
                metrics["mutation"] = empty_mutation_metrics
            if qd_mode != "off":
                metrics["qd"] = empty_qd_metrics
            
            train_state = train_state.replace(
                sampler=sampler,
                update_state=UpdateState.REPLAY,
                num_replay_updates=train_state.num_replay_updates + 1,
                replay_last_level_batch=levels,
            )
            return (rng, train_state), metrics
        
        def on_mutate_levels(rng: chex.PRNGKey, train_state: TrainState):
            """
                This mutates the previous batch of replay levels and potentially adds them to the level buffer.
                This also updates the policy iff `config["exploratory_grad_updates"]` is True.
            """
            sampler = train_state.sampler
            rng, rng_mutate, rng_reset = jax.random.split(rng, 3)
            
            # mutate
            parent_levels = train_state.replay_last_level_batch
            if qd_mode != "off":
                qd_niches = jnp.full((config["num_train_envs"],), -1, dtype=jnp.int32)
                if qd_mode in ("map_elites", "random_archive"):
                    # Derive a separate key without changing the original mutator key.
                    parent_levels, qd_niches = map_elites.sample_archive_parents(
                        sampler["qd_archive"], parent_levels, jax.random.fold_in(rng_mutate, 65537),
                        round_index, qd_max_age, config["qd_parent_prob"],
                    )
            if selection == "fixed":
                child_levels = jax.vmap(mutate_level, (0, 0, None))(
                    jax.random.split(rng_mutate, config["num_train_envs"]), parent_levels, config["num_edits"]
                )
            else:
                select_key, mutate_key = jax.random.split(rng_mutate)
                batch_size = config["num_train_envs"]
                if selection == "bandit":
                    indices = adaptive_mutation.sample_arms(
                        select_key, sampler["mutation_bandit"], batch_size, config["bandit_epsilon"]
                    )
                else:
                    indices = adaptive_mutation.sample_uniform(select_key, batch_size, num_arms)
                children = adaptive_mutation.mutate_batch(
                    mutate_level, jax.random.split(mutate_key, batch_size), parent_levels, indices, arms
                )
                stats = adaptive_mutation.candidate_stats(parent_levels, children)
                child_levels = adaptive_mutation.safe_children(parent_levels, children, stats[0])
            if parity is not None:
                parent_slots = jax.vmap(level_sampler.find, (None, 0))(sampler, parent_levels)
                trace_selection("mutation", rng, child_levels, parent_slots, parent_levels)
            init_obs, init_env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(jax.random.split(rng_reset, config["num_train_envs"]), child_levels, env_params)

            # rollout
            (
                (rng, train_state, hstate, last_obs, last_env_state, last_value),
                (obs, actions, rewards, dones, log_probs, values, info),
            ) = sample_trajectories_rnn(
                rng,
                env,
                env_params,
                train_state,
                ActorCritic.initialize_carry((config["num_train_envs"],)),
                init_obs,
                init_env_state,
                config["num_train_envs"],
                config["num_steps"],
            )
            trace_rollout(rng, obs, actions, rewards, dones, log_probs, values, last_value)
            advantages, targets = compute_gae(config["gamma"], config["gae_lambda"], last_value, values, rewards, dones)
            pace_scores, pace_metrics = score_pace_rollout(train_state, obs, actions, dones, rewards, advantages)
            pace_priority = None if pace_scores is None else pace_scores[None]
            if selection == "fixed":
                sampler = insert_levels(sampler, child_levels, dones, rewards, values, advantages, pace_priority)
            else:
                rollout = dones, rewards, values, advantages
                scores = candidate_scores(*rollout) if pace_scores is None else pace_scores
                fitness = adaptive_mutation.assess(
                    scores, stats, config["bandit_reward"], eval_env.max_height * eval_env.max_width
                )
                if selection == "bandit":
                    sampler = {**sampler, "mutation_bandit": adaptive_mutation.update(
                        sampler["mutation_bandit"], indices, fitness, config["bandit_ema_alpha"], round_index
                    )}
                insertion = rollout if pace_priority is None else (*rollout, pace_priority)
                sampler = adaptive_mutation.insert_valid(insert_levels, sampler, child_levels, stats[0], insertion)
            sampler = observe_qd(sampler, child_levels, dones, rewards, values, rng)
            trace_sampler(sampler)
            
            # Update: train_state only modified if exploratory_grad_updates is on
            (rng, train_state), losses = update_actor_critic_rnn(
                rng,
                train_state,
                ActorCritic.initialize_carry((config["num_train_envs"],)),
                (obs, actions, dones, log_probs, values, targets, advantages),
                config["num_train_envs"],
                config["num_steps"],
                config["num_minibatches"],
                config["epoch_ppo"],
                config["clip_eps"],
                config["entropy_coeff"],
                config["critic_coeff"],
                update_grad=config["exploratory_grad_updates"],
            )
            
            metrics = {
                "losses": jax.tree_util.tree_map(lambda x: x.mean(), losses),
                "mean_num_blocks": child_levels.wall_map.sum() / config["num_train_envs"],
                "replay": jnp.zeros(5, dtype=jnp.float32),
                **pace_metrics,
            }
            if selection != "fixed":
                metrics["mutation"] = adaptive_mutation.batch_metrics(stats, fitness, indices, num_arms)
            if qd_mode != "off":
                valid, unchanged, _, _ = adaptive_mutation.candidate_stats(parent_levels, child_levels)
                histogram = jnp.bincount(jnp.maximum(qd_niches, 0),
                                         weights=(qd_niches >= 0).astype(jnp.float32), length=qd_size)
                values = jnp.array([
                    1, (qd_niches >= 0).sum(), map_elites.unique_count(parent_levels),
                    map_elites.unique_count(child_levels), valid.mean(), unchanged.mean(),
                ], dtype=jnp.float32)
                metrics["qd"] = jnp.concatenate((values, histogram))
            
            train_state = train_state.replace(
                sampler=sampler,
                update_state=UpdateState.DR,
                num_mutation_updates=train_state.num_mutation_updates + 1,
                mutation_last_level_batch=child_levels,
            )
            return (rng, train_state), metrics
    
        input_rng = rng
        rng, rng_replay = jax.random.split(rng)
        
        # The train step makes a decision on which branch to take, either on_new, on_replay or on_mutate.
        # on_mutate is only called if the replay branch has been taken before (as it uses `train_state.update_state`).
        if config["use_accel"]:
            s = train_state.update_state
            branch = (1 - s) * level_sampler.sample_replay_decision(train_state.sampler, rng_replay) + 2 * s
        else:
            branch = level_sampler.sample_replay_decision(train_state.sampler, rng_replay).astype(int)
        
        if parity is not None:
            jax.debug.callback(parity.round_start, round_index, branch, input_rng, rng_replay, ordered=True)
        updated, metrics = jax.lax.switch(
            branch,
            [
                on_new_levels,
                on_replay_levels,
                on_mutate_levels,
            ],
            rng, train_state
        )
        if dual_scores:
            key, state = updated
            next_weight = hybrid_teacher.config_coefficient(config, round_index + 1)
            updated = key, state.replace(sampler=hybrid_teacher.reweight(state.sampler, next_weight))
        if parity is not None:
            def record(key, state, index, losses):
                parity.snapshot(state, key, index, "end", losses)
            jax.debug.callback(record, *updated, round_index, metrics["losses"], ordered=True)
        return updated, metrics
    
    def eval(rng: chex.PRNGKey, train_state: TrainState):
        """
        This evaluates the current policy on the set of evaluation levels specified by config["eval_levels"].
        It returns (states, cum_rewards, episode_lengths), with shapes (num_steps, num_eval_levels, ...), (num_eval_levels,), (num_eval_levels,)
        """
        rng, rng_reset = jax.random.split(rng)
        levels = Level.load_prefabs(config["eval_levels"])
        num_levels = len(config["eval_levels"])
        init_obs, init_env_state = jax.vmap(eval_env.reset_to_level, (0, 0, None))(jax.random.split(rng_reset, num_levels), levels, env_params)
        states, rewards, episode_lengths = evaluate_rnn(
            rng,
            eval_env,
            env_params,
            train_state,
            ActorCritic.initialize_carry((num_levels,)),
            init_obs,
            init_env_state,
            env_params.max_steps_in_episode,
        )
        mask = jnp.arange(env_params.max_steps_in_episode)[..., None] < episode_lengths
        cum_rewards = (rewards * mask).sum(axis=0)
        return states, cum_rewards, episode_lengths # (num_steps, num_eval_levels, ...), (num_eval_levels,), (num_eval_levels,)
    
    @jax.jit
    def train_and_eval_step(runner_state, _):
        """
            This function runs the train_step for a certain number of iterations, and then evaluates the policy.
            It returns the updated train state, and a dictionary of metrics.
        """
        # Train
        (rng, train_state), metrics = jax.lax.scan(train_step, runner_state, None, config["eval_freq"])
        batches, unique, copies, frontier, ess = metrics["replay"].sum(axis=0)
        denom = jnp.maximum(batches, 1)
        metrics["replay"] = {
            "replay/batches": batches,
            "replay/mean_unique_levels": unique / denom,
            "replay/max_copies": metrics["replay"][:, 2].max(),
            "replay/mean_max_copies": copies / denom,
            "replay/mean_frontier_share": frontier / denom,
            "replay/mean_batch_ess": ess / denom,
        }
        if log_pace:
            pace_metrics = metrics["pace"]
            metrics["pace"] = {key: value.mean() for key, value in pace_metrics.items()}
            metrics["pace"]["pace/max_score"] = pace_metrics["pace/max_score"].max()
        if qd_mode != "off":
            totals = metrics.pop("qd").sum(axis=0)
            batches, parents, unique_parents, unique_children, valid, unchanged = totals[:6]
            frequencies = totals[6:] / jnp.maximum(parents, 1)
            denom = jnp.maximum(batches, 1)
            metrics["qd"] = {
                "qd/parents_from_archive": parents,
                "qd/archive_parent_ratio": parents / (denom * config["num_train_envs"]),
                "qd/unique_parents": unique_parents / denom,
                "qd/unique_children": unique_children / denom,
                "qd/valid_children_ratio": valid / denom,
                "qd/unchanged_children_ratio": unchanged / denom,
                "qd/niche_selection_entropy": -(frequencies * jnp.log(jnp.maximum(frequencies, 1e-20))).sum(),
            }
        if selection != "fixed":
            batches, reward, unique, invalid, changed, *counts = metrics.pop("mutation").sum(axis=0)
            counts = jnp.stack(counts)
            frequencies = counts / jnp.maximum(counts.sum(), 1)
            denom = jnp.maximum(batches, 1)
            metrics["adaptive_mutation"] = {
                "adaptive_mutation/batches": batches,
                "adaptive_mutation/mean_reward": reward / denom,
                "adaptive_mutation/operator_entropy": -(frequencies * jnp.log(jnp.maximum(frequencies, 1e-20))).sum(),
                "adaptive_mutation/unique_children_ratio": unique / denom,
                "adaptive_mutation/invalid_children_ratio": invalid / denom,
                "adaptive_mutation/mean_changed_cells": changed / denom,
            }
            if selection == "bandit":
                bandit = train_state.sampler["mutation_bandit"]
                counts, estimates = bandit["count"], bandit["ema_reward"]
            else:
                estimates = jnp.zeros_like(counts)
            for i in range(num_arms):
                metrics["adaptive_mutation"][f"adaptive_mutation/operator_{i}_count"] = counts[i]
                metrics["adaptive_mutation"][f"adaptive_mutation/operator_{i}_ema_reward"] = estimates[i]

        # Eval
        rng, rng_eval = jax.random.split(rng)
        states, cum_rewards, episode_lengths = jax.vmap(eval, (0, None))(jax.random.split(rng_eval, config["eval_num_attempts"]), train_state)
        
        # Collect Metrics
        eval_solve_rates = jnp.where(cum_rewards > 0, 1., 0.).mean(axis=0) # (num_eval_levels,)
        eval_returns = cum_rewards.mean(axis=0) # (num_eval_levels,)
        
        # just grab the first run
        states, episode_lengths = jax.tree_util.tree_map(lambda x: x[0], (states, episode_lengths)) # (num_steps, num_eval_levels, ...), (num_eval_levels,)
        images = jax.vmap(jax.vmap(env_renderer.render_state, (0, None)), (0, None))(states, env_params) # (num_steps, num_eval_levels, ...)
        frames = images.transpose(0, 1, 4, 2, 3) # WandB expects color channel before image dimensions when dealing with animations for some reason
        
        metrics["update_count"] = train_state.num_dr_updates + train_state.num_replay_updates + train_state.num_mutation_updates
        metrics["eval_returns"] = eval_returns
        metrics["eval_solve_rates"] = eval_solve_rates
        metrics["eval_ep_lengths"]  = episode_lengths
        metrics["eval_animation"] = (frames, episode_lengths)
        metrics["dr_levels"] = jax.vmap(env_renderer.render_level, (0, None))(train_state.dr_last_level_batch, env_params)
        metrics["replay_levels"] = jax.vmap(env_renderer.render_level, (0, None))(train_state.replay_last_level_batch, env_params)
        metrics["mutation_levels"] = jax.vmap(env_renderer.render_level, (0, None))(train_state.mutation_last_level_batch, env_params)
        
        weight_sampler = train_state.sampler
        if dual_scores:
            weight = hybrid_teacher.config_coefficient(config, metrics["update_count"])
            weight_sampler = hybrid_teacher.reweight(weight_sampler, weight, raw_rank=rank_priorities)
        highest_scoring_level = level_sampler.get_levels(weight_sampler, weight_sampler["scores"].argmax())
        highest_weighted_level = level_sampler.get_levels(weight_sampler, level_sampler.level_weights(weight_sampler).argmax())
        
        metrics["highest_scoring_level"] = env_renderer.render_level(highest_scoring_level, env_params)
        metrics["highest_weighted_level"] = env_renderer.render_level(highest_weighted_level, env_params)
        
        return (rng, train_state), metrics
    
    def eval_checkpoint(og_config):
        """
            This function is what is used to evaluate a saved checkpoint *after* training. It first loads the checkpoint and then runs evaluation.
            It saves the states, cum_rewards and episode_lengths to a .npz file in the `results/run_name/seed` directory.
        """
        rng_init, rng_eval = jax.random.split(jax.random.PRNGKey(10000))
        def load(rng_init, checkpoint_directory: str):
            with open(os.path.join(checkpoint_directory, 'config.json')) as f: config = json.load(f)
            checkpoint_manager = ocp.CheckpointManager(os.path.join(os.getcwd(), checkpoint_directory, 'models'), item_handlers=ocp.StandardCheckpointHandler())

            train_state_og: TrainState = create_train_state(rng_init)
            step = checkpoint_manager.latest_step() if og_config['checkpoint_to_eval'] == -1 else og_config['checkpoint_to_eval']

            loaded_checkpoint = checkpoint_manager.restore(step)
            params = loaded_checkpoint['params']
            train_state = train_state_og.replace(params=params)
            return train_state, config
        
        train_state, config = load(rng_init, og_config['checkpoint_directory'])
        states, cum_rewards, episode_lengths = jax.vmap(eval, (0, None))(jax.random.split(rng_eval, og_config["eval_num_attempts"]), train_state)
        save_loc = og_config['checkpoint_directory'].replace('checkpoints', 'results')
        os.makedirs(save_loc, exist_ok=True)
        np.savez_compressed(os.path.join(save_loc, 'results.npz'), states=np.asarray(states), cum_rewards=np.asarray(cum_rewards), episode_lengths=np.asarray(episode_lengths), levels=config['eval_levels'])
        return states, cum_rewards, episode_lengths

    if config['mode'] == 'eval':
        return eval_checkpoint(config) # evaluate and exit early

    # Set up the train states
    rng = jax.random.PRNGKey(config["seed"])
    rng_init, rng_train = jax.random.split(rng)
    
    train_state = create_train_state(rng_init)
    runner_state = (rng_train, train_state)
    completed = 0
    if resume_dir:
        rng_train, train_state, completed = mutation_checkpoint.restore(
            resume_dir, config.get("resume_checkpoint_step", -1), train_state, config.as_dict()
        )
        runner_state = (rng_train, train_state)
        print(f"Resumed training after {completed} rounds.", flush=True)
    stop_after = config.get("stop_after_updates")
    if stop_after is None:
        stop_after = config["num_updates"]
    interval = config["eval_freq"]
    if completed % interval or (resume_dir and completed >= stop_after):
        raise ValueError("resume must start on an evaluation boundary before the target round")
    if parity is not None:
        parity.snapshot(runner_state[1], runner_state[0], completed, "initial")
    
    # And run the train_eval_sep function for the specified number of updates
    if config["checkpoint_save_interval"] > 0:
        checkpoint_manager = setup_checkpointing(config, train_state, env, env_params)
        if dual_scores and resume_dir:
            initial = {"round": completed, "optimizer_steps": int(train_state.step),
                       "leaves": teacher_parity.tree_digests(mutation_checkpoint.payload(*runner_state))}
            (checkpoint_manager.directory.parent / "hybrid_initial_state.json").write_text(
                json.dumps(initial, indent=2) + "\n", encoding="utf-8"
            )
    for eval_step in range(completed // interval, stop_after // interval):
        start_time = time.time()
        runner_state, metrics = train_and_eval_step(runner_state, None)
        if parity is not None:
            jax.effects_barrier()
            parity.snapshot(runner_state[1], runner_state[0], metrics["update_count"], "evaluation_boundary")
        curr_time = time.time()
        metrics['time_delta'] = curr_time - start_time
        state = runner_state[1]
        weights = None
        if dual_scores:
            weight = hybrid_teacher.config_coefficient(config, metrics["update_count"])
            weights = hybrid_teacher.reweight(state.sampler, weight, raw_rank=rank_priorities)
        state_info = train_state_to_log_dict(state, level_sampler, weight_sampler=weights)
        if dual_scores:
            state_info["log"].update(hybrid_teacher.statistics(state.sampler, weight))
        diagnostics = log_eval(metrics, state_info)
        if config["checkpoint_save_interval"] > 0:
            diagnostics = {key: float(value) for key, value in diagnostics.items()}
            diagnostics["num_updates"] = int(metrics["update_count"])
            if (dual_scores or use_har or level_sampler.replay_max_copies or selection != "fixed" or qd_mode != "off") and (
                not np.isfinite(list(diagnostics.values())).all()
                or diagnostics["level_sampler/nonfinite_active_scores"]
                or diagnostics["level_sampler/nonfinite_active_weights"]
            ):
                raise FloatingPointError(f"nonfinite teacher statistics at update {diagnostics['num_updates']}")
            with open(checkpoint_manager.directory.parent / "teacher_metrics.jsonl", "a", encoding="utf-8") as file:
                file.write(json.dumps(diagnostics) + "\n")
            checkpoint_manager.save(eval_step, args=ocp.args.StandardSave(mutation_checkpoint.payload(*runner_state)))
            checkpoint_manager.wait_until_finished()
    if config["checkpoint_save_interval"] > 0:
        checkpoint_manager.close()
    if parity is not None:
        jax.effects_barrier()
    return runner_state[1]

if __name__=="__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="JAXUED_TEST")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    # === Train vs Eval ===
    parser.add_argument("--mode", type=str, default='train')
    parser.add_argument("--checkpoint_directory", type=str, default=None)
    parser.add_argument("--checkpoint_to_eval", type=int, default=-1)
    parser.add_argument("--debug_teacher_parity", metavar="TRACE_JSONL")
    parser.add_argument("--resume_checkpoint_directory")
    parser.add_argument("--resume_checkpoint_step", type=int, default=-1)
    parser.add_argument("--initialize_bandit_on_resume", action="store_true")
    parser.add_argument("--initialize_qd_on_resume", action="store_true")
    parser.add_argument("--fork_hybrid_on_resume", action="store_true")
    parser.add_argument("--allow_legacy_resume", action="store_true")
    parser.add_argument("--stop_after_updates", type=int)
    # === CHECKPOINTING ===
    parser.add_argument("--checkpoint_save_interval", type=int, default=2)
    parser.add_argument("--max_number_of_checkpoints", type=int, default=60)
    # === EVAL ===
    parser.add_argument("--eval_freq", type=int, default=250)
    parser.add_argument("--eval_num_attempts", type=int, default=10)
    parser.add_argument("--eval_levels", nargs='+', default=[
        "SixteenRooms",
        "SixteenRooms2",
        "Labyrinth",
        "LabyrinthFlipped",
        "Labyrinth2",
        "StandardMaze",
        "StandardMaze2",
        "StandardMaze3",
    ])
    group = parser.add_argument_group('Training params')
    # === PPO === 
    group.add_argument("--lr", type=float, default=1e-4)
    group.add_argument("--max_grad_norm", type=float, default=0.5)
    mut_group = group.add_mutually_exclusive_group()
    mut_group.add_argument("--num_updates", type=int, default=30000)
    mut_group.add_argument("--num_env_steps", type=int, default=None)
    group.add_argument("--num_steps", type=int, default=256)
    group.add_argument("--num_train_envs", type=int, default=32)
    group.add_argument("--num_minibatches", type=int, default=1)
    group.add_argument("--gamma", type=float, default=0.995)
    group.add_argument("--epoch_ppo", type=int, default=5)
    group.add_argument("--clip_eps", type=float, default=0.2)
    group.add_argument("--gae_lambda", type=float, default=0.98)
    group.add_argument("--entropy_coeff", type=float, default=1e-3)
    group.add_argument("--critic_coeff", type=float, default=0.5)
    # === PLR ===
    group.add_argument("--score_function", type=str, default="MaxMC",
                       choices=["MaxMC", "pvl", "recent_learnability", "history_aware_regret", "hybrid", "pace_pg"])
    group.add_argument("--dual_score_tracking", action="store_true")
    group.add_argument("--hybrid_mode", choices=hybrid_teacher.MODES)
    group.add_argument("--hybrid_start_round", type=int, default=hybrid_teacher.DEFAULTS["hybrid_start_round"])
    group.add_argument("--hybrid_end_round", type=int, default=hybrid_teacher.DEFAULTS["hybrid_end_round"])
    group.add_argument("--har_history_weight", type=float, default=0.2)
    group.add_argument("--replay_max_copies", type=int, default=0)
    group.add_argument("--replay_preserve_frontier_mass", action=argparse.BooleanOptionalAction, default=False)
    group.add_argument("--exploratory_grad_updates", action=argparse.BooleanOptionalAction, default=False)
    group.add_argument("--level_buffer_capacity", type=int, default=4000)
    group.add_argument("--replay_prob", type=float, default=0.8)
    group.add_argument("--staleness_coeff", type=float, default=0.3)
    group.add_argument("--temperature", type=float, default=0.3)
    group.add_argument("--topk_k", type=int, default=4)
    group.add_argument("--minimum_fill_ratio", type=float, default=0.5)
    group.add_argument("--prioritization", type=str, default="rank", choices=["rank", "topk"])
    group.add_argument("--buffer_duplicate_check", action=argparse.BooleanOptionalAction, default=True)
    # === ACCEL ===
    group.add_argument("--use_accel", action=argparse.BooleanOptionalAction, default=False)
    group.add_argument("--num_edits", type=int, default=5)
    group.add_argument("--adaptive_mutation", action="store_true")
    group.add_argument("--mutation_selection", choices=["fixed", "uniform", "bandit"])
    group.add_argument("--bandit_strategy", choices=["epsilon_greedy"], default="epsilon_greedy")
    group.add_argument("--bandit_epsilon", type=float, default=0.15)
    group.add_argument("--bandit_ema_alpha", type=float, default=0.1)
    group.add_argument("--bandit_reward", choices=["batch_rank", "changed_cells"], default="batch_rank")
    group.add_argument("--mutation_arms", default="1,3,5,10")
    group.add_argument("--qd_mode", choices=["off", "shadow", "map_elites", "random_archive"], default="off")
    group.add_argument("--qd_bins", default="5,5")
    group.add_argument("--qd_wall_edges")
    group.add_argument("--qd_deadend_edges")
    group.add_argument("--qd_parent_prob", type=float, default=0.5)
    group.add_argument("--qd_max_age", type=int, default=1024)
    # === ENV CONFIG ===
    group.add_argument("--agent_view_size", type=int, default=5)
    # === DR CONFIG ===
    group.add_argument("--n_walls", type=int, default=25)

    pace_args = parser.add_argument_group("PACE-PG")
    pace_args.add_argument("--pace_score_chunk_size", type=int, default=1)
    pace_args.add_argument("--pace_score_entropy", action="store_true")
    pace_args.add_argument("--pace_log_diagnostics", action="store_true")
    
    config = vars(parser.parse_args())
    if config["pace_score_chunk_size"] < 1:
        parser.error("--pace_score_chunk_size must be positive")
    try:
        bins = [int(value) for value in config["qd_bins"].split(",")]
        if len(bins) != 2 or any(value < 1 for value in bins):
            raise ValueError
        for arg, default, num in zip(("qd_wall_edges", "qd_deadend_edges"), map_elites.DEFAULT_EDGES, bins):
            raw = config[arg]
            edges = list(default) if raw is None else ([float(np.float32(x)) for x in raw.split(",")] if raw else [])
            if len(edges) != num - 1 or any(not 0 < x < 1 for x in edges) or any(
                a >= b for a, b in zip(edges, edges[1:])
            ):
                parser.error(f"--{arg} needs {num - 1} strictly increasing values within (0, 1)")
            config[arg] = edges
        config["qd_bins"] = bins
    except ValueError:
        parser.error("invalid --qd_bins or QD edge list")
    if not 0 <= config["qd_parent_prob"] <= 1 or config["qd_max_age"] < 0:
        parser.error("QD parent probability must be in [0, 1] and max age nonnegative")
    selection = config["mutation_selection"]
    if config["adaptive_mutation"] and selection not in (None, "bandit"):
        parser.error("--adaptive_mutation conflicts with --mutation_selection fixed/uniform")
    config["mutation_selection"] = selection or ("bandit" if config["adaptive_mutation"] else "fixed")
    if config["qd_mode"] != "off" and (
        not config["use_accel"] or config["score_function"] != "recent_learnability"
        or config["mutation_selection"] != "fixed" or config["num_edits"] != 5
        or config["replay_max_copies"]
    ):
        parser.error("QD v1 requires ACCEL, recent_learnability, fixed 5 edits and uncapped replay")
    if config["debug_teacher_parity"] and (
        config["mode"] != "train" or not config["use_accel"] or config["mutation_selection"] != "fixed"
        or config["qd_mode"] not in ("off", "shadow") or not config["buffer_duplicate_check"]
    ):
        parser.error("parity trace requires fixed ACCEL off/shadow training with duplicate checking")
    if config["mutation_selection"] != "fixed" and not config["use_accel"]:
        parser.error("adaptive/uniform mutation requires --use_accel")
    try:
        arms = [int(value) for value in config["mutation_arms"].split(",")]
    except ValueError:
        parser.error("--mutation_arms requires a comma-separated list of integers")
    if len(set(arms)) != len(arms) or any(not 1 <= value <= adaptive_mutation.MAX_EDITS for value in arms):
        parser.error("--mutation_arms must be distinct and between 1 and 100")
    config["mutation_arms"] = arms
    if not 0 <= config["bandit_epsilon"] <= 1 or not 0 < config["bandit_ema_alpha"] <= 1:
        parser.error("epsilon must be in [0, 1] and EMA alpha in (0, 1]")
    if config["eval_freq"] <= 0:
        parser.error("--eval_freq must be positive")
    if config["resume_checkpoint_step"] < -1:
        parser.error("--resume_checkpoint_step must be -1 or a nonnegative checkpoint index")
    if (config["initialize_bandit_on_resume"] or config["initialize_qd_on_resume"] or config["allow_legacy_resume"]
            or config["resume_checkpoint_step"] != -1) and not config["resume_checkpoint_directory"]:
        parser.error("resume options require --resume_checkpoint_directory")
    if not 0 <= config["har_history_weight"] <= 1:
        parser.error("--har_history_weight must be between 0 and 1")
    if config["replay_max_copies"] < 0:
        parser.error("--replay_max_copies must be nonnegative")
    if config["replay_preserve_frontier_mass"] and not config["replay_max_copies"]:
        parser.error("--replay_preserve_frontier_mass requires a positive --replay_max_copies")
    hybrid_score = config["score_function"] == "hybrid"
    config["hybrid_mode"] = config["hybrid_mode"] or ("smooth" if hybrid_score else "recent_control")
    config["dual_score_tracking"] |= hybrid_score
    if config["hybrid_mode"] != "recent_control" and not hybrid_score:
        parser.error("an active hybrid schedule needs --score_function hybrid")
    if config["dual_score_tracking"] and (
        config["score_function"] not in ("recent_learnability", "hybrid")
        or config["mutation_selection"] != "fixed" or config["qd_mode"] != "off" or config["replay_max_copies"]
        or not 0 <= config["hybrid_start_round"] < config["hybrid_end_round"]
    ):
        parser.error("dual scoring requires Recent/hybrid, fixed mutation, uncapped replay, QD off and ordered rounds")
    if config["fork_hybrid_on_resume"] and (not config["resume_checkpoint_directory"] or not hybrid_score):
        parser.error("hybrid fork requires --resume_checkpoint_directory and --score_function hybrid")
    if config["num_env_steps"] is not None:
        config["num_updates"] = config["num_env_steps"] // (config["num_train_envs"] * config["num_steps"])
    stop_after = config["stop_after_updates"]
    if stop_after is not None and (
        not 0 < stop_after <= config["num_updates"] or stop_after % config["eval_freq"]
    ):
        parser.error("--stop_after_updates must be a positive evaluation boundary within the training budget")
    config["group_name"] = ''.join([str(config[key]) for key in sorted([a.dest for a in parser._action_groups[2]._group_actions])])
    
    if config['mode'] == 'eval':
        os.environ['WANDB_MODE'] = 'disabled'
    
    # wandb.login()
    main(config, project=config["project"])
