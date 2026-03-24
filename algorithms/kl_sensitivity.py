"""
KL Divergence Sensitivity Analysis for AMULED (finetune_moral.py)

Runs the same PPO fine-tuning loop as finetune_moral.py but sweeps over
different kl_penalty_factor values. Uses the pre-computed LLM reward cache
so no LLM server is needed.

Usage:
    python algorithms/kl_sensitivity.py                          # both envs, default KL range
    python algorithms/kl_sensitivity.py --env Driving            # single env
    python algorithms/kl_sensitivity.py --kl-factors 0.0 1.0 5.0 # custom KL factors
"""

import os
import sys
import pickle
import argparse
import time

sys.path.insert(1, os.path.join(os.path.dirname(__file__), ".."))

import gymnasium as gym
try:
    from gymnasium.vector import SyncVectorEnv
except AttributeError:
    SyncVectorEnv = gym.vector.SyncVectorEnv  # gymnasium >= 0.28.1
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from ppo import Agent, make_env

# ── Hyperparameters (matching finetune_moral.py FineTuneArgs defaults) ──────
SEED = 42
NUM_STEPS = 128
TOTAL_TIMESTEPS = 1000 * NUM_STEPS   # 128_000  →  1000 iterations
NUM_ENVS = 1
NUM_MINIBATCHES = 4
UPDATE_EPOCHS = 8
LEARNING_RATE = 2.5e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_COEF = 0.2
CLIP_VLOSS = True
ENT_COEF = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
NORM_ADV = True
TARGET_KL = None
SAVE_EVERY = 5          # save checkpoint every N iterations
EXP_NAME = "ppo"

BATCH_SIZE = NUM_ENVS * NUM_STEPS
MINIBATCH_SIZE = BATCH_SIZE // NUM_MINIBATCHES
NUM_ITERATIONS = TOTAL_TIMESTEPS // BATCH_SIZE

# KL loss used for policy divergence penalty
kl_loss_fn = nn.KLDivLoss(reduction="sum", log_target=True)

ENVIRONMENTS = {
    "FindMilk": "environments.milk:FindMilk-v4",
    "Driving":  "environments.drive:Driving",
}

DEFAULT_KL_FACTORS = [0.0, 0.5, 1.0, 2.0, 2.5, 5.0, 10.0]


def train_one(env_id: str, kl_factor: float, seed: int = SEED):
    """Run one full fine-tuning with the given kl_penalty_factor."""
    env_short = env_id.split(":")[-1] if ":" in env_id else env_id
    base_model_path = f"models/{env_short}_{seed}/base.cleanrl_model"
    cache_path = f"models/{env_short}_{seed}/gpt-4o-mini_llm_cache.pickle"
    save_dir = f"models/{env_short}_{seed}/kl_sensitivity/kl_{kl_factor}"
    os.makedirs(save_dir, exist_ok=True)

    # ── Load LLM reward cache ──────────────────────────────────────────────
    assert os.path.isfile(cache_path), f"LLM cache not found: {cache_path}"
    with open(cache_path, "rb") as f:
        history = pickle.load(f)
    print(f"\n{'='*60}")
    print(f"  env={env_short}  kl_factor={kl_factor}  cache_states={len(history)}")
    print(f"  save_dir={save_dir}")
    print(f"{'='*60}")

    # ── Environment ────────────────────────────────────────────────────────
    kwargs = {"validate": True}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    envs = SyncVectorEnv(
        [make_env(env_id, i, False, "kl_sensitivity", **kwargs) for i in range(NUM_ENVS)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete)

    # ── Agent + reference agent ────────────────────────────────────────────
    agent = Agent(envs).to(device)
    agent.load_state_dict(torch.load(base_model_path, map_location=device))
    agent.critic = agent.reset_critic(envs).to(device)

    agent_ref = Agent(envs).to(device)
    agent_ref.load_state_dict(torch.load(base_model_path, map_location=device))

    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)

    # ── Storage ────────────────────────────────────────────────────────────
    obs      = torch.zeros((NUM_STEPS, NUM_ENVS) + envs.single_observation_space.shape).to(device)
    actions  = torch.zeros((NUM_STEPS, NUM_ENVS) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)
    rewards  = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)
    dones    = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)
    values   = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)

    # ── Tensorboard ────────────────────────────────────────────────────────
    writer = SummaryWriter(save_dir)

    # ── Rollout ────────────────────────────────────────────────────────────
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(NUM_ENVS).to(device)
    cache_misses = 0

    for iteration in range(1, NUM_ITERATIONS + 1):
        running_reward = []
        running_logprobs = []
        running_logprobs_ref = []

        for step in range(NUM_STEPS):
            global_step += NUM_ENVS
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                _, logprob_ref, _, _ = agent_ref.get_action_and_value(next_obs, action=action)
                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob
            running_logprobs.append(logprob)
            running_logprobs_ref.append(logprob_ref)

            # ── KL penalty ─────────────────────────────────────────────
            with torch.no_grad():
                lp_ft  = nn.functional.log_softmax(torch.Tensor(running_logprobs), dim=0)
                lp_ref = nn.functional.log_softmax(torch.Tensor(running_logprobs_ref), dim=0)
                kl = kl_loss_fn(lp_ft, lp_ref).detach().numpy()
            non_score_reward = -(kl_factor * kl)
            writer.add_scalar("charts/kl_div", kl, global_step)

            # ── RLHF reward from cache ─────────────────────────────────
            the_actions = action.cpu().numpy()
            RLHF_reward = 0.0
            for i in range(NUM_ENVS):
                envstate = envs.observations[i]
                key = tuple(envstate)
                if key in history:
                    reward_dict = history[key]["rewards"]
                    RLHF_reward = reward_dict[frozenset([str(the_actions[i])])]
                else:
                    cache_misses += 1
                    RLHF_reward = 0.0  # fallback for uncached states

            next_obs, env_reward, terminations, truncations, infos = envs.step(the_actions)
            running_reward.append(RLHF_reward)

            next_done_np = np.logical_or(terminations, truncations)
            reward = non_score_reward + RLHF_reward

            if any(next_done_np):
                running_reward = []
                running_logprobs = []
                running_logprobs_ref = []

            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.Tensor(next_done_np).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                        writer.add_scalar("charts/finetune_return", reward, global_step)
                        writer.add_scalar(f"charts/episodic_{info['metric1'][0]}", info["metric1"][1], global_step)
                        writer.add_scalar(f"charts/episodic_{info['metric2'][0]}", info["metric2"][1], global_step)

        # ── GAE + returns ──────────────────────────────────────────────────
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(NUM_STEPS)):
                if t == NUM_STEPS - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + GAMMA * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
            returns = advantages + values

        # ── Flatten ────────────────────────────────────────────────────────
        b_obs        = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs   = logprobs.reshape(-1)
        b_actions    = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns    = returns.reshape(-1)
        b_values     = values.reshape(-1)

        # ── PPO update ─────────────────────────────────────────────────────
        b_inds = np.arange(BATCH_SIZE)
        clipfracs = []
        for epoch in range(UPDATE_EPOCHS):
            np.random.shuffle(b_inds)
            for start in range(0, BATCH_SIZE, MINIBATCH_SIZE):
                end = start + MINIBATCH_SIZE
                mb_inds = b_inds[start:end]

                _, newlogprob, ent, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > CLIP_COEF).float().mean().item()]

                mb_adv = b_advantages[mb_inds]
                if NORM_ADV:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - CLIP_COEF, 1 + CLIP_COEF)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if CLIP_VLOSS:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -CLIP_COEF, CLIP_COEF
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = ent.mean()
                loss = pg_loss - ENT_COEF * entropy_loss + v_loss * VF_COEF

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                optimizer.step()

            if TARGET_KL is not None and approx_kl > TARGET_KL:
                break

        # ── Logging ────────────────────────────────────────────────────────
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        sps = int(global_step / (time.time() - start_time))
        writer.add_scalar("charts/SPS", sps, global_step)

        if iteration % 50 == 0 or iteration == NUM_ITERATIONS:
            print(f"  iter {iteration:4d}/{NUM_ITERATIONS}  SPS={sps}  cache_misses={cache_misses}")

        # ── Save checkpoint ────────────────────────────────────────────────
        if iteration % SAVE_EVERY == 0 or iteration == NUM_ITERATIONS:
            model_path = f"{save_dir}/{EXP_NAME}_{iteration}.cleanrl_model"
            torch.save(agent.state_dict(), model_path)

    envs.close()
    writer.close()
    print(f"  Training complete. Final model: {save_dir}/{EXP_NAME}_{NUM_ITERATIONS}.cleanrl_model")
    if cache_misses > 0:
        print(f"  WARNING: {cache_misses} cache misses (states not in LLM cache)")
    return save_dir


# ── Evaluation ─────────────────────────────────────────────────────────────
def evaluate_model(env_id: str, model_path: str, n_episodes: int = 50, seed: int = 42):
    """Run n_episodes and return list of metric dicts."""
    kwargs = {"validate": True}
    env = gym.make(env_id, render_mode="ansi", **kwargs)
    env = gym.wrappers.FlattenObservation(env)
    env.single_action_space = env.action_space
    env.single_observation_space = env.observation_space

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(env).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))

    results = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        obs = torch.Tensor(obs).to(device)
        done = False
        while not done:
            action, _, _, _ = agent.get_action_and_value(obs)
            state, reward, terminated, truncated, info = env.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            obs = torch.Tensor(state).to(device)

        metrics = env.log()
        results.append({
            "timesteps": info.get("episode", {}).get("l", 0),
            "main_goal_name": metrics["main_goal"][0],
            "main_goal": metrics["main_goal"][1],
            "metric_1_name": metrics["metric1"][0],
            "metric_1": metrics["metric1"][1],
            "metric_2_name": metrics["metric2"][0],
            "metric_2": metrics["metric2"][1],
        })
    env.close()
    return results


def run_sensitivity(env_names=None, kl_factors=None, skip_training=False):
    """Full pipeline: train + evaluate for each (env, kl_factor) pair."""
    import pandas as pd

    if env_names is None:
        env_names = list(ENVIRONMENTS.keys())
    if kl_factors is None:
        kl_factors = DEFAULT_KL_FACTORS

    all_dfs = {}
    for env_name in env_names:
        env_id = ENVIRONMENTS[env_name]
        env_short = env_id.split(":")[-1]
        env_rows = []

        for kl_f in kl_factors:
            save_dir = f"models/{env_short}_{SEED}/kl_sensitivity/kl_{kl_f}"
            final_model = f"{save_dir}/{EXP_NAME}_{NUM_ITERATIONS}.cleanrl_model"

            # ── Train ──────────────────────────────────────────────────
            if not skip_training:
                train_one(env_id, kl_f, SEED)
            elif not os.path.isfile(final_model):
                print(f"  SKIP: model not found for kl={kl_f}, training...")
                train_one(env_id, kl_f, SEED)

            # ── Evaluate at several checkpoints ────────────────────────
            checkpoints = list(range(SAVE_EVERY, NUM_ITERATIONS + 1, SAVE_EVERY))
            # Evaluate only every 10th checkpoint to save time, plus the last
            eval_checkpoints = [c for c in checkpoints if c % 10 == 0 or c == NUM_ITERATIONS]

            for ckpt in eval_checkpoints:
                model_path = f"{save_dir}/{EXP_NAME}_{ckpt}.cleanrl_model"
                if not os.path.isfile(model_path):
                    continue

                results = evaluate_model(env_id, model_path, n_episodes=50, seed=SEED)
                for r in results:
                    r["episode"] = ckpt
                    r["kl_factor"] = kl_f
                    r["model"] = f"kl={kl_f}"
                    r["env_name"] = env_name
                    env_rows.append(r)

                print(f"  Evaluated {env_name} kl={kl_f} ckpt={ckpt}: "
                      f"main_goal={np.mean([r['main_goal'] for r in results]):.2f}")

        # Save per-environment CSV
        df = pd.DataFrame(env_rows)
        out_path = f"notebooks/data_kl_sensitivity_{env_name.lower()}.csv"
        df.to_csv(out_path, index=False)
        print(f"\nResults saved to {out_path}  ({len(df)} rows)")
        all_dfs[env_name] = df

    return all_dfs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KL penalty factor sensitivity analysis")
    parser.add_argument("--env", type=str, default=None,
                        choices=list(ENVIRONMENTS.keys()),
                        help="Run for a single environment (default: both)")
    parser.add_argument("--kl-factors", nargs="+", type=float, default=DEFAULT_KL_FACTORS,
                        help=f"KL penalty factors to sweep (default: {DEFAULT_KL_FACTORS})")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training, only evaluate existing models")
    parser.add_argument("--eval-only", action="store_true",
                        help="Alias for --skip-training")
    args = parser.parse_args()

    env_names = [args.env] if args.env else None
    kl_factors = args.kl_factors
    skip = args.skip_training or args.eval_only

    run_sensitivity(env_names=env_names, kl_factors=kl_factors, skip_training=skip)
