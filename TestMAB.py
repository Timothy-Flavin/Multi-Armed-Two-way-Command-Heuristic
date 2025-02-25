from flexibuddiesrl import DQN, PG, Agent
import argparse
import os
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import torch
from typing import Callable
from wrapped_environments import (
    CartpoleWrapped,
    OvercookWrapped,
    Wrapper,
    TTTWrapped,
    TTTWrappedRoles,
    TTTLeverWrapped,
)  # cartpole env for testing
from enum import Enum
from flexibuff import FlexibleBuffer, FlexiBatch
from MABTheory import MATCH


GREED = True
# keyboard.hook(handle_key_event)


class Episode_Type(Enum):
    RL = 1
    EVAL = 4
    MATCH_EVAL = 5


# from wrapped_environments import CartpoleWrapped, MatrixGame
# Takes the kind of run, RL, Human demo, test, etc and calls the right action handlers
# model.get_action or human_input_handler. etc
def handle_actions(
    model: Agent,
    obs,
    avail_actions,
    episode_type,
    agent_id,
    human_renderer: Callable,
    human_input_processor: Callable,
    epsilon=0.1,
):
    action = 0
    lp = -1.0

    if episode_type == Episode_Type.EVAL:
        with torch.no_grad():
            discrete_actions, continuous_actions, dlp, clp, value = model.train_actions(
                obs, avail_actions, step=True
            )
            action = discrete_actions[0]  # TODO: Make env wrapper process this better
            if isinstance(dlp, int):
                # Because non policy-gradient algorithms return 0
                lp = dlp
            else:
                lp = dlp[0]

    elif episode_type == Episode_Type.RL:
        discrete_actions, continuous_actions, dlp, clp, value = model.train_actions(
            obs, avail_actions, step=True
        )
        action = discrete_actions[0]  # TODO: Make this handle both action spaces
        if isinstance(dlp, int):
            lp = dlp  # For non policy gradient methods that return zero
        else:
            lp = dlp[0]
    return int(action), lp


def actions_match(
    memory: FlexibleBuffer,
    models: list[Agent],
    obs,
    n_agents,
    n_actions,
    env: Wrapper,
    match_modules: list[MATCH],
    n_steps=1,
    current_step=0,
    device="cuda:0",
    verbose=False,
    gamma=0.99,
    priors=[None, None],
    share_listener_rewards: bool = False,
    n_same_actions_required: int = 1,
):
    with torch.no_grad():
        if verbose:
            print("\n\n\nBefore Everything: ")
            print(match_modules[0])
            print(match_modules[1])
        if not torch.is_tensor(obs):
            obs = torch.from_numpy(obs).float().to(device)

        command_targets = np.zeros((n_agents, n_agents))
        command_contents = np.zeros((n_agents, n_agents))
        actions = np.zeros(shape=n_agents, dtype=np.int64)
        la = np.ones(shape=(n_agents, n_actions))
        for agent_id in range(n_agents):
            avail = env.get_avail_agent_actions(agent_id)
            if avail is not None:
                la[agent_id] = np.array(avail)
        if verbose:
            print(f"la: {la}")

        # Update the match modules with the last n_steps of memory if this is a match update step
        if current_step != 0 and current_step % n_steps == 0:
            speaker_adv = np.zeros((n_agents, n_agents))
            for agent_id in range(n_agents):
                command_targets[agent_id] = match_modules[agent_id].targets
            if verbose:
                print(f"Updating MATCH from Command matrix: {command_targets}")
            idx = np.arange(memory.idx - n_steps, memory.idx)
            for ix in range(idx.shape[0]):
                if idx[ix] < 0:
                    idx[ix] += memory.mem_size
            if verbose:
                print(f"idx: {idx}")
            for agent_id in range(n_agents):
                # Get the steps that we have recorded since last match update
                adv, arms = match_modules[agent_id].calc_reward(
                    buffer=memory,
                    agent=models[agent_id],
                    idx=idx,
                    agent=models[agent_id],
                    adv_type="gae",
                    legal_actions=True,
                    device=device,
                    gamma=gamma,
                    share_listener_rewards=share_listener_rewards,
                    n_same_actions_required=n_same_actions_required,
                )
                if share_listener_rewards:
                    arm_bool = arms > 0
                else:
                    arm_bool = arms == np.max(
                        arms
                    )  # maybe replace this with armbool[selected]=True
                match_modules[agent_id].update_listener(
                    adv=adv, arms=arm_bool, verbose=verbose
                )
                speaker_adv[:, agent_id] = arms
                if verbose:
                    print(
                        f"  a_id: \n  {agent_id}\n  adv: \n  {adv},\n arms: \n  {arms}\n  speaker adv: \n  {speaker_adv[:, agent_id]}"
                    )
            augmented_targets = command_targets.copy()
            for agent_id in range(n_agents):
                augmented_targets[agent_id, agent_id] = 1
            for agent_id in range(n_agents):
                match_modules[agent_id].update_speaker(
                    adv=speaker_adv[agent_id],
                    verbose=verbose,
                    n_options=augmented_targets.sum(axis=0).flatten() - 1,
                )

        # Set up the command matrix from scratch if this is a command step
        if current_step % n_steps == 0:
            if verbose:
                print("\nUpdating content and targets because it's an update episode")
            # later will be 3d for actions with more than 1 number
            for agent_id in range(n_agents):
                # Get the command targets from each agent 'agent_id'
                command_targets[agent_id] = match_modules[agent_id].choose_target(
                    prior=priors[agent_id], available_teammates=np.ones(n_agents)
                )

                # Get the command contents for each agent
                for target in range(n_agents):
                    if target == agent_id or command_targets[agent_id, target] > 0:
                        act, _, lp, __, qv = models[agent_id].train_actions(
                            obs[target], la[target]
                        )
                        command_contents[agent_id, target] = act[
                            0
                        ]  # TODO: act is a tensor for multi output domains

            for agent_id in range(n_agents):
                match_modules[agent_id].command_contents = command_contents[agent_id]

        # If it is not an update turn, just use the last command targets but current command contents
        else:
            if verbose:
                print("\nUpdating command content but not targets")
            for agent_id in range(n_agents):
                command_targets[agent_id] = match_modules[agent_id].targets
                for target in range(n_agents):
                    if target == agent_id or command_targets[agent_id, target] > 0:
                        act, _, lp, __, qv = models[agent_id].train_actions(
                            obs[target], la[target]
                        )
                        command_contents[agent_id, target] = act[0]
        if verbose:
            print(f"command_targets: {command_targets}")
            print(f"command_contents: {command_contents}")

        # choose who to listen to and update oracles from last time?
        for agent_id in range(n_agents):
            if current_step % n_steps == 0:  # only set my value on an update step
                options = command_targets[:, agent_id].copy().flatten()
                action, leader = match_modules[agent_id].policy_with_oracle(
                    commanded_by=options,
                    prior=None,
                    told_to=command_contents[:, agent_id].copy().flatten(),
                )
                if verbose:
                    print(
                        f"a: {agent_id}, chosing new action {action} from: {options}, leader: {match_modules[agent_id].selected}"
                    )
            else:  # Keep listening to the same one if this is not an update step
                action = command_contents[match_modules[agent_id].selected, agent_id]
                if verbose:
                    print(
                        f"not updating, listening to [{action}] from the same one as last time: {match_modules[agent_id].selected}"
                    )
            actions[agent_id] = action
        if verbose:
            input()
    return actions, la, 0


def actions_no_match(
    memory,
    models,
    episode_type,
    epsilon,
    obs,
    n_agents,
    n_actions,
    env: Wrapper,
):
    actions = np.zeros(shape=n_agents, dtype=np.int64)
    la = np.ones(shape=(n_agents, n_actions))
    if memory.discrete_log_probs is not None:
        log_probs = np.ones(shape=(n_agents, 1))
    else:
        log_probs = None
    for agent_id in range(n_agents):
        avail = env.get_avail_agent_actions(agent_id)
        avail_actions = np.array(avail) if avail is not None else None
        if avail_actions is not None:
            la[agent_id] = avail_actions
        action, log_prob = handle_actions(
            models[agent_id],
            obs[agent_id],
            avail_actions,
            episode_type,
            agent_id,
            env.display,
            env.human_action,
            epsilon,
        )
        if log_probs is not None:
            log_probs[agent_id] = log_prob
        actions[agent_id] = action
    return actions, la, log_probs


# This function is pretty long but there is a lot of repeat code otherwise
def run_multi_agent_episodes(
    env: CartpoleWrapped,
    models: list[Agent],
    n_agents,
    episode_type: Episode_Type,
    memory: FlexibleBuffer,
    imitation_memory: FlexibleBuffer = None,
    max_steps=5000,
    epsilon=0.1,  # For Egreedy episodes
    supervised_reg=False,  # If true, will learn supervised objective as well
    expert_reward=False,
    episodic=False,
    learn_during_rand=False,
    display=False,
    graph_progress=False,
    reward_checkpoint_bin=[10, 40, 60, 70, 80, 100],  #   #
    model_path="",
    use_match=False,
    n_shot=1,
    n_step=5,
    online=False,
    verbose=False,
    save_models=False,
):
    print(
        f"Running {episode_type} episodes online: {online}, episodic: {episodic} ms:{max_steps}"
    )
    # input()
    current_episode = 0
    current_episode_step = 0
    overall_step = 0
    ep_reward_hist = []
    smooth_reward_hist = []
    ep_expert_reward_hist = []
    smooth_expert_reward_hist = []
    recent_rewards = deque(maxlen=50)
    recent_expert_rewards = deque(maxlen=50)
    human_likeness = []
    loss_hist = []

    while overall_step < max_steps:
        # print(
        #    f"overall: {overall_step} ep# {current_episode}"
        # )  # ep step: {current_episode_step}")
        idx_before = memory.idx
        steps_recorded_before = memory.steps_recorded
        ep_inds = None if memory.episode_inds is None else memory.episode_inds.copy()
        ep_lens = None if memory.episode_lens is None else memory.episode_lens.copy()
        done = False
        obs, info = env.reset()
        ep_reward_hist.append(0)
        ep_expert_reward_hist.append(0)
        loss_hist.append(0)
        current_episode_step = 0

        if current_episode % n_shot == 0:
            matches = (
                [  # TODO: Make match have a reset function instead of making new ones
                    MATCH(n_agents, i, single=False, lambda_=0.90, gamma=0.90)
                    for i in range(n_agents)
                ]
            )

        while not done:
            actions, la = None, None

            actions, la, log_probs = actions_match(
                memory=memory,
                models=models,
                obs=obs,
                n_agents=n_agents,
                n_actions=n_actions,
                env=env,
                match_modules=matches,
                n_steps=n_step,
                current_step=current_episode_step,
                device=device,
                verbose=verbose,
                share_listener_rewards=False,
                n_same_actions_required=1,
            )
            # input()
            if display:
                env.display(obs, None, 0)

            obs_, reward, terminated, truncated, info = env.step(actions)

            # env.display(obs, avail_actions, agent_id)
            ep_reward_hist[current_episode] += reward
            done = terminated or truncated
            er = 0
            if expert_reward:
                er = env.expert_reward(obs)
                ep_expert_reward_hist[current_episode] += er

            # print(actions)
            actions = actions.reshape([actions.shape[0], 1])
            # print(actions)
            # print(env.env)

            memory.save_transition(
                terminated=terminated,
                action_mask=[la],  # list for action dims
                registered_vals={
                    "obs": obs,
                    "obs_": obs_,
                    "discrete_actions": actions,
                    "global_rewards": reward,
                    "global_auxiliary_rewards": er,
                    "discrete_log_probs": log_probs,
                },
            )

            if episode_type == Episode_Type.RL:
                inds = np.arange(n_agents)
                np.random.shuffle(inds)
                # if episodic:
                if online and memory.steps_recorded > 999:
                    exp = memory.sample_transitions(
                        as_torch=True,
                        device=device,
                        idx=np.arange(0, memory.steps_recorded),
                    )
                    for agent_id in inds:
                        aloss, closs = models[agent_id].reinforcement_learn(
                            batch=exp, agent_num=agent_id
                        )
                    memory.reset()
                    loss_hist[-1] += closs

                elif not online and memory.steps_recorded > 255:
                    exp = memory.sample_transitions(256, as_torch=True, device=device)
                    for agent_id in inds:
                        aloss, closs = models[agent_id].reinforcement_learn(
                            exp, agent_id
                        )
                    loss_hist[-1] += closs

            overall_step += 1
            current_episode_step += 1
            obs = obs_

        # if current_episode % n_shot == (n_shot - 1):
        recent_rewards.append(ep_reward_hist[-1])
        recent_expert_rewards.append(ep_expert_reward_hist[-1])

        if len(smooth_reward_hist) > 10:
            er = smooth_reward_hist[-1]
            smooth_reward_hist.append(
                0.98 * er + 0.02 * ep_reward_hist[current_episode]
            )

        else:
            smooth_reward_hist.append(sum(recent_rewards) / len(recent_rewards))
        # same for expert rewards:
        if len(smooth_expert_reward_hist) > 10:
            smooth_expert_reward_hist.append(
                0.98 * smooth_expert_reward_hist[-1] + 0.02 * ep_expert_reward_hist[-1]
            )
        else:
            smooth_expert_reward_hist.append(
                sum(recent_expert_rewards) / len(recent_expert_rewards)
            )

        if current_episode % 50 == 0:
            print(
                f"Episode: {current_episode:<4}  "
                f"Episode steps: {current_episode_step:<4}  "
                f"Return: {ep_reward_hist[current_episode]:<5.1f} smooth: {smooth_reward_hist[-1]}"
            )

        if (
            save_models
            and episode_type == Episode_Type.RL
            and len(reward_checkpoint_bin) > 0
        ):
            mbin = min(reward_checkpoint_bin)
            # print(smooth_reward_hist[-1])
            # print(mbin)
            if smooth_reward_hist[-1] > mbin:
                print(
                    "Checkpointing model at : ",
                    model_path + f"r_{mbin}_supreg{supervised_reg}/",
                )
                models[0].save(
                    model_path + f"r_{mbin}_supreg{supervised_reg}_{args.runid}/"
                )
                reward_checkpoint_bin.pop(reward_checkpoint_bin.index(mbin))

        if current_episode % 500 == 0 and current_episode > 1 and graph_progress:
            smr = np.array(smooth_reward_hist)
            smer = np.array(smooth_expert_reward_hist)
            # m = max(np.max(smr), np.max(smer))

            # print(human_likeness)
            plt.plot(smr, color=[0, 0.0, 0.9])
            plt.plot(smer, color=[0.5, 0.5, 0.9])
            plt.plot(np.array(human_likeness).mean(axis=-1), color=[0.2, 0.9, 0.2])
            plt.plot(loss_hist / np.max(np.array(loss_hist)), color=[0.9, 0.2, 0.9])
            plt.hlines(
                1 / 6.0,
                colors=[[0.1, 0.9, 0.3]],
                xmin=0,
                xmax=smr.shape[0] - 1,
                linestyles=["dashed"],
            )
            plt.hlines(0, colors=[[0.1, 0.1, 0.1]], xmin=0, xmax=smr.shape[0] - 1)
            plt.hlines(1.0, colors=[[0.1, 0.1, 0.1]], xmin=0, xmax=smr.shape[0] - 1)

            plt.legend(
                [
                    "Smooth rewards",
                    "Expert rewards",
                    "Human Likeness",
                    f"Loss {np.max(np.array(loss_hist))}",
                    "Random Human Likeness",
                    "zero",
                    "one",
                ]
            )
            plt.title("Three Objective history")
            plt.ylabel("Reward")
            plt.xlabel("episode")
            plt.grid()
            fig = plt.gcf()
            plt.show()
            file = input("save to? ")
            fig.savefig(file)

        current_episode += 1

        # if MATCH:
        # print("Samplers summary: ")
        # for aid in range(n_agents):
        #    MATCH_Wrappers[aid].final_dir(prior=None)
    if episode_type == Episode_Type.RL or episode_type == Episode_Type.EVAL:
        return (
            np.array(smooth_reward_hist),
            np.array(smooth_expert_reward_hist),
            np.array(human_likeness),
        )


def get_agent(obs, args, device, n_actions):
    return {
        "DQ": DQN(
            obs_dim=obs.shape[1],
            continuous_action_dims=0,  # continuous_env.action_space.shape[0],
            max_actions=None,  # continuous_env.action_space.high,
            min_actions=None,  # continuous_env.action_space.low,
            discrete_action_dims=[n_actions],
            hidden_dims=[64, 64],
            device=device,
            lr=1e-4,
            activation="relu",
            dueling=True,
            n_c_action_bins=0,
        ),
        "SDQ": DQN(
            obs_dim=obs.shape[1],
            continuous_action_dims=0,  # continuous_env.action_space.shape[0],
            max_actions=None,  # continuous_env.action_space.high,
            min_actions=None,  # continuous_env.action_space.low,
            discrete_action_dims=[n_actions],
            hidden_dims=[64, 64],
            device=device,
            lr=1e-4,
            activation="relu",
            dueling=True,
            n_c_action_bins=0,
            entropy=0.03,
        ),
        "MDQ": DQN(
            obs_dim=obs.shape[1],
            continuous_action_dims=0,  # continuous_env.action_space.shape[0],
            max_actions=None,  # continuous_env.action_space.high,
            min_actions=None,  # continuous_env.action_space.low,
            discrete_action_dims=[n_actions],
            hidden_dims=[64, 64],
            device=device,
            lr=1e-4,
            activation="relu",
            dueling=True,
            n_c_action_bins=0,
            entropy=0.03,
            munchausen=0.9,
        ),
        "PPO": PG(
            obs_dim=obs.shape[1],
            discrete_action_dims=[n_actions],
            # continuous_action_dim=continuous_env.action_space.shape[0],
            hidden_dims=np.array([96, 64]),
            # min_actions=continuous_env.action_space.low,
            # max_actions=continuous_env.action_space.high,
            gamma=0.977,
            device=device,
            entropy_loss=0.001,
            mini_batch_size=256,
            n_epochs=4,
            lr=1e-3,
            advantage_type="gae",
            norm_advantages=True,
            anneal_lr=2000000,
            value_loss_coef=0.05,
            ppo_clip=0.15,
            # value_clip=0.5,
            orthogonal=True,
            activation="tanh",
            starting_actorlogstd=0,
            gae_lambda=0.8,
        ),
        "PG": PG(
            obs_dim=obs.shape[1],
            discrete_action_dims=[n_actions],
            hidden_dims=np.array([64, 64]),
            gamma=0.99,
            device=device,
            entropy_loss=0.01,
            mini_batch_size=64,
            n_epochs=1,
            lr=3e-4,
            advantage_type="gae",
            norm_advantages=False,
            anneal_lr=200000,
            value_loss_coef=0.5,
            orthogonal=True,
            activation="tanh",
            starting_actorlogstd=0,
            gae_lambda=0.95,
        ),
    }


def get_env(args, results_path):
    if args.env == "overcooked":
        env = OvercookWrapped(render=True)
        obs = env.get_obs()
        n_actions = 6
        n_agents = 2
        reward_bin = [10, 40, 60, 70, 80, 100]

    elif args.env == "cartpole":
        env = CartpoleWrapped()
        n_actions = env.n_actions
        n_agents = env.n_agents
        obs, states = env.reset()
        results_path += "Cartpole/"

    elif args.env == "ttt":
        env = TTTWrapped(
            nfirst=2, n_moves=2, render_mode=None, random_op=True, obs_as_array=True
        )
        n_actions = 9
        n_agents = 2
        results_path += "TTT/"
        obs, info = env.reset()
        reward_bin = [
            0.0,
            0.2,
            0.4,
            0.6,
            0.7,
            0.8,
            0.9,
            0.95,
        ]
    elif args.env == "ttt_roles":
        env = TTTWrappedRoles(
            nfirst=2, n_moves=2, render_mode=None, random_op=True, obs_as_array=True
        )
        n_actions = 9
        n_agents = 2
        results_path += "TTTRoles/"
        obs, info = env.reset()
        reward_bin = [
            0.0,
            0.2,
            0.4,
            0.6,
            0.7,
            0.8,
            0.9,
            0.95,
        ]
    elif args.env == "ttt_lever":
        env = TTTLeverWrapped(
            nfirst=2, n_moves=2, render_mode=None, random_op=True, obs_as_array=True
        )
        n_actions = 9
        n_agents = 2
        results_path += "TTTLever/"
        obs, info = env.reset()
        reward_bin = [
            0.0,
            0.2,
            0.4,
            0.6,
            0.7,
            0.8,
            0.9,
            0.95,
        ]
    return env, n_actions, n_agents, results_path, obs, reward_bin


def organize_models(args):
    res_paths = []
    dir_lists = []
    paths = open(args.model_paths, "r").readlines()
    i = 0
    while i < len(paths):
        print(paths[i], "  respaths")
        if paths[i] == "" or paths[i] == "\n":
            i += 1
            break
        res_paths.append(paths[i].replace("\n", ""))
        dir_lists.append([])
        i += 1
    for j in range(len(dir_lists)):
        while i < len(paths):
            print(paths[i].replace("\n", ""))
            if paths[i] == "" or paths[i] == "\n":
                i += 1
                break
            dir_lists[j].append(paths[i].replace("\n", ""))
            i += 1

    model_dirs = []
    graph_names = []
    algos = []
    # print(res_paths)
    # print(dir_lists)
    # input("hmm")
    for i, rp in enumerate(res_paths):
        for d in dir_lists[i]:
            # if d[-1] != "0":
            print(d[2:6].replace("_", ""))
            if float(d[2:6].replace("_", "")) <= 0.7:
                continue
            # if d[-1] != "0":
            # continue
            # print(d)
            if os.path.isdir(rp + d):
                print(f"is dir: {rp+d}")
                model_dirs.append(rp + d)
                graph_names.append(d[0:6] + "_" + d[-1] + model_fams[i])
                algos.append(model_fams[i])

    # sort model names dir names and algos by float(modeldirs[i][2:5])

    for md in graph_names:
        print(md)
        print(md[8:11])
        print(float(md[2:5]) + (1000 if md[8:11] == "MDQ" else 0))

    if True:
        model_dirs = [
            x
            for _, x in sorted(
                zip(
                    [
                        (float(md[2:5]) + (1000 if md[8:11] == "MDQ" else 0))
                        for md in graph_names
                    ],
                    model_dirs,
                )
            )
        ]

        algos = [
            x
            for _, x in sorted(
                zip(
                    [
                        (float(md[2:5]) + (1000 if md[8:11] == "MDQ" else 0))
                        for md in graph_names
                    ],
                    algos,
                )
            )
        ]
        graph_names = [
            x
            for _, x in sorted(
                zip(
                    [
                        (float(md[2:5]) + (1000 if md[8:11] == "MDQ" else 0))
                        for md in graph_names
                    ],
                    graph_names,
                )
            )
        ]
    return graph_names, model_dirs, algos


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="Multi-Agent MAB tuner / runner",
        description="This program trains MARL models \
            on wrapped environments with additional human \
            data. The effects of the human data curation \
            are then recorded and graphed for comparison",
        epilog="Text at the bottom of help",
    )

    parser.add_argument(
        "-e",
        "--env",
        action="store",
        choices=["cartpole", "overcooked", "ttt", "ttt_roles", "ttt_lever"],
    )
    parser.add_argument("-rid", "--runid", action="store", default=0)
    parser.add_argument("-nsh", "--n_shot", action="store", default=0)
    parser.add_argument("-nst", "--n_step", action="store", default=0)
    parser.add_argument("-rls", "--rlsteps", action="store", default=0)
    parser.add_argument("-g", "--graph", action="store_true")
    parser.add_argument(
        "-cuda", "--cuda_device", action="store", choices=["cuda:0", "cuda:1", "cpu"]
    )
    parser.add_argument("-eval", "--evaluate_pairwise", action="store_true")
    parser.add_argument_group("-m_fams", "--model_families", action="store", nargs="+")
    parser.add_argument(
        "-paths", "--model_paths", action="store", default="model_paths.txt"
    )

    args = parser.parse_args()

    model_fams = args.model_families

    reward_bin = []
    arg_to_env_str = {
        "cartpole": "Cartpole",
        "overcooked": "Overcooked",
        "ttt": "TTT",
        "ttt_roles": "TTTRoles",
        "ttt_lever": "TTTLever",
    }
    results_path = "./PaperExperiment/"

    env, n_actions, n_agents, results_path, obs, reward_bin = get_env(
        args, results_path
    )

    print(env.get_state_feature_names())
    for r in model_fams:
        rp = results_path + f"algo_{r}/model_checkpoints/"
        print(f"avail results_path: {rp}")
        print(f"resulting model directories: {os.listdir(rp)}")
    device = torch.device(args.cuda_device if torch.cuda.is_available() else "cpu")

    graph_names, model_dirs, algos = organize_models()

    the_agents = get_agent(obs, args, device, n_actions)
    the_agents2 = get_agent(obs, args, device, n_actions)
    # two sets so we don't accidentally parameter share by loading in-place
    mean_scores = np.zeros((len(model_dirs), len(model_dirs)))
    last_scores = np.zeros((len(model_dirs), len(model_dirs)))

    mem = FlexibleBuffer(
        num_steps=5000,
        track_action_mask=False,
        discrete_action_cardinalities=[n_actions],
        path=results_path,
        name="test_" + args.env,
        n_agents=n_agents,
        individual_registered_vars={
            "discrete_log_probs": ([1], np.float32),
            "obs": ([obs.shape[1]], np.float32),
            "obs_": ([obs.shape[1]], np.float32),
            "discrete_actions": ([1], np.int64),
        },
        global_registered_vars={
            "global_rewards": (None, np.float32),
            "global_auxiliary_rewards": (None, np.float32),
        },
    )
    for i in range(len(model_dirs)):
        for j in range(0, len(model_dirs)):
            print(f"loading {model_dirs[i]}")
            a1: Agent = the_agents[algos[i]]  # fix this
            a1.load(model_dirs[i] + "/")
            a1.eval_mode = True
            print(f"loading {model_dirs[j]}")
            a2: Agent = the_agents2[algos[j]]
            a2.load(model_dirs[j] + "/")
            a2.eval_mode = True

            rew, er, hl = run_multi_agent_episodes(
                env=env,
                models=[a1, a2],
                n_agents=2,
                episode_type=Episode_Type.RL,
                memory=mem,
                imitation_memory=mem,
                max_steps=(
                    10 if args.env not in ["ttt", "ttt_roles", "ttt_lever"] else 5
                )
                * 200,
                MATCH=True,
                n_shot=int(args.n_shot),
                n_step=int(args.n_step),
                save_models=False,
                online=True,
                episodic=False,
            )
            rew = np.array(rew)
            mean_scores[i, j] = rew.mean()
            print(mean_scores[i])
            last_scores = rew[
                (np.arange(rew.shape[0]) % args.nshot) == (args.nshot - 1)
            ].mean()
            print(f"{i*len(model_dirs)+j}/{len(model_dirs)**2}")

    np.save(f"score_trial_greed0_{args.n_shot}_{args.n_step}", mean_scores)
    # Set ticks and labels

    if args.graph:
        fig, ax = plt.subplots()
        im = ax.imshow(mean_scores)
        ax.set_xticks(np.arange(len(graph_names)))
        ax.set_yticks(np.arange(len(graph_names)))
        ax.set_xticklabels(graph_names)
        ax.set_yticklabels(graph_names)

        # Rotate the x-tick labels for better readability
        fig.colorbar(im, ax=ax)
        plt.title("Scores of different paired agents")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        plt.show()
    exit()
    if args.env == "cartpole":
        dirpath = "./PaperExperiment/Cartpole/"
    elif args.env == "overcooked":
        dirpath = "./PaperExperiment/Overcooked/"
    elif args.env == "ttt":
        dirpath = "./PaperExperiment/TTT/"
    elif args.env == "ttt_roles":
        dirpath = "./PaperExperiment/TTTRoles/"
    elif args.env == "ttt_lever":
        dirpath = "./PaperExperiment/TTTLever/"
    dirpath = dirpath + f"algo_{args.algorithm}/"

    rew, exprew, hl = run_multi_agent_episodes(
        env=env,
        models=[model, model],
        n_agents=n_agents,
        memory=r_mem,
        imitation_memory=h_mem,
        episode_type=Episode_Type.RL,
        max_steps=int(args.rlsteps),
        supervised_reg=args.supreg,
        graph_progress=args.graph,
        model_path=dirpath + "model_checkpoints/",
        online=online,
        episodic=episodic,
        reward_checkpoint_bin=reward_bin,
    )

    filename = f"demo_{args.demo}_human_reg{args.supreg}_hf_{args.hum_feedback}_run{args.runid}_"

    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    print("Saving to " + dirpath + filename)
    np.save(dirpath + filename + "human_likeness.npy", hl)
    np.save(dirpath + filename + "rew.npy", rew)
    np.save(dirpath + filename + "exprew.npy", exprew)
