from flexibuddiesrl import DQN, PG, TD3, DDPG, Agent
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
from MABSamplers import Agent_With_Oracle


class Episode_Type(Enum):
    RAND = 0
    RL = 1
    HUMAN = 3
    EVAL = 4
    EGREEDY = 5  # RL discrete action but with some randomness added in


keys_down = {}


def handle_key_event(key_event):
    if key_event.name not in keys_down:
        keys_down[key_event.name] = True
    else:
        keys_down[key_event.name] = not keys_down[key_event.name]
    # print(key_event.name)


GREED = True
# keyboard.hook(handle_key_event)


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
    # print(episode_type)
    # print(f"obs: {obs}, avail actions: {avail_actions} \n\n\n")
    if episode_type == Episode_Type.RAND:
        action = np.random.choice(
            a=np.arange(avail_actions.shape[0]),
            p=avail_actions / np.sum(avail_actions),
        )
        # print(f"Rand action: {action}")
    elif episode_type == Episode_Type.HUMAN:
        human_renderer(obs, avail_actions, agent_id)
        action = human_input_processor(obs, avail_actions, agent_id, keys_down)
        print(f"Human action: {action}")
    elif episode_type == Episode_Type.EVAL:
        with torch.no_grad():
            # action = model.deterministic_action(obs, avail_actions)
            # action, lp, qold = model.train_actions(obs, avail_actions, step=True)
            discrete_actions, continuous_actions, dlp, clp, value = model.train_actions(
                obs, avail_actions, step=True
            )
            action = discrete_actions[0]  # TODO: Make env wrapper process this better
            if isinstance(dlp, int):
                # Because non policy-gradient algorithms return 0
                lp = dlp
            else:
                lp = dlp[0]
            # print(f"Model action: {action}")
    elif episode_type == Episode_Type.EGREEDY:
        if np.random.random() < epsilon:
            action = np.random.choice(
                a=np.arange(avail_actions.shape[0]),
                p=avail_actions / np.sum(avail_actions),
            )  # env.action_space.sample()#model.get_action(obs, avail_actions)
            # print(f"E Greedy Rand action: {action}")
        else:
            action = model.ego_actions(obs, avail_actions)
            # print(f"E Greedy model action: {action}")
    elif episode_type == Episode_Type.RL:
        # action, lp, qold
        discrete_actions, continuous_actions, dlp, clp, value = model.train_actions(
            obs, avail_actions, step=True
        )
        action = discrete_actions[0]  # TODO: Make this handle both action spaces
        if isinstance(dlp, int):
            lp = dlp  # For non policy gradient methods that return zero
        else:
            lp = dlp[0]
        # print(obs)
        # print(action)
        # print(f"RL schochastic action: {action}")
    # print(action, lp)
    return int(action), lp


def check_discrete_human_likeness(model: Agent, m_batches: list, agent_id=0):
    # print(model.hum_mem_buff.steps_recorded)

    n_right = 0
    n_total = 0
    batch: FlexiBatch
    for batch in m_batches:
        mask = None
        if batch.action_mask is not None:
            mask = batch.action_mask[0][
                agent_id
            ]  # gets action mask for discrete output 0 and this agent
        model_ac = model.ego_actions(
            observations=batch.obs[agent_id], legal_actions=mask
        )
        n_total += batch.discrete_actions.shape[1]

        n_right += np.sum(
            model_ac == batch.discrete_actions[agent_id, :, 0].cpu().numpy()
        ).item()
    return n_right / n_total


def actions_match(
    memory,
    models,
    episode_type,
    epsilon,
    obs,
    n_agents,
    n_actions,
    env: Wrapper,
    MATCH: list[Agent_With_Oracle],
    n_steps=1,
    current_step=0,
    device="cuda:0",
    verbose=False,
):
    if verbose:
        print("\n\n\nBefore Everything: ")
        print(MATCH[0])
        print(MATCH[1])
    if not torch.is_tensor(obs):
        obs = torch.from_numpy(obs).float().to(device)

    actions = np.zeros(shape=n_agents, dtype=np.int64)
    la = np.ones(shape=(n_agents, n_actions))
    for agent_id in range(n_agents):
        avail = env.get_avail_agent_actions(agent_id)
        avail_actions = np.array(avail) if avail is not None else None
        if avail_actions is not None:
            la[agent_id] = avail_actions

    # print(f"legal_actions: {la}")
    # update listener / incoming MAB from last step
    if current_step != 0 and current_step % n_steps == 0:
        if verbose:
            print(
                f"Current step {current_step} not zero and % nsteps ({n_steps}) is zero so updating value"
            )
        for agent_id in range(n_agents):
            disc = 1.0
            target = 0
            for r in reversed(MATCH[agent_id].reward):
                target *= MATCH[agent_id].gamma  #
                disc *= MATCH[agent_id].gamma
                target += r
            if verbose:
                print(
                    f"aid{agent_id} rewards: {MATCH[agent_id].reward} discounted to {target}"
                )
            ev = MATCH[agent_id].agent.expected_V(obs[agent_id], la[agent_id])
            target += disc * ev
            if verbose:
                print(f"ev: {ev}, discounted: {ev*disc}")
                print(
                    f"target + disc ev: {target} where previous value: {MATCH[agent_id].value} for listening to {MATCH[agent_id].leader}"
                )
            MATCH[agent_id].update_listener(target - MATCH[agent_id].value)
            MATCH[agent_id].reward = []
            MATCH[agent_id].value = 0
        if verbose:
            print("\nAfter Listener Update: ")
            print(MATCH[0])
            print(MATCH[1])

    if verbose:
        print("\nDoing this episode's commands now")
    command_recipients = np.zeros(
        (n_agents, n_agents)
    )  # agents command themselves always
    command_contents = np.zeros(
        (n_agents, n_agents)
    )  # later will be 3d for actions with more than 1 number
    for agent_id in range(n_agents):
        # Only choose targets if this is an update round
        if current_step % n_steps == 0:
            if agent_id == 0 and GREED:
                target = MATCH[agent_id].choose_target(
                    np.array([0, 100]), np.ones(n_agents)
                )  # agents are always alive here
            else:
                target = MATCH[agent_id].choose_target(
                    None, np.ones(n_agents)
                )  # agents are always alive here
        else:
            target = MATCH[agent_id].target

        if verbose:
            print(f"aid: {agent_id} chose: {target} to command")

        command_recipients[target, agent_id] = 1
        act, _, lp, __, qv = MATCH[agent_id].agent.train_actions(
            obs[target], la[target]
        )
        command_contents[target, agent_id] = act
        if target != agent_id:  # action commanded to itself if it didnt just
            act, _, lp, __, qv = MATCH[agent_id].agent.train_actions(
                obs[agent_id], la[agent_id]
            )
            command_contents[agent_id, agent_id] = act
    if verbose:
        print(f"Command matrix: {command_recipients}")
    # Record expected values before commands, take actions, and update outgoing MABs
    for agent_id in range(n_agents):
        if current_step % n_steps == 0:  # only set my value on an update step
            MATCH[agent_id].record_value(
                MATCH[agent_id].agent.expected_V(
                    obs=obs[agent_id], legal_action=la[agent_id]
                )
            )
            if agent_id == 0:
                action, commander = MATCH[agent_id].policy_with_oracle(
                    command_recipients[agent_id] * int(not GREED),
                    command_contents[agent_id],
                )
            else:
                action, commander = MATCH[agent_id].policy_with_oracle(
                    command_recipients[agent_id], command_contents[agent_id]
                )
            if verbose:
                print(
                    f"aid: {agent_id} took action from commander: {commander} updating commanders based on who listened"
                )
            # if agent_id != commander:
            # print(f"aid != commander so update commander's listener well")
            if command_recipients[agent_id, commander] > 0:
                if verbose:
                    print(
                        f"updating agent[{commander}]'s speaker [{agent_id}] positively"
                    )
                MATCH[commander].update_oracle(
                    adv=1, sampled=agent_id, verbose=int(verbose)
                )
            for commander_id in range(n_agents):
                if (  # If this commander_id sent a command but it is not the one that got followed
                    command_recipients[agent_id, commander_id] > 0
                    and commander_id != commander
                ):
                    if verbose:
                        print(
                            f"updating agent[{commander_id}]'s speaker [{agent_id}] negatively"
                        )
                    MATCH[commander_id].update_oracle(
                        adv=-1, sampled=agent_id, verbose=int(verbose)
                    )  # if this is someone else's command and we didn't listen -1
        else:  # Keep listening to the same one if this is not an update step
            action = command_contents[agent_id, MATCH[agent_id].leader]
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
    central_mixer=None,
    episodic=False,
    learn_during_rand=False,
    display=False,
    graph_progress=False,
    reward_checkpoint_bin=[10, 40, 60, 70, 80, 100],  #   #
    model_path="",
    MATCH=None,
    n_shot=1,
    n_step=5,
    online=False,
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

    quit_early = False
    if episode_type == Episode_Type.HUMAN:  # RAND, EVAL, HUMAN, EGREEDY
        quit_early = (
            input("Would you like to record a human episode? (y,n)").lower() == "n"
        )

    while overall_step < max_steps and not quit_early:
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
            # print("Resetting match samplers")
            MATCH_Wrappers = [
                Agent_With_Oracle(
                    agent=models[0], n_agents=n_agents, oracle_num=0, sampler="Thompson"
                ),
                Agent_With_Oracle(
                    agent=models[1], n_agents=n_agents, oracle_num=1, sampler="Thompson"
                ),
            ]
        # if current_episode % n_shot == (n_shot - 1):
        # for aid in range(n_agents):
        # MATCH_Wrappers[aid].greed()

        while not done:
            actions, la = None, None
            if MATCH is None:
                actions, la, log_probs = actions_no_match(
                    memory,
                    models,
                    episode_type,
                    epsilon,
                    obs,
                    n_agents,
                    n_actions,
                    env,
                )
            else:
                actions, la, log_probs = actions_match(
                    memory,
                    models,
                    episode_type,
                    epsilon,
                    obs,
                    n_agents,
                    n_actions,
                    env,
                    MATCH_Wrappers,
                    n_steps=n_step,
                    current_step=current_episode_step,
                    device=device,
                )
            # input()
            if display:
                env.display(obs, None, 0)

            obs_, reward, terminated, truncated, info = env.step(actions)

            if MATCH is not None:
                for aid in range(n_agents):
                    MATCH_Wrappers[aid].record_reward(reward)

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

            if episode_type != Episode_Type.EVAL:

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

                if supervised_reg:  # TODO: make sample in order for rnn
                    exp = imitation_memory.sample_transitions(
                        256, as_torch=True, device=device
                    )
                    # print(exp)
                    models[agent_id].imitation_learn(
                        exp.obs[agent_id], exp.discrete_actions[agent_id]
                    )

            if episode_type == Episode_Type.RAND:
                print("yeehaw")
                if learn_during_rand:
                    for agent_id in range(n_agents):
                        exp = memory.sample_episodes(256, as_torch=True, device=device)
                        for e in exp:
                            models[agent_id].reinforcement_learn(
                                e, agent_id, critic_only=True
                            )
            overall_step += 1
            current_episode_step += 1
            # print(obs)
            # print(obs_)
            obs = obs_

        hl = np.zeros(n_agents)
        if episode_type != Episode_Type.EVAL:
            if imitation_memory is not None and imitation_memory.steps_recorded > 100:
                for hi in range(n_agents):
                    if memory.steps_recorded < 256:
                        break
                    f_batches = imitation_memory.sample_episodes(
                        max_batch_size=256, as_torch=True, device=device
                    )
                    hl[hi] = check_discrete_human_likeness(
                        models[agent_id], f_batches, hi
                    )

        # if current_episode % n_shot == (n_shot - 1):
        human_likeness.append(hl)
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

        if episode_type == Episode_Type.RL and len(reward_checkpoint_bin) > 0:
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

        if episode_type == Episode_Type.RL and overall_step >= max_steps:
            models[agent_id].save(
                model_path
                + f"r_{smooth_reward_hist[-1]}_supreg{supervised_reg}_{agent_id}/"
            )

        if episode_type == Episode_Type.HUMAN:
            save = input("Save episode? ") == "y"
            if save:
                print("saved to drive")
                # print(memory)
                FlexibleBuffer.save(memory)
                idx_before = memory.idx
                steps_recorded_before = memory.steps_recorded
                ep_inds = memory.episode_inds
                ep_lens = memory.episode_lens
            else:
                memory.idx = idx_before
                memory.steps_recorded = steps_recorded_before
                memory.episode_inds = ep_inds
                memory.episode_lens = ep_lens
            quit_early = input("Another episode?").lower() == "n"

        if episode_type == Episode_Type.RAND:
            if recent_rewards[-1] > 0.1:
                print("good episode, saving")
                idx_before = memory.idx
                steps_recorded_before = memory.steps_recorded
                ep_inds = memory.episode_inds
                ep_lens = memory.episode_lens
            else:
                print("bad episode, not saving")
                # overall_step = overall_before
                # memory.idx = idx_before
                # memory.steps_recorded = steps_recorded_before
                # memory.episode_inds = ep_inds
                # memory.episode_lens = ep_lens
        # print(
        #    f"ce: {current_episode}, {current_episode % 50 == 0} and {current_episode > 1} and {graph_progress}"
        # )
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

        if MATCH:
            print("Samplers summary: ")
            for aid in range(n_agents):
                MATCH_Wrappers[aid].final_dir(prior=None)
    if (
        episode_type == Episode_Type.RL
        or episode_type == Episode_Type.RAND
        or episode_type == Episode_Type.EGREEDY
        or episode_type == Episode_Type.EVAL
    ):
        return (
            np.array(smooth_reward_hist),
            np.array(smooth_expert_reward_hist),
            np.array(human_likeness),
        )


def offline_update(models: list[Agent], n, memory: FlexibleBuffer, verbose=False):
    print(memory)
    if memory.steps_recorded < 10:
        return [0], [0], [0]
    critic_loss = []
    actor_loss = []
    hlike = []
    for i in range(n):
        batches = memory.sample_episodes(
            max_batch_size=400, as_torch=True, device=device
        )
        b: FlexiBatch
        for b in batches:
            # print(b.terminated)
            for ag in range(n_agents):
                aloss, closs = models[ag].reinforcement_learn(b, ag, True)
                critic_loss.append(closs)
                actor_loss.append(
                    models[ag].imitation_learn(b.obs[ag], b.discrete_actions[ag])
                )
            for id in range(n_agents):
                hlike.append(check_discrete_human_likeness(models[ag], batches, id))

    if True:
        # for h in hlike:
        # h2.append(h.mean())
        critic_loss = np.array(critic_loss)
        crmax = np.max(critic_loss)
        plt.plot(critic_loss / crmax)
        plt.plot(actor_loss)
        plt.plot(hlike)
        plt.legend([f"Critic Loss {crmax}", "Actor Imitation loss", "Human Likeness"])
        plt.show()

    return critic_loss, actor_loss, hlike


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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="Multi-Agent Experiment Runner",
        description="This program trains MARL models \
            on wrapped environments with additional human \
            data. The effects of the human data curation \
            are then recorded and graphed for comparison",
        epilog="Text at the bottom of help",
    )

    parser.add_argument("-d", "--demo", action="store_true")
    parser.add_argument("-hf", "--hum_feedback", action="store_true")
    parser.add_argument(
        "-a", "--algorithm", action="store", choices=["PPO", "DQ", "SDQ", "MDQ", "PG"]
    )
    parser.add_argument(
        "-e",
        "--env",
        action="store",
        choices=["cartpole", "overcooked", "ttt", "ttt_roles", "ttt_lever"],
    )
    parser.add_argument("-r", "--record", action="store_true")
    parser.add_argument("-sr", "--supreg", action="store_true")
    parser.add_argument("-m", "--mix", action="store_true")
    parser.add_argument("-rid", "--runid", action="store", default=0)
    parser.add_argument("-nsh", "--n_shot", action="store", default=0)
    parser.add_argument("-nst", "--n_step", action="store", default=0)
    parser.add_argument("-rls", "--rlsteps", action="store", default=1000)
    parser.add_argument("-rands", "--randsteps", action="store", default=0)
    parser.add_argument("-egr", "--egreedy", action="store", default=0)
    parser.add_argument("-g", "--graph", action="store_true")
    parser.add_argument(
        "-cuda", "--cuda_device", action="store", choices=["cuda:0", "cuda:1", "cpu"]
    )
    parser.add_argument("-eval", "--evaluate_pairwise", action="store_true")

    args = parser.parse_args()

    reward_bin = []
    arg_to_env_str = {
        "cartpole": "Cartpole",
        "overcooked": "Overcooked",
        "ttt": "TTT",
        "ttt_roles": "TTTRoles",
        "ttt_lever": "TTTLever",
    }
    results_path = "./PaperExperiment/"
    if args.env == "sc2":
        # env, n_actions, n_agents = get_sc2_env()
        # obs, states = env.reset()
        results_path += "SC2/memories/"

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

    elif args.env == "matrix":
        # env = MatrixGame()
        n_actions = 2
        n_agents = 2
        # obs, states = env.reset()
        results_path += "Matrix/"

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

    print(env.get_state_feature_names())
    device = torch.device(args.cuda_device if torch.cuda.is_available() else "cpu")

    if args.evaluate_pairwise:
        model_fams = ["PPO", "MDQ"]
        res_paths = []
        dir_lists = []
        for r in model_fams:
            res_paths.append(results_path + f"algo_{r}/model_checkpoints/")
            dir_lists.append(os.listdir(res_paths[-1]))
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
                if float(d[2:6].replace("_", "")) <= 0.8:
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

        if False:
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

        print(model_dirs)
        print(graph_names)
        print(algos)

        input("huh")
        the_agents = get_agent(obs, args, device, n_actions)
        the_agents2 = get_agent(obs, args, device, n_actions)
        scores = np.zeros((len(model_dirs), len(model_dirs)))

        h_mem = FlexibleBuffer(
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
                    episode_type=Episode_Type.EVAL,
                    memory=h_mem,
                    imitation_memory=h_mem,
                    max_steps=(10 if args.env not in ["ttt", "ttt_roles"] else 5) * 200,
                    MATCH=None,
                    n_shot=int(args.n_shot),
                    n_step=int(args.n_step),
                )
                scores[i, j] = np.array(rew).mean()
                print(scores[i])
                print(f"{i*len(model_dirs)+j}/{len(model_dirs)**2}")

        fig, ax = plt.subplots()
        im = ax.imshow(scores)
        np.save(f"score_trial_greed0_{args.n_shot}_{args.n_step}", scores)
        # Set ticks and labels
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

    h_mem = FlexibleBuffer.load(results_path, "test_" + args.env)
    if h_mem is None:
        h_mem = FlexibleBuffer(
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
    print(h_mem)

    online = False
    episodic = False
    # if args.algorithm == "SAC":
    #     model = SAC(
    #         obs_size=obs.shape[1],
    #         action_size=n_actions,
    #         device=device,
    #         directory="./SAC_test/",
    #         mixer=None,
    #         Q_lr=5e-5,
    #         Pi_lr=2e-5,
    #     )
    #     model.save("bruh")
    if args.algorithm == "DQ":
        model = DQN(
            obs_dim=obs.shape[1],
            continuous_action_dims=0,  # continuous_env.action_space.shape[0],
            max_actions=None,  # continuous_env.action_space.high,
            min_actions=None,  # continuous_env.action_space.low,
            discrete_action_dims=[env.action_space.n],
            hidden_dims=[64, 64],
            device="cuda:0",
            lr=1e-4,
            activation="relu",
            dueling=True,
            n_c_action_bins=0,
        )

    if args.algorithm == "SDQ":
        model = DQN(
            obs_dim=obs.shape[1],
            continuous_action_dims=0,  # continuous_env.action_space.shape[0],
            max_actions=None,  # continuous_env.action_space.high,
            min_actions=None,  # continuous_env.action_space.low,
            discrete_action_dims=[env.action_space.n],
            hidden_dims=[64, 64],
            device="cuda:0",
            lr=1e-4,
            activation="relu",
            dueling=True,
            n_c_action_bins=0,
            entropy=0.03,
        )
    if args.algorithm == "MDQ":
        model = DQN(
            obs_dim=obs.shape[1],
            continuous_action_dims=0,  # continuous_env.action_space.shape[0],
            max_actions=None,  # continuous_env.action_space.high,
            min_actions=None,  # continuous_env.action_space.low,
            discrete_action_dims=[env.action_space.n],
            hidden_dims=[64, 64],
            device="cuda:0",
            lr=1e-4,
            activation="relu",
            dueling=True,
            n_c_action_bins=0,
            entropy=0.03,
            munchausen=0.9,
        )

    if args.algorithm == "PPO":
        model = PG(
            obs_dim=obs.shape[1],
            discrete_action_dims=[env.action_space.n],
            # continuous_action_dim=continuous_env.action_space.shape[0],
            hidden_dims=np.array([96, 64]),
            # min_actions=continuous_env.action_space.low,
            # max_actions=continuous_env.action_space.high,
            gamma=0.977,
            device="cuda",
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
        )
        # model=PG.load()
        # model.eval_mode = True
        online = True

    if args.algorithm == "PG":
        model = PG(
            obs_dim=obs.shape[1],
            discrete_action_dims=[env.action_space.n],
            hidden_dims=np.array([64, 64]),
            gamma=0.99,
            device="cuda",
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
        )

        online = True

    if args.record:
        # human_buff(env, h_mem, n_agents=n_agents)
        if args.env == "cartpole":
            env = CartpoleWrapped("human")
        run_multi_agent_episodes(
            env=env,
            models=[model, model],
            n_agents=n_agents,
            episode_type=Episode_Type.HUMAN,
            memory=h_mem,
            max_steps=5000,
            supervised_reg=args.supreg,
        )
        if args.env == "cartpole":
            env = CartpoleWrapped()

        sb = input("Save buffer to drive?")
        if sb == "y":
            FlexibleBuffer.save(h_mem)

    if args.demo:
        print("Doing offline update 1")
        off_bell, off_slearn, off_hlikenes = offline_update([model, model], 300, h_mem)
        # plt.plot(off_bell / np.max(off_bell))
        # plt.plot(off_hlikenes)
        # plt.legend([f"off bell {np.max(off_bell)}", "likeness"])
        # plt.show()

    # r_mem = FlexibleBuffer.load(results_path, "test_" + args.env)
    r_mem = None
    if r_mem is None:
        r_mem = FlexibleBuffer(
            num_steps=10000,
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
    # print(r_mem.obs.shape)
    # print(r_mem)
    # exit()
    if int(args.egreedy) > 0:
        print("Sampling random trajectories")
        rew, exprew, hl = run_multi_agent_episodes(
            env=env,
            models=[model, model],
            n_agents=n_agents,
            episode_type=Episode_Type.EGREEDY,
            memory=r_mem,
            max_steps=int(args.egreedy),
            supervised_reg=False,
            display=False,
            graph_progress=args.graph,
            epsilon=0.1,
        )

        if args.env == "matrix":
            print("Done with rand episodes")
            print(
                f"Agent 1: {model.utility_function(np.array([[1,0,0,0]]))} {model.utility_function(np.array([[0,1,0,0]])), model.get_value(np.array([[0,0,1,0]]))}"
            )
            print(
                f"Agent 2: {model.utility_function(np.array([[1,0,0,1]]))} {model.utility_function(np.array([[0,1,0,1]])), model.utility_function(np.array([[0,0,1,1]]))}"
            )

    if int(args.randsteps) > 0:
        print("Sampling random trajectories")
        rew, exprew, hl = run_multi_agent_episodes(
            env=env,
            models=[model, model],
            n_agents=n_agents,
            episode_type=Episode_Type.RAND,
            memory=r_mem,
            max_steps=int(args.randsteps),
            supervised_reg=False,
            display=False,
            graph_progress=args.graph,
        )

        if args.env == "matrix":
            print("Done with rand episodes")
            print(
                f"Agent 1: {model.utility_function(np.array([[1,0,0,0]]))} {model.utility_function(np.array([[0,1,0,0]])), model.get_value(np.array([[0,0,1,0]]))}"
            )
            print(
                f"Agent 2: {model.utility_function(np.array([[1,0,0,1]]))} {model.utility_function(np.array([[0,1,0,1]])), model.utility_function(np.array([[0,0,1,1]]))}"
            )

    if args.demo and int(args.egreedy) > 0:  # args.demo
        print("Doing offline update")
        off_bell, off_slearn, off_hlikenes = offline_update(
            models=[model, model], n=500, memory=r_mem
        )
        # plt.plot(off_bell / np.max(off_bell))
        # plt.plot(off_hlikenes)
        # plt.legend([f"off bell {np.max(off_bell)}", "likeness"])
        # plt.show()

    if args.env == "cartpole":
        dirpath = "./PaperExperiment/Cartpole/"
    elif args.env == "overcooked":
        dirpath = "./PaperExperiment/Overcooked/"
    elif args.env == "ttt":
        dirpath = "./PaperExperiment/TTT/"
    elif args.env == "ttt_roles":
        dirpath = "./PaperExperiment/TTTRoles/"
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
