import matplotlib.pyplot as plt
import numpy as np
import os


def summarize_plot_data(rewards, expert_rewards, human_likeness):
    # Make the arrays not jagget
    nep = 100000000
    for r in rewards:
        if len(r) < nep:
            nep = len(r)
    for i in range(len(rewards)):
        rewards[i] = rewards[i][:nep]
        expert_rewards[i] = expert_rewards[i][:nep]
        human_likeness[i] = human_likeness[i][:nep]
    rewards = np.array(rewards)
    expert_rewards = np.array(expert_rewards)
    human_likeness = np.array(human_likeness)

    # get means and standard deviations
    print(human_likeness)
    smr = np.mean(rewards, axis=0)
    smer = np.mean(expert_rewards, axis=0)
    hlike = np.mean(human_likeness[0], axis=-1)
    smr_sd = np.std(rewards, axis=0)
    smer_sd = np.std(expert_rewards, axis=0)
    hlike_sd = np.std(human_likeness[0], axis=-1)

    return smr, smer, hlike, smr_sd, smer_sd, hlike_sd


def plot_performance(
    rewards, expert_rewards, human_likeness, dirpath, filename, supreg, hf, algo
):
    smr, smer, hlike, smr_sd, smer_sd, hlike_sd = summarize_plot_data(
        rewards, expert_rewards, human_likeness
    )
    x = np.arange(smr.shape[0])
    fig, axes = plt.subplots(nrows=2)

    hlike = hlike.flatten()
    hlike_sd = hlike_sd.flatten()
    axes[0].plot(smr, color="blue")
    axes[0].plot(smer, color="red")
    axes[0].legend(
        [
            "MDP rewards",
            "Expert rewards",
        ]
    )
    axes[0].fill_between(
        x, y1=smer - smer_sd, y2=smer + smer_sd, color=[0.9, 0.2, 0.1, 0.3]
    )
    axes[0].fill_between(
        x, y1=smr - smr_sd, y2=smr + smr_sd, color=[0.1, 0.2, 0.9, 0.3]
    )

    axes[1].plot(hlike, color=[0.1, 0.9, 0.3])
    axes[1].hlines(
        0.267,
        colors=[[0.1, 0.9, 0.3]],
        xmin=0,
        xmax=smr.shape[0] - 1,
        linestyles=["dashed"],
    )
    axes[1].legend(
        [
            "Human Likeness",
            "Random Human Likeness",
        ]
    )
    axes[1].fill_between(
        x, y1=hlike - hlike_sd, y2=hlike + hlike_sd, color=[0.1, 0.9, 0.3, 0.3]
    )
    axes[1].hlines(0, colors=[[0.1, 0.1, 0.1]], xmin=0, xmax=smr.shape[0] - 1)
    axes[1].hlines(1.0, colors=[[0.1, 0.1, 0.1]], xmin=0, xmax=smr.shape[0] - 1)

    axes[0].set_title(
        f"Reward and Human Likeness {algo}: [hf {hum_feedback}], [demo {demo}], [hr {supreg}]"
    )
    axes[0].set_ylabel("Moving Average Reward")
    axes[1].set_ylabel("Probability")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylim(0, 1)
    axes[0].set_ylim(0, 120)
    axes[0].grid()
    axes[1].grid()
    # fig = plt.gcf()
    # fig.show()
    if not os.path.exists(dirpath + "figures/"):
        os.makedirs(dirpath + "figures/")
    fig.savefig(dirpath + "figures/" + filename)


def make_bat():
    for env in ["Cartpole", "SC2"]:
        for algo in ["PPO", "PPO"]:
            for demo in [True, False]:
                for supreg in [True, False]:
                    for hum_feedback in [True, False]:
                        for runid in range(5):
                            continue


if __name__ == "__main__":
    for env in ["TTT", "Overcooked"]:
        for algo in ["DQ", "PPO"]:
            dirpath = f"./PaperExperiment/{env}/" + f"algo_{algo}/"
            for demo in [True, False]:
                for supreg in [True, False]:
                    for hum_feedback in [True, False]:
                        hl = []
                        r = []
                        er = []
                        for runid in range(5):
                            filename = f"demo_{demo}_human_reg{supreg}_hf_{hum_feedback}_run{runid}_"
                            print(dirpath + filename)
                            if os.path.exists(
                                dirpath + filename + "human_likeness.npy"
                            ):
                                hl.append(
                                    np.load(dirpath + filename + "human_likeness.npy")
                                )
                                r.append(np.load(dirpath + filename + "rew.npy"))
                                er.append(np.load(dirpath + filename + "exprew.npy"))
                            else:
                                break
                        if len(hl) == 0:
                            continue
                        plot_performance(
                            r,
                            er,
                            hl,
                            dirpath,
                            filename[:-2],
                            supreg,
                            hum_feedback,
                            algo,
                        )
