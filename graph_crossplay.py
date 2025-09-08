import numpy as np
import matplotlib.pyplot as plt
import os

LEVEL = "overcooked"
if LEVEL == "TTT":
    graph_names = [
        "PPO R: 0.0 seed: 0",
        "PPO R: 0.4 seed: 0",
        "PPO R: 0.9 seed: 0",
        "PPO R: 0.0 seed: 1",
        "PPO R: 0.4 seed: 1",
        "PPO R: 0.9 seed: 1",
        "MDQ R: 0.0 seed: 0",
        "MDQ R: 0.4 seed: 0",
        "MDQ R: 0.9 seed: 0",
        "MDQ R: 0.0 seed: 1",
        "MDQ R: 0.4 seed: 1",
        "MDQ R: 0.9 seed: 1",
    ]
    root = "./"
elif LEVEL == "overcooked":
    graph_names = [
        "PPO R: 40  seed: 0",
        "PPO R: 80  seed: 0",
        "PPO R: 100 seed: 0",
        "PPO R: 40  seed: 1",
        "PPO R: 80  seed: 1",
        "PPO R: 100 seed: 1",
        "PPO R: 180 seed: 1",
        "MDQ R: 40  seed: 0",
        "MDQ R: 80  seed: 0",
        "MDQ R: 100 seed: 0",
        "MDQ R: 40  seed: 1",
        "MDQ R: 80  seed: 1",
        "MDQ R: 100 seed: 1",
        "MDQ R: 166 seed: 1",
    ]
    root = "./overcookedscores/"


for use_match in [False, True]:
    for nshot in [25, 50, 100, 200, 500] if LEVEL == "TTT" else [1, 2, 3, 4, 5]:
        for n_step in [1, 2, 3, 4, 5] if LEVEL == "TTT" else [5, 10, 15, 20, 25]:
            for advantage_type in ["monte", "gae", "None"]:
                for stubborn in [False, True]:
                    for mode in ["mean", "last"]:
                        if os.path.exists(
                            f"{root}match_{use_match}_{mode}_score_{nshot}_{n_step}_{advantage_type}_{stubborn}.npy"
                        ):
                            scores = np.load(
                                f"{root}match_{use_match}_{mode}_score_{nshot}_{n_step}_{advantage_type}_{stubborn}.npy"
                            )
                        else:
                            continue
                        # print(
                        #    f"{root}match_{use_match}_{mode}_score_{nshot}_{n_step}_{advantage_type}_{stubborn}.npy"
                        # )
                        # print(scores.shape)
                        if False:
                            scores[1] = np.array(
                                [
                                    31.82,
                                    52.72,
                                    67.28,
                                    37.76,
                                    52.76,
                                    79.06,
                                    26.94,
                                    35.4,
                                    28.02,
                                    12.54,
                                    36.54,
                                    50.3,
                                    56.42,
                                    107.7,
                                ]
                            )
                            scores[0] = np.array(
                                [
                                    31.48,
                                    44.32,
                                    35.88,
                                    27.14,
                                    24.26,
                                    40.96,
                                    15.62,
                                    23.52,
                                    31.04,
                                    32.36,
                                    16.22,
                                    10.6,
                                    2.82,
                                    9.06,
                                ]
                            )
                            scores[3] = np.array(
                                [
                                    67,
                                    47.66,
                                    148,
                                    86,
                                    175,
                                    128,
                                    93.3,
                                    60,
                                    65,
                                    150,
                                    41,
                                    70,
                                    152,
                                    125,
                                ]
                            )
                            scores[4] = np.array(
                                [
                                    111,
                                    38,
                                    55.3,
                                    41,
                                    32,
                                    116,
                                    135,
                                    79.6,
                                    102,
                                    12,
                                    59,
                                    60,
                                    28,
                                    77,
                                ]
                            )
                            graph_names[0] = "MDQN PBR"
                            graph_names[1] = "PPO PBR"
                            graph_names[3] = "PPO OM"
                            graph_names[4] = "MDQN OM"
                        if False:
                            scores[0] = np.array(
                                [
                                    -0.21188628,
                                    -0.29584076,
                                    -0.20894605,
                                    -0.75678952,
                                    -0.51931076,
                                    -0.55947318,
                                    -0.35839677,
                                    -0.11089039,
                                    0.06327153,
                                    -0.97824456,
                                    -0.97715295,
                                    -0.44132854,
                                ]
                            )
                            scores[1] = np.array(
                                [
                                    -0.12306626,
                                    0.14277359,
                                    0.01765025,
                                    -1.0,
                                    -1.0,
                                    -1.0,
                                    -0.40717958,
                                    -0.40405242,
                                    -0.12772761,
                                    -1.0,
                                    -1.0,
                                    -1.0,
                                ]
                            )
                            scores[3] = np.array(
                                [
                                    0.01541592,
                                    0.0764255,
                                    0.29199783,
                                    0.06013245,
                                    0.21407224,
                                    0.210127,
                                    0.1432584,
                                    -0.29474402,
                                    -0.60741178,
                                    0.21311884,
                                    0.447923,
                                    0.12573154,
                                ]
                            )
                            scores[4] = np.array(
                                [
                                    -0.35981497,
                                    0.07939706,
                                    0.36643528,
                                    -0.24241677,
                                    -0.00585408,
                                    0.55788737,
                                    0.06444411,
                                    0.38679748,
                                    0.44089022,
                                    -0.0434205,
                                    -0.07143315,
                                    0.55579378,
                                ]
                            )
                            graph_names[0] = "MDQN PBR"
                            graph_names[1] = "PPO PBR"
                            graph_names[3] = "PPO OM"
                            graph_names[4] = "MDQN OM"

                        fig, ax = plt.subplots()
                        im = ax.imshow(scores)
                        ax.set_xticks(np.arange(len(graph_names)))
                        ax.set_yticks(np.arange(len(graph_names)))
                        ax.set_xticklabels(graph_names)
                        ax.set_yticklabels(graph_names)

                        # Rotate the x-tick labels for better readability
                        if LEVEL == "overcooked":
                            im.set_clim(0, 120)
                            fig.colorbar(im, ax=ax)
                        else:
                            im.set_clim(-1, 1)
                            fig.colorbar(im, ax=ax)
                        plt.title(
                            f"{mode} Scores of {'stubborn' if stubborn else ''} match: {use_match}, nshot {nshot}, nstep {n_step} adv {advantage_type}"
                        )
                        plt.setp(
                            ax.get_xticklabels(),
                            rotation=45,
                            ha="right",
                            rotation_mode="anchor",
                        )
                        plt.show()
