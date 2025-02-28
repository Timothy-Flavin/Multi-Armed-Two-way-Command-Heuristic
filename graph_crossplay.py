import numpy as np
import matplotlib.pyplot as plt
import os

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

for use_match in [False, True]:
    for nshot in [25, 50, 100, 200, 500]:
        for n_step in [1, 2, 3, 4, 5]:
            for advantage_type in ["monte", "gae", "None"]:
                for stubborn in [False, True]:
                    for mode in ["mean", "last"]:
                        if os.path.exists(
                            f"match_{use_match}_{mode}_score_{nshot}_{n_step}_{advantage_type}_{stubborn}.npy"
                        ):
                            scores = np.load(
                                f"match_{use_match}_{mode}_score_{nshot}_{n_step}_{advantage_type}_{stubborn}.npy"
                            )
                        else:
                            continue

                        if True:
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
