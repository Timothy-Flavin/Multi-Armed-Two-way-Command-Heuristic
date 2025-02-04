import numpy as np
import random
from flexibuddiesrl import Agent
from abc import ABC, abstractmethod


class MAB_Sampler(ABC):

    @abstractmethod
    def choose(self, prior, active):
        return 0

    @abstractmethod
    def update(self, advantage, sampled=0, verbose=0):
        return 0

    @abstractmethod
    def dist_mode(self, prior=None):
        return 0


class Thompson_Multinomial(MAB_Sampler):
    def __init__(self, alpha, prior_strength=1, experience_strength=1, explore_n=2):
        self.advantageCounts = np.zeros(len(alpha))
        self.ns = np.zeros(len(alpha))
        self.alpha = alpha
        self.prior_strength = prior_strength
        self.exp_strength = experience_strength
        self.epsilon = 0  # doesn't do anything here
        self.explore_n = explore_n
        self.last_prior = np.zeros(len(alpha))
        # self.chosen_before = np.zeros(len(alpha))

    def choose(self, prior=None, active=1):
        # print(active)
        # print(self.ns)
        if prior is None:
            prior = np.copy(self.alpha)
        unexplored = np.logical_and(self.ns < self.explore_n, active > 0)
        # print(unexplored)
        # print(np.arange(len(self.ns))[unexplored])
        # input("go>")
        if np.sum(unexplored) > 0:
            self.sampled = np.random.choice(np.arange(len(self.ns))[unexplored])
            return self.sampled

        # print(f"{self.prior_strength*prior+self.exp_strength*self.advantageCounts} =  {self.prior_strength}*{prior}+{self.exp_strength}*{self.advantageCounts}")
        alphas = self.prior_strength * prior + self.exp_strength * self.advantageCounts
        sampledMean = np.random.dirichlet(
            np.maximum(alphas, np.ones(alphas.shape[0]) / 10), 1
        )[0]
        # print(sampledMean)
        self.sampled = np.argmax(sampledMean * active)
        self.active = active
        return self.sampled

    def dist_mode(self, prior=None):
        if prior is None:
            prior = self.alpha
        return self.prior_strength * prior + self.exp_strength * self.advantageCounts

    def update(self, advantage, sampled: int = None, verbose=0):

        if sampled is not None:
            self.sampled = sampled
        self.ns[self.sampled] += 1
        norm = len(self.advantageCounts) - 1
        if verbose > 0:
            print(
                f"Sampled adv: {int(advantage>0)*abs(advantage)}, non sampled adv: {int(advantage<0)*abs(advantage)/norm}"
            )

        self.advantageCounts[self.sampled] += int(advantage > 0) * abs(advantage)
        self.advantageCounts[: self.sampled] = (
            self.advantageCounts[: self.sampled]
            + (advantage < 0) * abs(advantage) / norm
        )
        if self.sampled < self.advantageCounts.shape[0] - 1:
            self.advantageCounts[self.sampled + 1 :] = self.advantageCounts[
                self.sampled + 1 :
            ] + (int(advantage < 0) * abs(advantage) / norm)
        self.advantageCounts *= 0.9
        if verbose > 0:
            print(self.advantageCounts)

    def print_state(self, prior=None):
        print("  State of Thompson multinomial Sampler: ")
        if prior is not None:
            print(f"  Prior: {prior}")
        print(f"  Advantage history: {self.advantageCounts}")
        print(
            f"  prior weight: {self.prior_strength}, sample weight: {self.exp_strength}"
        )
        if prior is not None:
            prior = self.alpha
        print(
            f"  Sampler arm State with prior: {(prior*self.prior_strength + self.advantageCounts*self.exp_strength)} = ({prior}*{self.prior_strength} + {self.advantageCounts}*{self.exp_strength})"
        )

    def __str__(self):
        retstr = "  State of Thompson multinomial Sampler: \n"
        retstr += f"  Advantage history: {self.advantageCounts} \n"
        retstr += f"  prior weight: {self.prior_strength}, sample weight: {self.exp_strength}\n"
        retstr += f"  Sampler arm State with alpha prior: {(self.alpha*self.prior_strength + self.advantageCounts*self.exp_strength)} = ({self.alpha}*{self.prior_strength} + {self.advantageCounts}*{self.exp_strength})"
        return retstr


class EG_Multinomial(MAB_Sampler):
    def __init__(
        self,
        n,
        Epsilon=0.3,
        decay=0.95,
        prior_strength=1,
        experience_strength=1,
        learning_rate=0.1,
        initial_val=0.0,
        explore_n=2,
    ):
        self.advantageCounts = np.zeros(n) + initial_val
        self.ns = np.zeros(n)
        self.n = n
        self.StartEpsilon = Epsilon
        self.epsilon = Epsilon
        self.decay = decay
        self.prior_strength = prior_strength
        self.exp_strength = experience_strength
        self.learning_rate = learning_rate
        self.explore_n = explore_n

    def choose(self, prior=None, active=1, verbose=0):
        if prior is None:
            prior = 0
        if np.sum(self.ns) < self.n * self.explore_n:
            if verbose > 0:
                print(
                    f"Choosing a never chosen teammate: {self.ns} {np.arange(len(self.ns))[self.ns<2]}"
                )
            self.sampled = np.random.choice(
                np.arange(len(self.ns))[self.ns < self.explore_n]
            )
            if verbose > 0:
                print(f"    choice: {self.sampled}, chosen so far: {self.ns}")

        elif random.random() < self.epsilon:
            if verbose > 0:
                print(f"eps {self.n}")
            self.sampled = np.random.randint(0, self.n - 1)
        else:
            if verbose > 0:
                print(
                    f"greedy {self.n}: {(prior*self.prior_strength + self.advantageCounts*self.exp_strength)*active}"
                )
            self.sampled = np.argmax(
                (prior * self.prior_strength + self.advantageCounts * self.exp_strength)
                * active
            )
        self.active = active
        return self.sampled

    def update(self, advantage, sampled: int = None, verbose=0):
        if sampled is not None:
            self.sampled = sampled
        if verbose > 0:
            print(f"Updating with adv: {advantage}")
        self.ns[self.sampled] += 1
        self.epsilon *= self.decay
        # print(self.epsilon)
        norm = len(self.advantageCounts) - 1
        if verbose > 0:
            print(
                f"Sampled adv: {self.advantageCounts}, non sampled adv: {int(advantage<0)*abs(advantage)/norm}, n's: {self.ns}"
            )
        self.advantageCounts[self.sampled] = (
            1 - self.learning_rate
        ) * self.advantageCounts[self.sampled] + self.learning_rate * advantage
        # self.advantageCounts[:self.sampled]= (1-self.learning_rate)*self.advantageCounts[:self.sampled] - self.learning_rate*advantage/norm
        # if self.sampled<self.advantageCounts.shape[0]-1:
        # self.advantageCounts[self.sampled+1:]=(1-self.learning_rate)*self.advantageCounts[self.sampled+1:] - self.learning_rate*advantage/norm

    def print_state(self, prior=None):
        print("  State of EG Multinomial Sampler: ")
        if prior is not None:
            print(f"  Prior: {prior}")
        print(f"  Advantage history: {self.advantageCounts}")
        print(
            f"  prior weight: {self.prior_strength}, sample weight: {self.exp_strength}"
        )
        if prior is not None:
            print(
                f"  Sampler arm State with prior: {(prior*self.prior_strength + self.advantageCounts*self.exp_strength)} = ({prior}*{self.prior_strength} + {self.advantageCounts}*{self.exp_strength})"
            )
        else:
            print(
                f"  Sampler arm State: {(self.advantageCounts*self.exp_strength)} ({self.advantageCounts}*{self.exp_strength})"
            )

    def dist_mode(self, prior=None):
        if prior is not None:
            print(
                f"  Sampler arm State with prior: {(prior*self.prior_strength + self.advantageCounts*self.exp_strength)} = ({prior}*{self.prior_strength} + {self.advantageCounts}*{self.exp_strength})"
            )
        else:
            return self.advantageCounts * self.exp_strength
            print(
                f"  Sampler arm State: {(self.advantageCounts*self.exp_strength)} ({self.advantageCounts}*{self.exp_strength})"
            )


class UCB_Multinomial(MAB_Sampler):
    def __init__(
        self,
        n,
        c=1.0,
        learning_rate=0.1,
        prior_strength=1,
        experience_strength=1,
        initial_val=0.0,
        explore_n=2,
    ):
        self.advantageCounts = np.zeros(n) + initial_val
        self.ns = np.zeros(n)
        self.n = n
        self.prior_strength = prior_strength
        self.exp_strength = experience_strength
        self.c = c
        self.learning_rate = learning_rate
        self.t = 1
        self.explore_n = explore_n
        # self.epsilon=1 # does nothgin

    def choose(self, prior=None, active=1, verbose=0):
        if prior is None:
            prior = 0
        if np.sum(self.ns) < self.n * self.explore_n:
            if verbose > 0:
                print(
                    f"Choosing a never chosen teammate: {self.ns} {np.arange(len(self.ns))[self.ns<2]}"
                )
            self.sampled = np.random.choice(
                np.arange(len(self.ns))[self.ns < self.explore_n]
            )
            if verbose > 0:
                print(f"    choice: {self.sampled}, chosen so far: {self.ns}")
        else:
            if verbose > 0:
                print(
                    f"greedy {self.n}: {(prior*self.prior_strength + self.advantageCounts*self.exp_strength)*active} + {self.c*np.sqrt(np.log(self.t)/self.ns)}"
                )
            self.sampled = np.argmax(
                (prior * self.prior_strength + self.advantageCounts * self.exp_strength)
                * active
                + self.c * np.sqrt(np.log(self.t) / self.ns)
            )
        # print((prior * self.prior_strength + self.advantageCounts * self.exp_strength))
        # print(self.c * np.sqrt(np.log(self.t) / self.ns))
        # print(f"t: {self.t}, ns: {self.ns}")
        # input()
        self.active = active
        return self.sampled

    def update(self, advantage, sampled: int = None, verbose=0):
        # print(advantage)
        if sampled is not None:
            self.sampled = sampled
        self.t += 1
        if verbose > 0:
            print(f"Updating with adv: {advantage}")
        self.ns[self.sampled] += 1
        # self.epsilon*=self.decay
        norm = len(self.advantageCounts) - 1
        if verbose > 0:
            print(
                f"Sampled adv: {self.advantageCounts}, non sampled adv: {int(advantage<0)*abs(advantage)/norm}, n's: {self.ns}"
            )
        self.advantageCounts[self.sampled] = (
            1 - self.learning_rate
        ) * self.advantageCounts[self.sampled] + self.learning_rate * advantage

    def print_state(self, prior=None):
        print("  State of EG Multinomial Sampler: ")
        if prior is not None:
            print(f"  Prior: {prior}")
        print(f"  Advantage history: {self.advantageCounts}")
        print(
            f"  prior weight: {self.prior_strength}, sample weight: {self.exp_strength}"
        )
        if prior is not None:
            print(
                f"  Sampler arm State with prior: {(prior*self.prior_strength + self.advantageCounts*self.exp_strength)} = ({prior}*{self.prior_strength} + {self.advantageCounts}*{self.exp_strength})"
            )
        else:
            print(
                f"  Sampler arm State: {(self.advantageCounts*self.exp_strength)} ({self.advantageCounts}*{self.exp_strength})"
            )

    def dist_mode(self, prior=None):
        if prior is not None:
            print(
                f"  Sampler arm State with prior: {(prior*self.prior_strength + self.advantageCounts*self.exp_strength)} = ({prior}*{self.prior_strength} + {self.advantageCounts}*{self.exp_strength})"
            )
        else:
            return self.advantageCounts * self.exp_strength
            print(
                f"  Sampler arm State: {(self.advantageCounts*self.exp_strength)} ({self.advantageCounts}*{self.exp_strength})"
            )


class PEEF_Multinomial:
    def __init__(
        self,
        n,
        Epsilon=0.3,
        decay=0.95,
        prior_strength=1,
        experience_strength=1,
        learning_rate=0.1,
    ):
        self.advantageCounts = np.zeros(len(n))
        self.ns = np.ones(len(n))
        self.n = n
        self.StartEpsilon = Epsilon
        self.epsilon = Epsilon
        self.decay = decay
        self.prior_strength = prior_strength
        self.exp_strength = experience_strength
        self.learning_rate = learning_rate

    def choose(self, prior, active):
        if random.random() < self.epsilon:
            self.sampled = np.random.randint(0, self.n - 1)
        else:
            self.sampled = np.argmax(
                (prior + self.prior_strength + self.advantageCounts * self.exp_strength)
                * active
            )
        self.active = active
        return self.sampled

    def update(self, advantage, verbose=0):
        self.epsilon *= self.decay
        norm = len(self.advantageCounts) - 1
        if verbose > 0:
            print(
                f"Sampled adv: {self.advantageCounts}, non sampled adv: {int(advantage<0)*abs(advantage)/norm}"
            )
        self.advantageCounts[self.sampled] = (
            1 - self.learning_rate
        ) * self.advantageCounts[self.sampled] + self.learning_rate * advantage
        # self.advantageCounts[:self.sampled]= (1-self.learning_rate)*self.advantageCounts[:self.sampled] - self.learning_rate*advantage/norm
        # if self.sampled<self.advantageCounts.shape[0]-1:
        # self.advantageCounts[self.sampled+1:]=(1-self.learning_rate)*self.advantageCounts[self.sampled+1:] - self.learning_rate*advantage/norm


class Agent_With_Oracle:
    def __init__(
        self, agent: Agent, n_agents, oracle_num, sampler="Thompson", gamma=0.99
    ):
        self.priors = np.ones(n_agents) / n_agents
        self.sampler_name = sampler

        if sampler == "Thompson":
            self.listener = Thompson_Multinomial(self.priors, 2.0, 0.1)
        elif sampler == "UCB":
            self.listener = UCB_Multinomial(n_agents, 0.5, 0.1, 1, 0.1, 0)
        else:
            self.listener = EG_Multinomial(
                n_agents, 0.5, decay=0.95, prior_strength=1, experience_strength=0.2
            )

        if sampler == "Thompson":
            self.oracle = Thompson_Multinomial(self.priors, 2.0, 0.1)
        elif sampler == "UCB":
            self.oracle = UCB_Multinomial(n_agents, 0.5, 0.1, 1, 0.1, 0)
        else:
            self.oracle = EG_Multinomial(
                n_agents, 0.5, decay=0.95, prior_strength=1, experience_strength=0.2
            )

        self.agent = agent
        self.gamma = gamma
        self.oracle_num = oracle_num
        self.target = self.oracle_num
        self.value = 0
        self.reward = []
        self.steps = 0

    def record_value(self, val):
        self.value = val

    def record_reward(self, reward):
        self.reward.append(reward)

    def policy_without_oracle(self, state, legal_actions):
        return (
            self.agent.train_actions(state, legal_actions, step=True),
            self.oracle_num,
        )

    def policy_with_oracle(self, active, actions):
        commander_choices = np.copy(active)
        self.avail_commands = np.copy(active)
        commander_choices[self.oracle_num] = 1
        leader = self.listener.choose(0.5, active=commander_choices)
        self.leader = leader
        chosen_action = actions[leader]
        return chosen_action, leader

    def update_listener(self, adv, verbose=0):
        if self.avail_commands[self.leader] > 0:
            self.listener.update(adv, self.leader, verbose)

    def choose_target(self, priors, active):
        if priors is None:
            priors = self.priors
        self.target = self.oracle.choose(priors, active)
        return self.target

    def update_oracle(self, adv, sampled=None, verbose=0):
        self.oracle.update(adv, sampled, verbose)

    def greed(self):
        self.listener.advantageCounts = np.power(self.listener.advantageCounts, 3)
        self.oracle.advantageCounts = np.power(self.oracle.advantageCounts, 3)

    def final_dir(self, prior=None):
        print(
            f"oracle [{self.oracle_num}] mode   : [{np.argmax(self.oracle.dist_mode(prior=prior))}] {self.oracle.dist_mode(prior=prior)}"
        )
        print(
            f"listener [{self.oracle_num}] mode : [{np.argmax(self.listener.dist_mode(prior=prior))}] {self.listener.dist_mode(prior=prior)}"
        )

    def __str__(self):
        restr = f"sampler type: {self.sampler_name}\n"
        restr += f"rewards: {self.reward}, last value: {self.value}\n"
        restr += "\nListener stuff: " + str(self.listener)
        restr += "\nSpeaker stuff: " + str(self.oracle)
        return restr


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def test_samplers(
        loc=np.array([0.2, 0.5, 2, -1]), scale=np.array([0.2, 0.2, 1.0, 0.5]), n=400
    ):
        o = np.argmax(loc)
        observed_returns = [[], [], []]
        choice_correct = [[], [], []]
        choices = [[], [], []]
        for i in range(n):
            advs = np.random.normal(loc=loc, scale=scale)
            sampler: MAB_Sampler
            for s, sampler in enumerate(MABSamplers):
                c = sampler.choose(prior=None, active=1)
                # print(f"sampler {s} chose {c}")
                # print(np.argmax(sampler.dist_mode()))
                choices[s].append(int(np.argmax(sampler.dist_mode()) == o))
                observed_returns[s].append(advs[c])
                choice_correct[s].append(int(o == c))
                sampler.update(advantage=advs[c], sampled=c, verbose=0)
                # print(sampler.advantageCounts)
                # print(
                #    f"c:{c}, r:{observed_returns[s][i]}, correct: {choice_correct[s][i]}"
                # )
            # input()
        # print(choices)
        # print(observed_returns)
        # print(choice_correct)
        # input()
        return np.array(observed_returns), np.array(choice_correct), np.array(choices)

    MABSamplers = [
        Thompson_Multinomial(np.zeros(4) + 5, 1, 1, explore_n=2),
        EG_Multinomial(
            n=4, Epsilon=1.0, decay=0.95, learning_rate=0.1, explore_n=2, initial_val=2
        ),
        UCB_Multinomial(n=4, explore_n=2),
    ]

    n_trials = 1000
    n_steps = 300

    loc = np.array([0.3, 0.4, 1.0, -0.5])
    scale = np.array([1.2, 1.2, 2.0, 1.5])
    observed_returns, choice_correct, modes = test_samplers(
        loc=loc, scale=scale, n=n_steps
    )
    for e in range(n_trials - 1):
        MABSamplers = [
            Thompson_Multinomial(np.zeros(4), 1, 1, explore_n=3),
            EG_Multinomial(
                n=4,
                Epsilon=0.3,
                decay=0.95,
                learning_rate=0.05,
                explore_n=3,
                initial_val=0,
            ),
            UCB_Multinomial(n=4, explore_n=3, initial_val=0, c=0.2, learning_rate=0.05),
        ]
        oret, ccor, mds = test_samplers(loc=loc, scale=scale, n=n_steps)
        observed_returns += oret
        choice_correct += ccor
        modes += mds
    # print(choice_correct)
    observed_returns = observed_returns / n_trials
    choice_correct = choice_correct / n_trials
    modes = modes / n_trials

    episode = np.arange(0, n_steps, 1, dtype=np.int32)
    plt.hlines([np.max(loc)], 0, n_steps)
    for s in range(len(MABSamplers)):
        plt.plot(observed_returns[s])
    plt.legend(["highest", "Thompson", "EG", "UCB"])
    plt.show()

    plt.hlines([1.0], 0, n_steps)
    for s in range(len(MABSamplers)):
        plt.plot(choice_correct[s])
    plt.legend(["highest", "Thompson", "EG", "UCB"])
    plt.ylim(bottom=0)
    plt.show()

    plt.hlines([1.0], 0, n_steps)
    for s in range(len(MABSamplers)):
        plt.plot(modes[s])
    plt.legend(["highest", "Thompson", "EG", "UCB"])
    plt.ylim(bottom=0)
    plt.show()
