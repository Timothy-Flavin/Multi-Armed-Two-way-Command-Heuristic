import numpy as np
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


class Thompson_Dirichlet_Sleepy(MAB_Sampler):
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


class Thompson_Beta_Sleepy(MAB_Sampler):
    def __init__(
        self,
        n_agents,
        decay_factor=0.95,
        prior_strength=1,
        experience_strength=1,
        id=0,
        single=True,
        speaker=True,
    ):
        self.speaker = speaker
        self.id = id
        self.single = single
        self.n_agents = n_agents
        self.decay_factor = decay_factor
        self.alpha = np.ones(self.n_agents)
        self.beta = np.ones(self.n_agents)
        self.avg_num_arms_pulled = n_agents / 2  # Running estimate
        self.arm_pul_probs = np.ones(self.n_agents)
        self.arm_pul_count = np.zeros(self.n_agents)

        self.active_probs = np.ones(self.n_agents) / 2

    def choose(self, active):
        """Samples success probabilities and selects teammates to suggest to."""
        print(active)
        self.active_probs = (
            self.decay_factor * self.active_probs + (1 - self.decay_factor) * active
        )
        need_to_explore = active * self.arm_pul_count
        if np.sum(need_to_explore) > 0:
            candidates = np.random.rand(self.n_agents) * (need_to_explore > 0).astype(
                float
            )
            if self.single:
                selected = (candidates == np.argmax(candidates)).astype(int)
                self.arm_pul_count += selected
                return
            else:
                selected = (candidates > 0.5).astype(int)
                if selected(self.id) == 1:
                    selected = selected * 0
                    selected[self.id] = 1
                self.arm_pul_count += selected
                return selected

        sampled_probs = np.random.beta(self.alpha, self.beta)
        print("before avail: ", sampled_probs)
        sampled_probs *= active
        print("after avail: ", sampled_probs)
        if self.single:
            return (sampled_probs == np.max(sampled_probs)).astype(int)
        else:
            selected = (sampled_probs > 0.5).astype(int)
            if selected(self.id) == 1:
                selected = selected * 0
                selected[self.id] = 1
            return selected

    def update(self, arms_pulled, advantage, n_options, debug=False):
        """Updates Beta distributions based on responses and normalizes no-suggestion arm."""
        self.alpha *= self.decay_factor
        self.beta *= self.decay_factor

        # Track how many arms are usually pulled when we choose to talk
        if arms_pulled[self.id] == 0:  # If not the "no suggestion" arm
            # self.arm_pul_probs = (self.decay_factor * self.arm_pul_probs) + (
            #     1 - self.decay_factor
            # ) * arms_pulled
            if debug:
                print(f"Debugging Thomp B Sleepy Update: arms pulled: {arms_pulled}")

            if self.speaker:
                # Update the Beta distribution parameters
                # TODO normalize by probability of arm being pulled
                self.alpha = self.alpha + (
                    arms_pulled * advantage * n_options  # n choices the other guy had
                )  # Do we need self arm pull prob
                self.beta = self.beta + (arms_pulled * (1 - advantage))
                self.alpha[self.id] = (
                    self.alpha[self.id]
                    + (arms_pulled * advantage).sum() / arms_pulled.sum()
                )
                self.beta[self.id] += (
                    arms_pulled * (1 - advantage)
                ).sum() / arms_pulled.sum()

            if not self.speaker:
                # Update the Beta distribution parameters
                alpha_advantage = (
                    arms_pulled * advantage * (advantage > 0).astype(float)
                )
                beta_advantage = (
                    arms_pulled * (-advantage) * (advantage < 0).astype(float)
                )
                self.alpha += (
                    alpha_advantage / self.active_probs
                )  # need to give more weight to the ones that are pulled rarely
                self.beta += beta_advantage / self.active_probs

    def dist_mode(self):
        return self.alpha / (self.alpha + self.beta)

    def __str__(self):
        restr = f"Thompson Beta Sampler: \n"
        restr += f"  alpha: {self.alpha}, beta: {self.beta}\n"
        restr += f"  avg num arms pulled: {self.avg_num_arms_pulled}\n"
        restr += f"  arm pull probs: {self.arm_pul_probs}\n"
        restr += f"  arm pull count: {self.arm_pul_count}\n"
        return restr


# # Example usage
# num_teammates = 3
# agent = Thompson_Dirichlet_Combinatorial_Semi_Sleepy(num_teammates, decay_factor=0.9)

# for t in range(100):  # Simulate 100 time steps
#     active = np.random.choice(
#         range(num_teammates),
#         size=np.random.randint(1, num_teammates + 1),
#         replace=False,
#     )
#     chosen_teammate = agent.choose(active)

#     td_error = np.random.rand()  # Simulated TD error as reward
#     agent.update(chosen_teammate, td_error, active)

#     print(f"Step {t}, Chose Teammate {chosen_teammate}, TD Error: {td_error:.2f}")


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


def check_baseline(n_agents, trials=1000):
    tot_arr = np.zeros((n_agents))
    for i in range(trials):
        tot_arr = tot_arr + (np.random.rand(n_agents, n_agents) > 0.5).astype(int).sum(
            axis=-1
        )
    print(tot_arr / trials)


class MATCH:
    def __init__(self, n_agents, id=0, stype="Thompson", single=True):
        if stype == "Thompson":
            self.listener = Thompson_Beta_Sleepy(
                n_agents=n_agents,
                decay_factor=0.95,
                prior_strength=1,
                experience_strength=1,
                speaker=False,
                id=id,
                single=True,
                n_explore=2,
            )
            self.speaker = Thompson_Beta_Sleepy(
                n_agents=n_agents,
                decay_factor=0.95,
                prior_strength=1,
                experience_strength=1,
                speaker=True,
                id=id,
                single=single,
                n_explore=2,
            )
        elif stype == "UCB":
            self.listener = UCB_Multinomial(n=n_agents, explore_n=2)
            self.speaker = UCB_Multinomial(n=n_agents, explore_n=2)

        self.type = stype
        self.id = id
        self.priors = np.zeros(n_agents) + 2
        self.n_agents = n_agents

    def policy_with_oracle(self, active, actions):
        commander_choices = np.copy(active)
        self.avail_commands = np.copy(active)
        commander_choices[self.id] = 1
        leader = self.listener.choose(0.5, active=commander_choices)
        self.leader = leader
        chosen_action = actions[leader]
        return chosen_action, leader

    def update_listener(self, adv, verbose=0):  # adv is one number
        if self.avail_commands[self.leader] > 0:
            self.listener.update(adv, self.leader, verbose)

    def choose_target(self, priors, active):
        if priors is None:
            priors = self.priors
        self.target = self.speaker.choose(priors, active)
        targets = np.zeros(self.n_agents)
        targets[self.target] = 1
        return targets

    def update_speaker(
        self, adv, sampled=None, verbose=0
    ):  # adv is a vector for all the people we spoke too
        if sampled is None:
            sampled = self.target
        adv = adv[sampled]
        if not sampled == self.id:
            self.speaker.update(adv, sampled, verbose)
            self.speaker.update((1 - adv) / (self.n_agents / 2), self.id, verbose)

    def __str__(self):
        restr = f"sampler type: {self.type}\n"
        # restr += f"rewards: {self.reward}, last value: {self.value}\n"
        restr += "\nListener stuff: " + str(self.listener)
        restr += "\nSpeaker stuff: " + str(self.speaker)
        return restr


class GMATCH:
    def __init__(n_agents):
        return 0


class IMATCH:
    def __init__(n_agents):
        return 0


def symmetric_advantage(n_agents):
    # randomly generate a matrix where the i,j entry is the negative of the j,i entry
    adv = np.random.rand(n_agents, n_agents) * 5
    adv = adv - adv.T
    return adv


def test_match(n_agents):
    adv = symmetric_advantage(n_agents)
    print("advantage: \n", adv)
    n_steps = 1000
    for i in range(n_steps):
        targets = np.zeros((n_agents, n_agents))
        speaker_adv = np.zeros((n_agents, n_agents))
        for a in range(n_agents):
            targets[a] = matches[a].choose_target(None, np.ones(n_agents))

        print(f"targets: \n{targets}")
        for a in range(n_agents):
            options = targets[:, a].flatten()
            _, leader = matches[a].policy_with_oracle(options, np.zeros(n_agents))
            print(f"a: {a}, chosing from: {options}, leader: {leader}")
            speaker_adv[leader, a] = 1 if a != leader else 0

        print(f"speaker_adv: \n{speaker_adv}")

        for speaker in range(n_agents):
            matches[speaker].update_speaker(speaker_adv[speaker])
            print(
                f"updateing speaker: {speaker} with adv: {speaker_adv[speaker]} and targets: {matches[speaker].target}"
            )
        for listener in range(n_agents):
            ad = np.random.normal(adv[listener, matches[listener].leader], scale=0.02)
            print(
                f" listener: {listener} followed leader: {matches[listener].leader} and got adv {ad}"
            )
            matches[listener].update_listener(ad)
        print("matches: ")
        for m in matches:
            print(m)

        input()


def test_sampler(
    n_arms, loc, scale, n_bandits=50, n_steps=500, single=True, debug=True
):
    regret = np.zeros((n_bandits, n_steps))
    arm_freq = np.zeros((n_arms, n_steps))
    for banditn in range(n_bandits):
        sampler = Thompson_Beta_Sleepy(
            n_agents=n_arms,
            decay_factor=0.95,
            prior_strength=1,
            experience_strength=1,
            id=0,
            single=single,
            speaker=True,
        )
        print(f"banditn: {banditn}, loc: {loc}")
        for i in range(n_steps):
            active = np.random.binomial(1, 0.5, n_arms)
            arms = sampler.choose(active=active)

            print(f"arms: {arms} chosen from active: {active}, single: {single}")
            arm_freq[:, i] = arm_freq[:, i] + arms / n_bandits
            if single:
                arm = np.argmax(arms)
                reward = np.random.normal(loc=loc[arm], scale=scale[arm])
                sampler.update(arms, reward, n_options=1)
                best = np.argmax(active * loc)
                print(
                    f"arm: {arm}, E-reward: {loc[arm]}, E-best: {loc[best]}, observed: {reward}"
                )
                regret[banditn, i] = loc[arm] - loc[best]
            # print(f"arm: {arm}, reward: {reward}, sampler: {sampler}")
            else:
                reward = np.random.normal(loc=loc[arms], scale=scale[arms])
                sampler.update(arms, reward.sum(), n_options=1)
                best = (active * loc) * (loc > 0).astype(float)
                print(
                    f"arm: {arms}, E-reward: {loc[arms].sum()}, E-best: {best.sum()}, observed: {reward.sum()}"
                )
                regret[banditn, i] = loc[arms].sum() - best.sum()
            if debug:
                print(sampler)
                input("next itr?")

    return regret, arm_freq


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # check_baseline(5)
    n_agents = 5
    loc = np.random.rand(n_agents)
    scale = np.random.rand(n_agents) / 5

    regret, arm_freq = test_sampler(
        n_arms=n_agents,
        loc=loc,
        scale=scale,
        n_bandits=50,
        n_steps=500,
        debug=True,
    )

    plt.plot(regret.mean(axis=0))
    plt.title("Regret")
    plt.show()

    plt.plot(arm_freq)
    plt.title("Arm Frequency")
    plt.legend([f"arm {i} m: {loc[i]}, stdv: {scale[i]}" for i in range(n_agents)])
    plt.show()

    matches = [MATCH(n_agents, i) for i in range(n_agents)]
    # gmatches = [GMATCH(n_agents, i) for i in range(n_agents)]
    # imatches = [IMATCH(n_agents, i) for i in range(n_agents)]
