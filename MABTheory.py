import numpy as np
from abc import ABC, abstractmethod
from scipy.stats import invgamma, norm
import matplotlib.pyplot as plt


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


class Thompson_Beta_CombiSemi(MAB_Sampler):
    def __init__(
        self,
        n_agents,
        decay_factor=0.95,
        prior_strength=1,
        experience_strength=1,
        id=0,
        single=True,
    ):
        self.id = id
        self.exp_strength = experience_strength
        self.prior_strength = prior_strength
        self.single = single
        self.n_agents = n_agents
        self.decay_factor = decay_factor
        self.alpha = np.ones(self.n_agents)
        self.beta = np.ones(self.n_agents)
        self.avg_num_arms_pulled = n_agents / 2  # Running estimate
        self.arm_pul_probs = np.ones(self.n_agents)

    def choose(self, available_teammates, prior=None):
        """Samples success probabilities and selects teammates to suggest to."""
        # Needs to be bool TODO
        available_teammates = np.array(available_teammates, dtype=bool)
        sampled_probs = np.random.beta(self.alpha, self.beta)
        # print("before avail: ", sampled_probs)
        # print(available_teammates)
        sampled_probs[np.logical_not(available_teammates)] = 0

        if prior is not None:
            sampled_probs = (
                self.exp_strength * sampled_probs + prior * self.prior_strength
            )
            sampled_probs = sampled_probs / (self.exp_strength + self.prior_strength)
        # print("after avail: ", sampled_probs)
        # input()
        # print("alpha beta: ", self.alpha, self.beta)
        if self.single:
            # print(f"sampled probs: {sampled_probs}")
            ss = sampled_probs == np.max(sampled_probs)
            # print(ss.astype(int))
            return ss.astype(int)
        else:
            selected = (sampled_probs > 0.5).astype(int)
            if selected[self.id] == 1:
                selected = selected * 0
                selected[self.id] = 1
            return selected

    def update(self, arms_pulled, advantage, n_options):
        """Updates Beta distributions based on responses and normalizes no-suggestion arm."""
        # print("arms pulled: ", arms_pulled)
        self.alpha *= self.decay_factor
        self.beta *= self.decay_factor
        # print(f"alpha: {self.alpha}, beta: {self.beta}: single: {self.single}")

        if arms_pulled.sum() == 0:
            return 0

        # print(f"arms pulled: {arms_pulled}, n options: {n_options}")
        # Track how many arms are usually pulled when we choose to talk
        if arms_pulled[self.id] == 0:  # If not the "no suggestion" arm
            self.arm_pul_probs = (self.decay_factor * self.arm_pul_probs) + (
                1 - self.decay_factor
            ) * arms_pulled

            # print(advantage * arms_pulled)
            # Update the Beta distribution parameters
            # TODO normalize by probability of arm being pulled
            self.alpha += (
                arms_pulled * advantage * n_options
            ) * self.exp_strength  # Do we need self arm pull prob
            self.beta += arms_pulled * (1 - advantage)
            self.beta[self.id] += (
                (advantage * arms_pulled).sum() / arms_pulled.sum() * self.exp_strength
            )
            self.alpha[self.id] += (
                ((1 - advantage) * arms_pulled).sum()
                / arms_pulled.sum()
                * self.exp_strength
            )
        # print(f"alpha: {self.alpha}, beta: {self.beta} After")

    def dist_mode(self, prior=None):
        return self.alpha / (self.alpha + self.beta)


class Thompson_Gaussian_Sleepy(MAB_Sampler):
    def __init__(
        self,
        n_agents,
        prior_strength=1,
        experience_strength=1,
        id=0,
        mu_0=0,
        n_0=1,
        alpha_0=1.5,
        beta_0=2.5,
        lambda_=0.99,
        gamma=0.99,
        bg_learning_rate=0.95,
        prob_weight_arms=True,
        min_arm_prob=0.1,
    ):
        self.id = id
        self.num_arms = n_agents
        self.lambda_ = lambda_
        self.gamma = gamma
        self.bg_learning_rate = bg_learning_rate
        self.mu = np.zeros(self.num_arms) + mu_0
        self.ns = np.zeros(self.num_arms) + n_0
        self.betas = np.zeros(self.num_arms) + beta_0
        self.alphas = np.zeros(self.num_arms) + alpha_0
        self.prob_weight_arms = prob_weight_arms

        self.prior_strength = prior_strength
        self.experience_strength = experience_strength
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0

        self.arm_avail_probs = np.ones(self.num_arms)
        self.sqrt_Gamma = np.sqrt(self.gamma)
        self.min_arm_prob = min_arm_prob

    def choose(self, available_teammates, priors=None):
        """Samples means and selects teammates to suggest to."""
        available_teammates = available_teammates.copy().astype(int)
        available_teammates[self.id] = 1
        if self.prob_weight_arms:
            self.arm_avail_probs = (self.bg_learning_rate * self.arm_avail_probs) + (
                1 - self.bg_learning_rate
            ) * available_teammates
            self.arm_avail_probs = np.maximum(self.arm_avail_probs, self.min_arm_prob)
        # print(self.arm_avail_probs, " avail")
        # input()
        sampled_means = np.zeros(self.num_arms)
        for a in range(self.num_arms):
            sigma_sq = invgamma.rvs(a=self.alphas[a], scale=self.betas[a])

            # print(f"alpha: {self.alphas[a]}, beta: {self.betas[a]}")
            # x = np.arange(0, 5, 0.001)
            # R = invgamma.pdf(x, a=self.alphas[a], scale=self.betas[a] + 1)
            sampled_means[a] = norm.rvs(loc=self.mu[a], scale=sigma_sq)
        # print(sampled_means)
        # print(available_teammates)
        scores = sampled_means - (1 - available_teammates) * 1e10

        if priors is not None:
            scores = self.experience_strength * scores + self.prior_strength * priors
        # print(scores)
        arm = np.argmax(scores)
        # print(arm)
        # print(self.arm_avail_probs)
        # input()
        return arm

    def update(self, arms_pulled, advantage):
        # Apply time decay to previously active arms
        # for a in range(self.num_arms):
        #    if a != arm:
        #        self.beta[a] *= self.gamma  # Increase uncertainty for inactive arms
        # print(self.betas, " before")

        # print("alphas betas ", self.alphas, self.betas)
        # print("normal means before", self.mu)
        # print("normal sigmas before", invgamma.mean(a=self.alphas, scale=self.betas))
        # input()
        self.alphas *= self.gamma  # TODO cant shrink past 1, also this feels wrong
        self.betas *= self.gamma

        arms_pulled = np.array(arms_pulled).astype(bool)
        for i in range(self.num_arms):
            if arms_pulled[i] < 0.99:
                self.betas[i] /= self.gamma

        self.alphas = np.maximum(self.alphas, self.alpha_0)
        self.betas = np.maximum(self.betas, self.beta_0)
        # self.betas /= self.gamma  # Increase uncertainty for inactive arms
        # print("what is this nonsense")
        # print(self.lambda_ / self.arm_avail_probs)

        # Update parameters for active arm
        muscale = min(
            (1 - self.lambda_) / self.arm_avail_probs[arms_pulled],
            (1 - self.lambda_) * 10,
        )
        # print(arms_pulled)
        # print(muscale)
        # print(muscale[arms_pulled])
        # print(advantage)

        prev_mu = self.mu[arms_pulled]
        self.mu[arms_pulled] = (1 - muscale) * prev_mu + muscale * advantage
        self.alphas[arms_pulled] += 0.5 / self.arm_avail_probs[arms_pulled]
        if True:
            self.betas[arms_pulled] += (
                0.5 * (advantage - prev_mu) ** 2 / self.arm_avail_probs[arms_pulled]
            )
        else:
            self.betas[arms_pulled] = (1 - muscale) * self.betas[
                arms_pulled
            ] + 0.5 * muscale * (advantage - prev_mu) ** 2
        self.ns[arms_pulled] += 1  # Increment only for pulled arm
        # print("alphas betas after: ", self.alphas, self.betas)
        # input()

    def dist_mode(self, prior=None):
        print(self.mu)


# # Example usage
# num_teammates = 3
# agent = Thompson_Dirichlet_Combinatorial_Semi_Sleepy(num_teammates, decay_factor=0.9)

# for t in range(100):  # Simulate 100 time steps
#     available_teammates = np.random.choice(
#         range(num_teammates),
#         size=np.random.randint(1, num_teammates + 1),
#         replace=False,
#     )
#     chosen_teammate = agent.choose(available_teammates)

#     td_error = np.random.rand()  # Simulated TD error as reward
#     agent.update(chosen_teammate, td_error, available_teammates)

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
    def __init__(
        self,
        n_agents,
        id=0,
        stype="Thompson",
        single=True,
        lambda_=0.99,
        gamma=0.95,
        arm_lr=0.95,
        listen_to_duplicate=True,
    ):
        if stype == "Thompson":
            self.listener = Thompson_Gaussian_Sleepy(
                n_agents=n_agents,
                gamma=gamma,
                lambda_=lambda_,
                prior_strength=1,
                experience_strength=1,
                id=id,
                bg_learning_rate=arm_lr,
            )
            self.speaker = Thompson_Beta_CombiSemi(
                n_agents=n_agents,
                decay_factor=gamma,
                prior_strength=1,
                experience_strength=1,
                id=id,
                single=single,
                n_explore=2,
            )
        else:
            print("not implemented")
        self.type = stype
        self.id = id
        self.priors = np.zeros(n_agents) + 2
        self.n_agents = n_agents
        self.commanded_by = np.zeros(n_agents)
        self.commanded_by[self.id] = 1
        self.listen_to_duplicate = listen_to_duplicate

        # These two vars are memorized during action selection
        # so that they can be used during update.
        self.told_to = np.zeros(n_agents)
        self.n_options = 1

    def policy_with_oracle(self, commanded_by, told_to, priors=None):
        self.told_to = np.copy(told_to)
        self.commanded_by = np.copy(commanded_by)
        self.n_options = np.sum(commanded_by)
        self.commanded_by[self.id] = 1
        leader = self.listener.choose(
            available_teammates=self.commanded_by, priors=priors
        )

        self.leaders = np.zeros(self.n_agents)
        self.leaders[leader] = 1
        if self.listen_to_duplicate:
            self.leaders[told_to == told_to[leader]] = 1
            # TODO for continuous actions, add an epsilon for similar actions
        chosen_action = told_to[leader]
        return chosen_action, leader

    def update_listener(self, adv, verbose=0):  # adv is one number
        self.listener.update(arms_pulled=self.leaders, advantage=adv)

    def choose_target(self, available_teammates, priors=None):
        if priors is None:
            priors = self.priors
        self.targets = self.speaker.choose(
            available_teammates=available_teammates, priors=priors
        )
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


def test_thompsons(n_agents, n_steps, n_trials):

    reward_loc = np.random.rand(n_agents) * 5 - 2
    reward_scale = np.random.rand(n_agents) + 0.1
    listen_prob = np.random.rand(n_agents)
    appear_prob = np.maximum(np.random.rand(n_agents), 0.1)

    print(
        "reward_loc: ",
        reward_loc,
        "\nreward_scale: ",
        reward_scale,
        "\nlisten_prob: ",
        listen_prob,
        "\nappear_prob: ",
        appear_prob,
        "\n",
    )
    speaker_single_arms_chosen = np.zeros((n_agents, n_steps, n_trials))
    speaker_multi_arms_chosen = np.zeros((n_agents, n_steps, n_trials))
    listener_arms_chosen = np.zeros((n_agents, n_steps, n_trials))
    listener_arms_appeared = np.zeros((n_agents, n_steps, n_trials))
    expected_rewards = np.zeros((n_agents, n_steps, n_trials))

    lowest_arm = np.argmin(reward_loc)
    lowest_val = np.min(reward_loc)
    diff = np.max(reward_loc) - lowest_val
    for j in range(n_trials):
        reward_loc[lowest_arm] = lowest_val
        speaker_single = Thompson_Beta_CombiSemi(
            n_agents, id=0, single=True, decay_factor=0.99
        )
        speaker_multi = Thompson_Beta_CombiSemi(
            n_agents, id=0, single=False, decay_factor=0.99
        )
        listener = Thompson_Gaussian_Sleepy(n_agents, id=0, lambda_=0.95, gamma=0.95)
        for i in range(n_steps):
            reward_loc[lowest_arm] = lowest_val + min(i * 2 / n_steps, 1.2) * diff
            available_teammates = np.random.rand(n_agents) < appear_prob
            listener_arms_appeared[:, i, j] = available_teammates
            listener_arms_appeared[0, i, j] = 1  # ID 0 is always available
            expected_rewards[:, i, j] = np.copy(listener.mu)

            speaker_single_arms_chosen[:, i, j] = speaker_single.choose(
                np.ones(n_agents)
            )
            speaker_multi_arms_chosen[:, i, j] = speaker_multi.choose(np.ones(n_agents))
            # print(f"singls arms chosen: {speaker_single_arms_chosen[:, i, j]}")
            # print(f"multi arms chosen: {speaker_multi_arms_chosen[:, i, j]}")
            # input()
            c = listener.choose(available_teammates, priors=None)

            listener_arms_chosen[c, i, j] = 1.0

            reward = np.random.normal(
                loc=reward_loc,
                scale=reward_scale,
            )
            followed = (np.random.rand(n_agents) < listen_prob).astype(float)
            # print(f"reward: {reward}, followed: {followed}")
            listener.update(
                arms_pulled=listener_arms_chosen[:, i, j], advantage=reward[c]
            )
            speaker_single.update(
                arms_pulled=speaker_single_arms_chosen[:, i, j],
                advantage=followed,
                n_options=1,
            )
            speaker_multi.update(
                arms_pulled=speaker_multi_arms_chosen[:, i, j],
                advantage=followed,
                n_options=1,
            )
            # print()
        if j % 10 == 0:
            print(
                f"{j}: listener stdev: {invgamma.mean(a=listener.alphas, scale=listener.betas)} a: {listener.alphas}, b: {listener.betas}"
            )

    l = []
    for i in range(n_agents):
        plt.plot(np.mean(speaker_single_arms_chosen[i, :, :], axis=-1))
        l.append(f"Speaker Single {i}")
    plt.title("Speaker Single")
    plt.legend(l)
    plt.show()

    l = []
    for i in range(n_agents):
        plt.plot(np.mean(speaker_multi_arms_chosen[i, :, :], axis=-1))
        l.append(f"Speaker Multi {i}")
    plt.legend(l)
    plt.title("Speaker Multi")
    plt.show()

    l = []
    chosenpercent = np.sum(listener_arms_chosen, axis=-1) / (
        np.sum(listener_arms_appeared, axis=-1) + 0.0001
    )
    print(chosenpercent)
    for i in range(n_agents):
        plt.plot(
            chosenpercent[i, :],
        )
        l.append(f"Listener {i}")
    plt.legend(l)
    plt.plot(np.mean(listener_arms_chosen, axis=-1))
    plt.title("Listener")
    plt.show()

    l = []
    for i in range(n_agents):
        plt.plot(np.mean(expected_rewards[i, :, :], axis=-1))
        l.append(f"Listener arm {i}")
    plt.legend(l)
    plt.title("Listener expected rewards")
    plt.show()


if __name__ == "__main__":
    # check_baseline(5)
    n_agents = 5

    test_thompsons(n_agents, 500, 500)
    exit()
    matches = [MATCH(n_agents, i) for i in range(n_agents)]
    # gmatches = [GMATCH(n_agents, i) for i in range(n_agents)]
    # imatches = [IMATCH(n_agents, i) for i in range(n_agents)]

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
