import numpy as np
from abc import ABC, abstractmethod
from scipy.stats import invgamma, norm
import matplotlib.pyplot as plt
from flexibuff import FlexibleBuffer
from flexibuddiesrl import Agent


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
        prior_strength=0.1,
        experience_strength=1,
        id=0,
        single=True,
        n_explore=5,
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
        self.arm_pul_count = np.zeros(self.n_agents)
        self.explored = np.ones(self.n_agents) * n_explore

        self.active_probs = np.ones(self.n_agents) / self.n_agents

    def choose(self, available_teammates: np.array, prior=None):
        """Samples success probabilities and selects teammates to suggest to."""
        # Needs to be bool TODO
        self.active_probs = (
            self.decay_factor * self.active_probs
            + (1 - self.decay_factor) * available_teammates
        )

        if (available_teammates * self.explored).sum() > 0:
            options = available_teammates * self.explored > 0
            chosen = np.zeros(self.n_agents)
            if self.single:
                arm = np.random.choice(np.arange(self.n_agents)[options])
                chosen[arm] = 1
            else:
                chosen = (options * np.random.rand(self.n_agents) > 0.5).astype(int)
            self.explored -= chosen
            return chosen

        # print(f"alpha beta: {self.alpha} {self.beta}")
        sampled_probs = np.random.beta(a=self.alpha, b=self.beta)
        sampled_probs[np.logical_not(available_teammates)] = 0
        # print("after avail: ", sampled_probs)

        if prior is not None:
            # print("prior nonsense")
            # print(prior)
            sampled_probs = (
                self.exp_strength * sampled_probs
                + prior / prior.sum() * self.prior_strength
            )
            sampled_probs = sampled_probs / (self.exp_strength + self.prior_strength)
        # print("after avail: ", sampled_probs)
        # input()
        # print("alpha beta: ", self.alpha, self.beta)
        if self.single:
            ss = (sampled_probs == np.max(sampled_probs)).astype(int)
        else:
            ss = (sampled_probs > 0.5).astype(int)
        # print(self.id)
        # print(ss)
        if ss[self.id] == 1:
            ss = np.zeros(self.n_agents)
            ss[self.id] = 1
        self.explored -= ss

        # print(f"ss: {ss}")
        # print(self.dist_mode())
        return ss

    def update(self, arms_pulled, advantage, n_options=1, debug=False):
        """Updates Beta distributions based on responses and normalizes no-suggestion arm."""
        # print("arms pulled: ", arms_pulled)
        self.alpha = self.decay_factor * self.alpha + (1 - self.decay_factor) * 1
        self.beta = self.decay_factor * self.beta + (1 - self.decay_factor) * 1

        # self.alpha = np.max(self.alpha, np.ones(self.n_agents))
        # self.beta = np.max(self.alpha, np.ones(self.n_agents))
        # print(f"alpha: {self.alpha}, beta: {self.beta}: single: {self.single}")
        # print(f"speaker [{self.id}] arms pulled {arms_pulled}")
        if arms_pulled.sum() == 0:  # we spoke to no one including ourself
            return 0
        # print(f"[{self.id}] arms pulled: {arms_pulled}, n options: {n_options}")
        # Track how many arms are usually pulled when we choose to talk
        # print(f"arms pulled [{self.id}]: {arms_pulled}")

        if not arms_pulled[self.id]:  # If not the "no suggestion" arm
            # print("updating")
            # input()
            # self.arm_pul_probs = (self.decay_factor * self.arm_pul_probs) + (
            #     1 - self.decay_factor
            # ) * arms_pulled
            if debug:
                print(f"Debugging Thomp B Sleepy Update: arms pulled: {arms_pulled}")

            # print(advantage * arms_pulled)
            # Update the Beta distribution parameters
            # print(
            #    f"updateing speaker {self.id} with arms pulled {arms_pulled}, adv: {advantage}, noptions: {n_options}"
            # )
            self.alpha += (
                arms_pulled * advantage * n_options
            ) * self.exp_strength  # Do we need self arm pull prob
            self.beta += arms_pulled * (1 - advantage) * self.exp_strength
            self.beta[self.id] += (
                (advantage * arms_pulled * n_options).sum()
                / arms_pulled.sum()
                * self.exp_strength
            )
            self.alpha[self.id] += (
                ((1 - advantage) * arms_pulled).sum()
                / arms_pulled.sum()
                * self.exp_strength
            )
        # print(f"alpha: {self.alpha}, beta: {self.beta} After")

    def dist_mode(self, prior=None):
        return self.alpha / (self.alpha + self.beta)

    def __str__(self):
        mystr = f"Speaker Thompson Sampler id {self.id}: \n"
        mystr += f"\n  alphas: {self.alpha}"
        mystr += f"\n  betas: {self.beta}"
        return mystr


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
        n_explore=5,
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
        self.n_explore = n_explore

        self.prior_strength = prior_strength
        self.experience_strength = experience_strength
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0

        self.arm_avail_probs = np.ones(self.num_arms)
        self.sqrt_Gamma = np.sqrt(self.gamma)
        self.min_arm_prob = min_arm_prob

        self.explored = np.ones(n_agents) * n_explore

    def choose(self, available_teammates, prior=None):
        """Samples means and selects teammates to suggest to."""
        available_teammates = available_teammates.copy().astype(int)

        if (available_teammates * self.explored).sum() > 0:
            options = available_teammates * self.explored > 0
            arm = np.random.choice(np.arange(self.num_arms)[options])
            self.explored[arm] -= 1
            return arm

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

        if prior is not None:
            scores = self.experience_strength * scores + self.prior_strength * prior
        # print(scores)
        arm = np.argmax(scores)
        # print(arm)
        # print(self.arm_avail_probs)
        # input()
        self.explored[arm] -= 1
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
                self.betas[i] /= self.gamma * 1.01

        self.alphas = np.maximum(self.alphas, self.alpha_0)
        self.betas = np.maximum(self.betas, self.beta_0)
        # self.betas /= self.gamma  # Increase uncertainty for inactive arms
        # print("what is this nonsense")
        # print(self.lambda_ / self.arm_avail_probs)

        # Update parameters for active arm
        muscale = min(
            (1 - self.lambda_) / self.arm_avail_probs[arms_pulled],
            (1 - self.lambda_) * 5,
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

    def __str__(self):
        mystr = f"Listener Thompson Sampler id {self.id}: \n"
        mystr += f"  mu: {self.mu}"
        mystr += f"\n  alphas: {self.alphas}"
        mystr += f"\n  betas: {self.betas}"
        return mystr


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
    def __init__(
        self,
        n_agents,
        id=0,
        stype="Thompson",
        single=True,
        lambda_=0.99,
        gamma=0.95,
        arm_lr=0.95,
        speaker_decay=0.99,
        listen_to_duplicate=True,
        n_explore=5,
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
                n_explore=n_explore,
            )
            self.speaker = Thompson_Beta_CombiSemi(
                n_agents=n_agents,
                decay_factor=speaker_decay,
                prior_strength=1,
                experience_strength=1,
                id=id,
                single=single,
                n_explore=n_explore,
            )
        else:
            print("not implemented")
        self.type = stype
        self.id = id
        self.prior = np.zeros(n_agents) + 0.5
        self.n_agents = n_agents
        self.commanded_by = np.zeros(n_agents)
        self.commanded_by[self.id] = 1
        self.listen_to_duplicate = listen_to_duplicate

        # These two vars are memorized during action selection
        # so that they can be used during update.
        self.told_to = np.zeros(n_agents)
        self.n_options = 1
        self.selected = 0

    def policy_with_oracle(self, commanded_by, told_to, prior=None):
        self.told_to = np.copy(told_to)
        self.commanded_by = np.copy(commanded_by)
        # self.n_options = np.sum(commanded_by)
        self.commanded_by[self.id] = 1
        leader = self.listener.choose(
            available_teammates=self.commanded_by, prior=prior
        )

        self.leaders = np.zeros(self.n_agents)
        self.leaders[leader] = 1
        self.selected = leader
        if self.listen_to_duplicate:
            self.leaders[told_to == told_to[leader]] = 1
            # TODO for continuous actions, add an epsilon for similar actions
        chosen_action = told_to[leader]
        return chosen_action, leader

    def update_listener(self, adv, verbose=0):  # adv is one number
        self.listener.update(arms_pulled=self.leaders, advantage=adv)

    def choose_target(self, available_teammates, prior=None):
        if prior is None:
            prior = self.prior
        self.targets = self.speaker.choose(
            available_teammates=available_teammates, prior=prior
        )
        return self.targets

    def update_speaker(
        self, adv, sampled=None, n_options=1, verbose=0
    ):  # adv is a vector for all the people we spoke too
        if sampled is None:
            sampled = self.targets.astype(bool)
        # adv = adv[sampled]
        # print(f"id[{self.id}]: {sampled}")
        self.speaker.update(arms_pulled=sampled, advantage=adv, n_options=n_options)
        # self.speaker.update((1 - adv) / (self.n_agents / 2), self.id, verbose)

    def __str__(self):
        restr = f"sampler type: {self.type}\n"
        # restr += f"rewards: {self.reward}, last value: {self.value}\n"
        restr += "\nListener stuff: " + str(self.listener)
        restr += "\nSpeaker stuff: " + str(self.speaker)
        return restr

    def calc_reward(
        self,
        buffer: FlexibleBuffer,
        agent: Agent,
        idx=np.arange(start=1, stop=10, step=1),
        adv_type="gae",
        r_key="global_rewards",
        s_key="obs",
        sp_key="obs_",
        t_key="terminated",
        v_key=None,
        legal_actions=None,
        gamma=0.99,
        gae_lambda=0.95,
        k_step=5,
        device="cuda",
    ):
        # rewards = buffer.__dict__[r_key][index_start:index_end]
        # if v_key is None:
        #     values = agent.expected_V(buffer.__dict__[s_key][self.id, :, :])
        # else:
        #     values = buffer.__dict__[values][self.id, index_start:index_end]
        # terminated = buffer.__dict__[t_key]
        samp = buffer.sample_transitions(idx=idx, as_torch=True, device=device)

        # TODO validate this code with an agent
        l_ac = None
        l_ac_ = None
        if legal_actions:
            l_ac = []
            l_ac_ = []
            for j in range(len(buffer.action_mask_)):
                l_ac.append(samp.action_mask[j][self.id])
                l_ac_.append(samp.action_mask_[j][self.id, idx[-1], :])

        if v_key is not None:
            values = samp.__dict__[v_key][self.id]
        else:
            values = agent.expected_V(
                samp.__dict__[s_key][self.id],
                legal_action=l_ac,
            )
        last_value = agent.expected_V(
            samp.__dict__[sp_key][self.id, -1],
            legal_action=l_ac_,
        )
        adv = 0
        if adv_type == "gae":
            adv = FlexibleBuffer.GAE(
                rewards=samp.__dict__[r_key],
                values=values,
                terminated=samp.terminated,
                last_value=last_value,
                gae_lambda=gae_lambda,
                gamma=gamma,
            )  # TODO add if statement for individual rewards
        elif adv_type == "td":
            adv = FlexibleBuffer.K_Step_TD(
                rewards=samp.__dict__[r_key],
                values=values,
                terminated=samp.terminated,
                last_value=last_value,
                gamma=gamma,
                k=k_step,
            )
        elif adv_type == "monte":
            adv = FlexibleBuffer.G(
                rewards=samp.__dict__[r_key],
                terminated=samp.terminated,
                last_value=last_value,
                gamma=gamma,
            )
            adv = adv - values
            print("adv_type not implemented")
        elif (
            adv_type == "ep_avg"
        ):  # Take td for each episode with the first and last value and avg them
            print("ep_avg not implemented")
        elif (
            adv_type == "truncate"
        ):  # if a terminal state is reached, ignore the rest of the episodes
            print("truncate not implemented")
        else:
            print("adv_type not recognized")

        return adv.sum().detach().cpu().numpy()


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


def skill_advantage(n_agents):
    askills = np.random.rand(n_agents) * 5
    adv = np.zeros((n_agents, n_agents))
    for i in range(n_agents):
        adv[i] = askills - askills[i]
    print(f"skills: {askills}, advantage: \n{adv}")
    return adv, askills


def test_thompsons(
    n_agents,
    n_steps,
    n_trials,
):

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
            c = listener.choose(available_teammates, prior=None)

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


def test_match(n_agents, n_steps, n_trials, adv, verbose=False, skillvec=np.ones(5)):
    perf = np.zeros(shape=(n_steps, n_trials))
    mus = np.zeros(shape=(n_steps, n_agents, n_agents))
    speakermus = np.zeros(shape=(n_steps, n_agents, n_agents))
    for trial in range(n_trials):
        matches = [
            MATCH(n_agents, i, single=False, lambda_=0.90, gamma=0.90)
            for i in range(n_agents)
        ]
        for i in range(n_steps):
            targets = np.zeros((n_agents, n_agents))
            speaker_adv = np.zeros((n_agents, n_agents))
            for a in range(n_agents):
                mtc = matches[a]
                mtc: MATCH
                targets[a] = mtc.choose_target(
                    prior=None, available_teammates=np.ones(n_agents)
                )
            if verbose:
                print(f"targets: \n{targets}")
            for a in range(n_agents):
                mtc = matches[a]
                mtc: MATCH
                options = targets[:, a].flatten().copy()
                _, leader = mtc.policy_with_oracle(
                    commanded_by=options,
                    prior=np.zeros(n_agents),
                    told_to=np.arange(n_agents),
                )
                if verbose:
                    print(f"a: {a}, chosing from: {options}, leader: {leader}")
                # TODO update all speakers for which the action matches
                speaker_adv[leader, a] = 1 if a != leader else 0
                perf[i, trial] += skillvec[leader]
            if verbose:
                print(f"speaker_adv: \n{speaker_adv}")
            augmented_targets = np.copy(targets)
            for ag in range(n_agents):
                augmented_targets[ag, ag] = 1
            for speaker in range(n_agents):
                mtc = matches[speaker]
                mtc: MATCH
                matches[speaker].update_speaker(
                    adv=speaker_adv[speaker],
                    n_options=augmented_targets.sum(axis=0).flatten() - 1,
                )
                if verbose:
                    print(
                        f"updateing speaker: {speaker} with adv: {speaker_adv[speaker]} and targets: {matches[speaker].targets} with n_options {augmented_targets.sum(axis=0).flatten()}"
                    )
            for listener in range(n_agents):
                mtc = matches[listener]
                mtc: MATCH
                l = mtc.selected
                ad = np.random.normal(loc=adv[listener, l], scale=0.5)
                if verbose:
                    print(
                        f" listener: {listener} followed leader: {l} from {mtc.leaders} and got adv {ad}"
                    )
                matches[listener].update_listener(adv=ad)

            for ag in range(n_agents):
                mus[i, ag] += matches[ag].listener.mu.copy()
                speakermus[i, ag] += matches[ag].speaker.dist_mode()
                # print(matches[ag].listener.mu.copy())
                # print(mus[i])
            if verbose:
                print("matches: ")
                for m in matches:
                    print(m)
                input()
        # print(matches[0].listener)
        # print(matches[1].listener)
        # print(matches[2].listener)
    mus = mus / n_trials
    speakermus = speakermus / n_trials
    avg_perf = perf.mean(axis=-1)
    plt.plot(avg_perf)
    plt.show()

    for agent in range(n_agents):
        leg = []
        for arm in range(n_agents):
            plt.plot(mus[:, agent, arm])
            leg.append(f"arm: {arm}")
        plt.legend(leg)
        plt.title(f"Agent {agent} mus over time")
        plt.show()

    for agent in range(n_agents):
        leg = []
        for arm in range(n_agents):
            plt.plot(speakermus[:, agent, arm])
            leg.append(f"arm: {arm}")
        plt.legend(leg)
        plt.title(f"Agent {agent} speaker probs over time")
        plt.show()


if __name__ == "__main__":
    # check_baseline(5)
    n_agents = 5
    n_steps = 500
    n_trials = 100
    # test_thompsons(n_agents, 500, 500)
    # exit()

    # gmatches = [GMATCH(n_agents, i) for i in range(n_agents)]
    # imatches = [IMATCH(n_agents, i) for i in range(n_agents)]

    adv, skillvec = skill_advantage(n_agents)
    test_match(
        n_agents=n_agents,
        n_steps=n_steps,
        n_trials=n_trials,
        adv=adv,
        verbose=False,
        skillvec=skillvec,
    )

# [[ 0.          1.98644892  2.82382293  1.67797208  3.96014136]
#  [-1.98644892  0.          0.83737401 -0.30847683  1.97369245]
#  [-2.82382293 -0.83737401  0.         -1.14585085  1.13631843]
#  [-1.67797208  0.30847683  1.14585085  0.          2.28216928]
#  [-3.96014136 -1.97369245 -1.13631843 -2.28216928  0.        ]]
