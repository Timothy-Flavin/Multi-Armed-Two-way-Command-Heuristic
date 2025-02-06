import numpy as np
from MABSamplers import Thompson_Multinomial, UCB_Multinomial


def check_baseline(n_agents, trials=1000):

    tot_arr = np.zeros((n_agents))
    for i in range(trials):
        tot_arr = tot_arr + (np.random.rand(n_agents, n_agents) > 0.5).astype(int).sum(
            axis=-1
        )
    print(tot_arr / trials)


class MATCH:
    def __init__(self, n_agents, id=0, type="Thompson"):
        if type == "Thompson":
            self.listener = Thompson_Multinomial(np.zeros(n_agents), 1, 1, explore_n=2)
            self.speaker = Thompson_Multinomial(np.zeros(n_agents), 1, 1, explore_n=2)
        elif type == "UCB":
            self.listener = UCB_Multinomial(n=n_agents, explore_n=2)
            self.speaker = UCB_Multinomial(n=n_agents, explore_n=2)

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

    def update_listener(self, adv, verbose=0):
        if self.avail_commands[self.leader] > 0:
            self.listener.update(adv, self.leader, verbose)

    def choose_target(self, priors, active):
        if priors is None:
            priors = self.priors
        self.target = self.speaker.choose(priors, active)
        return self.target

    def update_oracle(self, adv, sampled=None, verbose=0):
        if not sampled == self.id:
            self.speaker.update(adv, sampled, verbose)
            self.speaker.update((1 - adv) / self.n_agents / 2, self.id, verbose)


class GMATCH:
    def __init__(n_agents):
        return 0


class IMATCH:
    def __init__(n_agents):
        return 0


if __name__ == "__main__":
    check_baseline(5)
    n_steps = 1000
    for i in range(n_steps):