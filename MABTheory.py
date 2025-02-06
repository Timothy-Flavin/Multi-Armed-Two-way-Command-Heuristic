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
    def __init__(self, n_agents, id=0, stype="Thompson"):
        if stype == "Thompson":
            self.listener = Thompson_Multinomial(np.zeros(n_agents), 1, 1, explore_n=2)
            self.speaker = Thompson_Multinomial(np.zeros(n_agents), 1, 1, explore_n=2)
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
        adv = adv[self.target]
        if not sampled == self.id:
            self.speaker.update(adv, sampled, verbose)
            self.speaker.update((1 - adv) / self.n_agents / 2, self.id, verbose)

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


if __name__ == "__main__":
    # check_baseline(5)
    n_agents = 5

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
            options[a] = 1
            print(f"a: {a}, chosing from: {options}, leader: {leader}")
            if (
                options.sum() > 1
            ):  # only update if there is more than one person to choose from
                speaker_adv[leader, a] = 1

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
