import numpy as np

"""
A self playing agent that uses regret matching to learn the optimal policy for rock-paper-scissors game.
No matter what the initial policy is it eventually converges to [0.33, 0.33, 0.33] if enough iterations are made.
"""


class RPSTrainer:
    def __init__(self):
        self.possible_actions = ['rock', 'paper', 'scissors']
        self.initial_policy = np.array([5, 2, 100]) / np.sum([5, 2, 100])
        self.policy_agent = self.initial_policy
        self.regret_sums_agent = np.array([0, 0, 0])

    @staticmethod
    def get_reward(action, opponents_action):
        if action == opponents_action:
            return 0
        if (action == 'rock' and opponents_action == 'paper') or \
                (action == 'paper' and opponents_action == 'scissors') or \
                (action == 'scissors' and opponents_action == 'rock'):
            return -1
        return 1

    def get_action(self):
        return np.random.choice(a=self.possible_actions, p=self.policy_agent)

    def derive_regrets(self, opponents_action, reward):
        regrets = []
        for action in self.possible_actions:
            possible_reward = self.get_reward(action, opponents_action)
            regrets.append(np.maximum(possible_reward - reward, 0))

        regrets = np.array(regrets)
        self.regret_sums_agent = np.maximum(self.regret_sums_agent + regrets, 0)

    def update_policy(self):
        normalizing_sum = np.sum(self.regret_sums_agent)
        if normalizing_sum > 0:
            self.policy_agent = self.regret_sums_agent / normalizing_sum
            return

        self.policy_agent = self.initial_policy

    def play_train(self):
        for i in range(1000):
            action = self.get_action()
            opponents_action = self.get_action()
            reward = self.get_reward(action, opponents_action)
            self.derive_regrets(opponents_action, reward)
            self.update_policy()

        print(f'rounded policy {np.round(self.policy_agent, 2)}')
