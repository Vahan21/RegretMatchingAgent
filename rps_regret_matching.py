import numpy as np

"""
A self playing agent that uses regret matching to learn the optimal policy for rock-paper-scissors game.
No matter what the initial policy is it eventually converges to [0.33, 0.33, 0.33] if enough iterations are made.
"""


class RPSTrainer:
    def __init__(self):
        self.possible_actions = ['rock', 'paper', 'scissors']
        self.n_actions = len(self.possible_actions)
        self.initial_policy = np.array([1, 1, 1]) / np.sum([1, 1, 1])
        self.policy_agent = self.initial_policy
        self.avg_policy = np.zeros(self.n_actions)
        self.regret_sums_agent = np.zeros(self.n_actions)

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
            regrets.append(possible_reward - reward)
        regrets = np.array(regrets)
        self.regret_sums_agent += regrets

    def update_policy(self):
        normalizing_sum = np.sum(np.maximum(self.regret_sums_agent, 0))
        if normalizing_sum > 0:
            self.policy_agent = np.maximum(self.regret_sums_agent, 0) / normalizing_sum
        else:
            self.policy_agent = self.initial_policy
        self.avg_policy += self.policy_agent

    def play_train(self, n_iterations):
        for i in range(n_iterations):
            action = self.get_action()
            opponents_action = self.get_action()
            reward = self.get_reward(action, opponents_action)
            self.derive_regrets(opponents_action, reward)
            self.update_policy()

        print(f'rounded normalized average policy {np.round(self.avg_policy / np.sum(self.avg_policy), 2)}')


RPSTrainer().play_train(1000)
