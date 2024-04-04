import numpy as np

class QLearningAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = np.zeros((state_size, action_size))

    def select_action(self, state, epsilon):
        if np.random.rand() <= epsilon:
            return np.random.choice(self.action_size)
        else:
            return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state, alpha, gamma):
        old_q_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state])
        new_q_value = (1 - alpha) * old_q_value + alpha * (reward + gamma * next_max)
        self.q_table[state, action] = new_q_value
