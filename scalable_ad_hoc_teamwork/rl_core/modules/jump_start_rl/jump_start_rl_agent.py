import numpy as np


class JumpStartRLAgent:

    def __init__(self, guidance_policy, exploration_policy, task_horizon, performance_threshold):
        # constructor for JumpStartRL
        self.guidance_policy = guidance_policy
        self.exploration_policy = exploration_policy
        self.task_horizon = task_horizon
        self.performance_threshold = performance_threshold

        self.current_step = 0
        self.current_evaluation_step = 0
        self.takeover_step = task_horizon
        self.mode = True  # True: Training, False: Testing

    def select_action(self, state, other_state=None):
        if (self.current_step < self.takeover_step and self.mode) or \
                (self.current_evaluation_step < self.takeover_step and not self.mode):
            action = self.guidance_policy.select_action(state, other_state)
        else:
            action = self.exploration_policy.select_action(state, other_state)
        if self.mode:
            self.current_step += 1
        else:
            self.current_evaluation_step += 1
        return action

    def reset_counter(self):
        self.current_step = 0

    def reset_evaluation_counter(self):
        self.current_evaluation_step = 0

    def switch_mode(self):
        self.mode = not self.mode

    def update_evaluation(self, stats):
        mean_reward = np.mean(stats)
        if mean_reward > self.performance_threshold:
            self.takeover_step -= 1

    def store_transition(self, transition):
        self.exploration_policy.store_transition(transition)


