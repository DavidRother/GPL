class RandomAgent:

    def __init__(self, action_space):
        self.action_space = action_space

    def select_action(self, observation, other_agents_states=None):
        return self.action_space.sample()

    def store_transition(self, *args, **kwargs):
        pass

    def info_callback(self, agent_id, info):
        pass

    def reset(self):
        pass
