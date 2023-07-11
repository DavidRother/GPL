class NoActionAgent:

    @staticmethod
    def select_action(observation):
        return 0

    def store_transition(self, *args, **kwargs):
        pass

    def info_callback(self, agent_id, info):
        pass

    def reset(self):
        pass
