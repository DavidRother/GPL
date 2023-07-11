

class AbstractRLPolicy:

    def select_action(self, state):
        pass

    def store_transition(self, state, **kwargs):
        pass

    def info_callback(self, agent_id, info):
        pass
