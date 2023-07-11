import random


class MetaAgent:

    def __init__(self, agents):
        self._agents = agents
        self.current_agent = agents[0]

    def select_action(self, state, other_agents_states=None):
        return self.current_agent.select_action(state, other_agents_states)

    def randomize_agent(self):
        self.current_agent = random.choice(self._agents)

    def store_transition(self, *args, **kwargs):
        self.current_agent.store_transition(*args)

    def info_callback(self, agent_id, info):
        self.current_agent.info_callback(agent_id, info)

    def reset(self):
        self.current_agent.reset()
        self.randomize_agent()

