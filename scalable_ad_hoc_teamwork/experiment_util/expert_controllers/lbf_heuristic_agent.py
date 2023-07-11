import random
from lbforaging.agents.heuristic_agent import H1, H2, H3, H4, H5


class LBFHeuristicAgent:

    def __init__(self, heuristic_id, player_id=""):
        self.agent = self.init_agent(heuristic_id, player_id)

    def select_action(self, state, *args, **kwargs):
        return self.agent.step(state)

    def store_transition(self, *args, **kwargs):
        pass

    def info_callback(self, agent_id, info):
        pass

    def reset(self):
        pass

    @staticmethod
    def init_agent(heuristic_id, player_id):
        if heuristic_id == "H1":
            return H1(player_id)
        elif heuristic_id == "H2":
            return H2(player_id)
        elif heuristic_id == "H3":
            return H3(player_id)
        elif heuristic_id == "H4":
            return H4(player_id)
        elif heuristic_id == "H5":
            return H5(player_id)
        else:
            raise ValueError("Heuristic ID not recognized")

