

def train(env, policies, num_iterations, all_done_cond=True, end_of_iter_func=None):
    rewards = {f"player_{idx}": 0 for idx, policy in enumerate(policies)}

    episode_length = 0

    for idx_iterations in range(num_iterations):

        stats = {"Episode Number": idx_iterations, "Last Episode Length": episode_length}
        stats.update({f"Last episode reward {player_string}": rewards[player_string] for player_string in rewards})

        for policy in policies:
            policy.reset()

        state, info = env.reset()

        for idx, policy in enumerate(policies):
            identity = f"player_{idx}"
            policy.info_callback(identity, info)

        done = {"player_0": False}
        truncation = {"player_0": False}

        rewards = {f"player_{idx}": 0 for idx, policy in enumerate(policies)}

        episode_length = 0

        while (not (all(done.values()) or all(truncation.values())) and all_done_cond) or \
                (not (any(done.values()) or any(truncation.values())) and not all_done_cond):

            episode_length += 1

            actions = {f"player_{idx}": policy.select_action(state[f"player_{idx}"])
                       for idx, policy in enumerate(policies)}
            next_state, reward, done, truncation, info = env.step(actions)

            policy_transitions = {}
            for idx, policy in enumerate(policies):
                identity = f"player_{idx}"
                transition = [state[identity], actions[identity], reward[identity],
                              next_state[identity], done[identity], truncation[identity]]
                policy_transitions[identity] = transition
                rewards[identity] += reward[identity]

            for idx, policy in enumerate(policies):
                identity = f"player_{idx}"
                policy.info_callback(identity, info)

            for idx, policy in enumerate(policies):
                identity = f"player_{idx}"
                other_transitions = [policy_transitions[other_identity] for other_identity in policy_transitions
                                     if other_identity != identity]
                policy.store_transition(policy_transitions[identity], other_transitions=other_transitions)

            state = next_state
            yield stats

        if end_of_iter_func:
            end_of_iter_func()


def train_interaction(env, policies, num_iterations, interactive_pair):
    rewards = {f"player_{idx}": 0 for idx, policy in enumerate(policies)}

    episode_length = 0

    for idx_iterations in range(num_iterations):

        stats = {"Episode Number": idx_iterations, "Last Episode Length": episode_length}
        stats.update({f"Last episode reward {player_string}": rewards[player_string] for player_string in rewards})
        state = env.reset()

        done = {"player_0": False}

        rewards = {f"player_{idx}": 0 for idx, policy in enumerate(policies)}

        episode_length = 0

        while not all(done.values()):

            episode_length += 1
            actions = {}
            for idx, policy in enumerate(policies):
                if idx == interactive_pair[0]:
                    identity = f"player_{interactive_pair[0]}"
                    other_agent_identity = f"player_{interactive_pair[1]}"
                    actions[identity] = policy.select_action(state[identity], [state[other_agent_identity]])
                else:
                    identity = f"player_{idx}"
                    actions[identity] = policy.select_action(state[identity])

            next_state, reward, done, truncation, info = env.step(actions)

            for idx, policy in enumerate(policies):
                if idx == interactive_pair[0]:
                    identity = f"player_{interactive_pair[0]}"
                    other_agent_identity = f"player_{interactive_pair[1]}"
                    transition = [state[identity], actions[identity], reward[other_agent_identity],
                                  next_state[identity], done[other_agent_identity],
                                  state[other_agent_identity], next_state[other_agent_identity]]
                    policy.store_transition(transition)
                rewards[f"player_{idx}"] += reward[f"player_{idx}"]

            state = next_state

            yield stats



