from collections import defaultdict


def evaluate(env, policies, num_iterations, end_of_iter_func=None, all_done_cond=True):
    rewards = {f"player_{idx}": 0 for idx, policy in enumerate(policies)}

    stats = {"Last Episode Length": []}
    stats.update({f"Last episode reward {player_string}": [] for player_string in rewards})

    for idx_iterations in range(num_iterations):

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

            for idx, policy in enumerate(policies):
                identity = f"player_{idx}"
                rewards[identity] += reward[identity]

            for idx, policy in enumerate(policies):
                identity = f"player_{idx}"
                policy.info_callback(identity, info)

            state = next_state
        stats["Last Episode Length"].append(episode_length)
        for player_string in rewards:
            stats[f"Last episode reward {player_string}"].append(rewards[player_string])
        if end_of_iter_func is not None:
            end_of_iter_func()
    return stats


def evaluate_interaction(env, policies, num_iterations, interactive_pair, end_of_iter_func=None, info_callback=None,
                         full_reset=True):
    rewards = {f"player_{idx}": 0 for idx, policy in enumerate(policies)}
    stats = {"Last Episode Length": []}
    stats.update({f"Last episode reward {player_string}": [] for player_string in rewards})

    for idx_iterations in range(num_iterations):
        state = env.reset(options={"full_reset": full_reset})

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
            state = next_state

            for idx, policy in enumerate(policies):
                identity = f"player_{idx}"
                rewards[identity] += reward[identity]

            if info_callback is not None:
                for idx, policy in enumerate(policies):
                    identity = f"player_{idx}"
                    info_callback(policy, info[identity])

        stats["Last Episode Length"].append(episode_length)
        for player_string in rewards:
            stats[f"Last episode reward {player_string}"].append(rewards[player_string])
        if end_of_iter_func is not None:
            end_of_iter_func()

    return stats


def cross_evaluation(env, policy_set, num_iterations, interactive_pair, end_of_iter_func=None, info_callback=None):
    accumulated_stats = defaultdict(list)
    for it in range(num_iterations):
        for policy_description in policy_set:
            stats = evaluate_interaction(env, policy_set[policy_description], 1,
                                         interactive_pair, end_of_iter_func, info_callback, full_reset=False)
            for entry in stats:
                accumulated_stats[f"{policy_description}_{entry}"].extend(stats[entry])
        env.reset()
    return accumulated_stats



