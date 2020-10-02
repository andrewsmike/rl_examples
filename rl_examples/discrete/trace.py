def sample_run_trace(agent, environment):
    state = environment.start_state()
    reward = None

    trace = []
    while not environment.state_is_terminal(state):
        action = agent.action(state)
        trace.append(state, action, reward)
        reward = environment.state_reward(state)
        state = environment.sample_state_action_result(state, action)

    return trace

