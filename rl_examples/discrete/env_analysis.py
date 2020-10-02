from itertools import count
from pprint import pprint
from random import choice

from environment import GridWorldEnvironment


def value_iteration_returns(problem, epsilon):
    """
    VALUE-ITERATION() from Russel & Norvig.
    Given a MDP, approximate the optimal-policy utility function for each state.
    Relative error from state utility to following state returns is < epsilon.
    ... What even is relative error? Meh.
    """
    next_returns = {state: 0 for state in problem.states()}

    for round_count in count(start=1):
        returns = next_returns

        next_returns = {
            state: problem.state_reward(state)
            + problem.discount() * max((
                sum(
                    problem.state_action_result_dist(
                        state, action
                    ).get(next_state, 0) * returns[next_state]
                    for next_state in problem.states()
                )
                for action in problem.state_actions(state)
            ), default=0)
            for state in problem.states()
        }
        max_update_size = max(abs(next_returns[state] - returns[state])
                              for state in problem.states())
        if max_update_size < epsilon:
            break

    return returns, round_count

def random_action(problem, state):
    actions = list(problem.state_actions(state))
    return choice(actions) if actions else None


def policy_evaluation(problem, policy, returns, steps=None, epsilon=None):
    """
    """
    next_returns = {state: 0 for state in problem.states()}

    for i in (count() if steps is None else range(steps)):
        returns = next_returns

        next_returns = {
            state: (problem.state_reward(state)
                    + problem.discount() * sum(
                        problem.state_action_result_dist(
                            state, policy[state]
                        ).get(next_state, 0) * returns[next_state]
                        for next_state in problem.states()
                    )
            )
            for state in problem.states()
        }
        max_update_size = max(abs(next_returns[state] - returns[state])
                              for state in problem.states())
        if epsilon is not None and max_update_size < epsilon:
            break

    return returns
    
    

def policy_iteration_policy(problem):
    """
    Finds the optimal policy to a given MDP.
    Also produces an approximation of the optimal utility function.
    """
    returns = {state: 0 for state in problem.states()}
    policy = {state: random_action(problem, state) for state in problem.states()}

    while True:
        returns = policy_evaluation(problem, policy, returns, epsilon=0.001)

        next_policy = {
            state: max(problem.state_actions(state), key=lambda action: (
                sum(returns[next_state] * next_state_p
                    for next_state, next_state_p in (
                        problem.state_action_result_dist(state, action).items()
                    )
                )
            ))
            for state in problem.states()
        }

        if next_policy == policy:
            break

        policy = next_policy

    return policy

def display_gridworld_returns(problem, state_utility):
    for y in range(problem.height):
        print([round(state_utility[(x, y)], 3) for x in range(problem.width)])

def display_gridworld_value_iteration_returns():
    gamma = 1.0
    problem = GridWorldEnvironment(gamma=gamma)
    for epsilon in [0.5, 0.2, 0.1, 0.05, 0.001]:
        state_utility, rounds = value_iteration_returns(problem, epsilon=epsilon)
        print(f"Took {rounds} rounds for value_iteration to converge to epsilon={epsilon} with gamma={gamma}.")
        display_gridworld_returns(problem, state_utility)

def display_gridworld_policy(problem, policy):
    for y in range(problem.height):
        print([policy[(x, y)] for x in range(problem.width)])

def display_gridworld_policy_iteration_policy():
    gamma = 1.0
    problem = GridWorldEnvironment(gamma=gamma)
    opt_policy = policy_iteration_policy(problem)
    display_gridworld_policy(problem, policy)

def main():
    #display_gridworld_value_iteration_returns()
    display_gridworld_policy_iteration_policy()

if __name__ == "__main__":
    main()
