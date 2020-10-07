from collections import defaultdict
from functools import partial
import logging
from typing import Any, Callable, Dict, Optional
from random import random

from rl_examples.agents.baseline import Agent
from rl_examples.discritizer import Discritizer, GridDiscritizer

"""
def sample_dist(dist_dict):
    actions, probs = zip(*dist_dict.items())
    return choices(actions, weights=probs, k=1)[0]
"""

class DiscritizingWrapperAgent(Agent):
    """
    Wrapper for agents using a discrete-state learning algorithm.
    TODO: Make a generic compatibility-layer BaseAgent.
    """

    def __init__(
            self,
            wrapped_agent_func: Agent,
            discritizer_func: Optional[Discritizer] = None,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)

        discritizer_func = discritizer_func or partial(
            GridDiscritizer,
            increment_count=15, # "Sane" defaults.
        )
        self.discritizer = discritizer_func(self.observation_space)

        self.wrapped_agent = wrapped_agent_func(
            observation_space=self.discritizer.output_space(),
            action_space=self.action_space,
        )

    def observe(self, observation, action, reward):
        discrete_observation = self.discritizer.discritized(observation)
        self.wrapped_agent.observe(discrete_observation, action, reward)

    def sample_action(self, observation):
        discrete_observation = self.discritizer.discritized(observation)
        return self.wrapped_agent.sample_action(discrete_observation)

    def reset(self):
        self.wrapped_agent.reset()

def first_seen_observation_actions(trace):
    seen_observation_actions = set()

    for observation, action, _ in trace:
        yield (observation, action) not in seen_observation_actions
        seen_observation_actions.add((observation, action))

def grid_actions_str(actions, velocity=False):
    if velocity:
        str_actions = list(enumerate([
            "(<v)",
            "(<)",
            "(<^)",
            "(v)",
            "()",
            "(^)",
            "(>v)",
            "(>)",
            "(>^)",
        ]))
    else:
        str_actions = list(enumerate("<^v>"))

    total_action_str = ""
    for action, action_str in str_actions:
        if action in actions:
            total_action_str += action_str
        else:
            total_action_str += " "

    return total_action_str

def best_action(action_values: Dict[Any, Any]) -> Optional[Any]:
    if not action_values:
        return None

    return max(
        action_values.items(),
        key=lambda action_value: action_value[1],
    )[0]

def trace_return(trace, final_value, gamma):
    return sum(
        (gamma ** reward_index) * reward
        for reward_index, (observation, action, reward) in enumerate(trace)
    ) + (gamma ** len(trace)) * final_value


class EGreedyPolicyAgent(Agent):
    def __init__(
            self,
            exploration_rate: float = 0.05,
            *args,
            **kwargs,
    ):
        self.exploration_rate = exploration_rate

        super().__init__(*args, **kwargs)

    def reset_params(self):
        super().reset_params()

        self.state_action_value = defaultdict(
            lambda: defaultdict(lambda: 0),
        )

    def egreedy_action(
            self,
            best_action: Dict[Any, Dict[Any, float]],
    ):
        if bool(best_action) and not (random() < self.exploration_rate):
            return best_action
        else:
            return self.action_space.sample()

    def sample_action(self, observation):
        return self.egreedy_action(
            best_action(self.state_action_value[observation]),
        )

class FirstVisitOnPolicyMCControlAgent(EGreedyPolicyAgent):
    def __init__(
            self,
            *args,
            **kwargs,
    ):
        super().__init__(
            trace_length=-1, # Hard requirement for MC learning.
            *args,
            **kwargs,
        )

    def reset_params(self):
        super().reset_params()

        self.state_action_returns = defaultdict(
            lambda: defaultdict(list),
        )

    def learn(self):
        include_observation = list(first_seen_observation_actions(self.trace))

        updated_states = set()
        state_returns = defaultdict(list) # For debugging vis.

        returns = 0
        for t in range(len(self.trace) - 1, -1, -1):
            state, action, reward = self.trace[t]

            returns = self.gamma * returns + reward

            if include_observation[t]:
                self.state_action_returns[state][action].append(returns)
                state_returns[state].append(returns)
                updated_states.add(state)

        # Update actual state-action value estimates for acting.
        for state in updated_states:
            for action, returns in self.state_action_returns[state].items():
                self.state_action_value[state][action] = (
                    sum(returns) / len(returns)
                )

        self.display_policy_summary()
        self.display_trace_summary(self.trace, state_returns)

    def reset(self):
        if len(self.trace) > 1:
            self.learn()

        super().reset()

    def display_policy_summary(self, action_counts=False):

        if action_counts:
            print("Action counts:")
            self.render_agent_data({
                state: ", ".join(
                    f"{action}={len(returns)}"
                    for action, returns in sorted(action_returns.items())
                )
                for state, action_returns in self.state_action_returns.items()
            })

        print("Mean action returns:")
        self.render_agent_data({
            state: ", ".join(
                f"{action}={sum(returns)/len(returns):2.3}"
                for action, returns in sorted(action_returns.items())
            )
            for state, action_returns in self.state_action_returns.items()
        })

        print("Max returns:")
        self.render_agent_data({
            state: f"{max_return:3.2}"
            for state, action_returns in self.state_action_returns.items()
            for max_return in (
                    max([sum(returns) / len(returns) for returns in action_returns.values()]),
            )
        })

        print("Current policy:")
        self.render_agent_data({
            state: (
                grid_actions_str(
                    {self.state_best_action[state]},
                   velocity=(self.action_space.n > 4),
                )
                if state in self.state_best_action
                else ""
            )
            for state in self.state_action_returns.keys()
        })

    def display_trace_summary(self, trace, state_returns):
        state_actions = {}
        for state, action, _ in trace:
            state_actions.setdefault(state, set()).add(action)

        print("Last episode:")
        self.render_agent_data({
            state: grid_actions_str(
                actions,
                velocity=(self.action_space.n > 4),
            )
            for state, actions in state_actions.items()
        })

        print("Max episode state-action returns for state:")
        self.render_agent_data({
            state: (f"{max(state_returns[state]):3.3}"
                    if state in state_returns
                    else "")
            for state, actions in state_actions.items()
        })

class TDNStepEGreedyPolicyAgent(EGreedyPolicyAgent):
    def __init__(
            self,
            n_steps: int = 4,
            alpha: float = 0.05,
            trace_length: Optional[int] = None,
            *args,
            **kwargs,
    ):
        self.alpha = alpha
        self.n_steps = n_steps

        if trace_length is None:
            trace_length = n_steps + 1

        assert (trace_length == -1) or (trace_length >= n_steps + 1), (
            f"TD(n) methods a trace_length >= n_steps + 1 " +
            f"({trace_length} <= {n_steps + 1})."
        )

        super().__init__(
            trace_length=trace_length,
            *args,
            **kwargs,
        )

    def observe(self, observation, action, reward):
        super().observe(observation, action, reward)

        if len(self.trace) >= self.n_steps + 1:
            self.learn(self.trace[-(self.n_steps + 1):])

    def reset(self):
        """
        When doing n-step learning, we need to do up to (n_steps - 1)
        episode termination learning steps, because the rest are covered in
        observe() calls.
        """
        # Do each of -(min(n_steps - 1, len(trace)):, ..., -2:.
        # So typical case is [-n_steps-1:], [-n_steps:], ... [-2:].
        # At least two elements needed, so stop when -1 is reached / don't learn.
        for start_index in range(min(self.n_steps + 1, len(self.trace)), 1, -1):
            self.learn(self.trace[-start_index:])

        super().reset()


class TDZeroEGreedyAgent(TDNStepEGreedyPolicyAgent):
    def __init__(
            self, 
            trace_length: int = 2,
            *args,
            **kwargs,
    ):
        assert trace_length == -1 or trace_length >= 2, (
            f"TD(0) methods need a trace_length >= 2 (not {trace_length}.)"
        )

        super().__init__(
            trace_length=trace_length,
            n_steps=1,
            *args,
            **kwargs,
        )
        
class SARSAOnPolicyTDZeroAgent(TDZeroEGreedyAgent):
    def __init__(
            self,
            trace_type: str = "replacing",
            *args,
            **kwargs,
    ):
        assert trace_type in ("accumulating", "replacing")
        self.trace_type = trace_type

        super().__init__(
            *args,
            **kwargs,
        )

    def learn(self, trace):
        prev_observation, prev_action, prev_reward = trace[-2]
        observation, action, reward = trace[-1]

        td_error = (
            reward
            + self.gamma * self.state_action_value[observation][action]
            - self.state_action_value[prev_observation][prev_action]
        )

        self.state_action_value[prev_observation][prev_action] += (
            self.alpha * td_error
        )
        
class ESARSAOffPolicyTDZeroAgent(TDZeroEGreedyAgent):
    def __init__(
            self, 
            *args,
            **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )

    def learn(self, trace):
        prev_observation, prev_action, prev_reward = trace[-2]
        observation, action, reward = trace[-1]

        action_values = self.state_action_value[observation]

        best_current_action = best_action(action_values)

        # TODO: Implement policy().log_prob, log_prob_dict
        expected_reward = sum(
            ( # pi(a | s)
                (1 - self.exploration_rate)
                if action == best_current_action else
                self.exploration_rate / len(action_values)
                # How to handle unknown/inf |A|!?!
            )
            * action_value # Q('s, 'a)
            for action, action_value in action_values.items()
        )

        esarsa_td_error = (
            reward
            + self.gamma * expected_reward
            - self.state_action_value[prev_observation][prev_action]
        )

        self.state_action_value[prev_observation][prev_action] += (
            self.alpha * esarsa_td_error
        )

class QLearningOffPolicyTDZeroAgent(TDZeroEGreedyAgent):
    def __init__(
            self, 
            *args,
            **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )

    def learn(self, trace):
        prev_observation, prev_action, prev_reward = self.trace[-2]
        observation, action, reward = self.trace[-1]

        action_values = self.state_action_value[observation]

        best_action_value = action_values[best_action(action_values)] # max()

        q_learning_td_error = (
            reward
            + self.gamma * best_action_value
            - self.state_action_value[prev_observation][prev_action]
        )

        self.state_action_value[prev_observation][prev_action] += (
            self.alpha * q_learning_td_error
        )

class QLearningOffPolicyTDNStepAgent(TDNStepEGreedyPolicyAgent):
    def __init__(
            self,
            *args,
            **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )

    def learn(self, trace):
        first_observation, first_action, first_reward = trace[0]

        if len(trace) >= self.n_steps + 1:
            final_observation, final_action, final_reward = trace[-1]

            final_action_values = self.state_action_value[final_observation]
            best_final_action_value = final_action_values[best_action(final_action_values)]

            trace = trace[:-1]
        else:
            best_final_action_value = 0

        n_step_bootstrapped_returns = (
            trace_return(
                trace,
                best_final_action_value,
                gamma=self.gamma,
            )
        )

        n_step_q_learning_td_error = (
            n_step_bootstrapped_returns
            - self.state_action_value[first_observation][first_action]
        )

        self.state_action_value[first_observation][first_action] += (
            self.alpha * n_step_q_learning_td_error
        )

class SARSAOnPolicyTDNStepAgent(TDNStepEGreedyPolicyAgent):
    def __init__(
            self,
            *args,
            **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )

    def learn(self, trace):
        first_observation, first_action, first_reward = trace[0]

        if len(trace) >= self.n_steps + 1:
            final_observation, final_action, final_reward = trace[-1]
            final_action_value = self.state_action_value[final_observation][final_action]

            trace = trace[:-1]
        else:
            final_action_value = 0

        n_step_bootstrapped_returns = (
            trace_return(
                trace,
                final_action_value,
                gamma=self.gamma,
            )
        )

        n_step_td_error = (
            n_step_bootstrapped_returns
            - self.state_action_value[first_observation][first_action]
        )

        self.state_action_value[first_observation][first_action] += (
            self.alpha * n_step_td_error
        )

class ESARSAOffPolicyTDNStepAgent(TDNStepEGreedyPolicyAgent):
    def __init__(
            self,
            *args,
            **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )

    def learn(self, trace):
        first_observation, first_action, first_reward = trace[0]

        if len(trace) >= self.n_steps + 1:
            final_observation, final_action, final_reward = trace[-1]

            final_action_values = self.state_action_value[final_observation]
            best_final_action = best_action(final_action_values)

            # TODO: Implement policy().log_prob, log_prob_dict
            final_expected_action_value = sum(
                ( # pi(a | s)
                    (1 - self.exploration_rate)
                    if action == best_final_action else
                    self.exploration_rate / len(final_action_values)
                    # How to handle unknown/inf |A|!?!
                )
                * action_value # Q('s, 'a)
                for action, action_value in final_action_values.items()
            )

            trace = trace[:-1]
        else:
            final_expected_action_value = 0

        n_step_bootstrapped_returns = (
            trace_return(
                trace,
                final_expected_action_value,
                gamma=self.gamma,
            )
        )

        n_step_td_error = (
            n_step_bootstraped_returns
            - self.state_action_value[first_observation][first_action]
        )

        self.state_action_value[first_observation][first_action] += (
            self.alpha * n_step_td_error
        )

def tree_backup_returns(
        state_action_value,
        policy_func,
        trace,
        gamma,
):
    t = sum(
        reward + (gamma ** trace_index) * sum(
            policy_func(state, other_action) * other_action_value
            for other_action, other_action_value in state_action_value[state].items()
            if (other_action != taken_action) or (trace_index == (len(trace) - 1))
        )
        for trace_index, (state, taken_action, reward) in enumerate(trace)
    )

    return t


def egreedy_policy_probs_func(state_action_value, exploration_rate):

    def egreedy_policy_prob(state, action):
        action_values = state_action_value[state]
        state_best_action = best_action(action_values)
        return (
            (1 - exploration_rate)
            if action == state_best_action else
            exploration_rate / len(action_values)
        )

    return egreedy_policy_prob

class TreeBackupOffPolicyTDNStepAgent(TDNStepEGreedyPolicyAgent):
    def __init__(
            self,
            *args,
            **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )

    def learn(self, trace):
        first_observation, first_action, first_reward = trace[0]

        tree_backup_return_est = tree_backup_returns(
            self.state_action_value,
            policy_func=egreedy_policy_probs_func(
                self.state_action_value,
                self.exploration_rate,
            ),
            trace=trace,
            gamma=self.gamma,
        )

        tree_td_error = (
            tree_backup_return_est
            - self.state_action_value[first_observation][first_action]
        )

        self.state_action_value[first_observation][first_action] += (
            self.alpha * tree_td_error
        )

# TODO: Do learning _before_ action selection, like in canonical algos, rather
# than after.
class SARSAOnPolicyTDLambdaAgent(EGreedyPolicyAgent):
    def __init__(
            self,
            trace_lambda: float = 0.7,
            trace_clip_min: float = 0.0,
            trace_type: str = "accumulating",
            alpha: float = 0.05,
            trace_length: Optional[int] = None,
            *args,
            **kwargs,
    ):
        self.alpha = alpha

        self.trace_lambda = trace_lambda
        self.trace_clip_min = trace_clip_min

        if trace_length is None:
            trace_length = 2

        assert (trace_length == -1) or (trace_length >= 2), (
            f"TD(\lambda) methods a trace_length >= 2 " +
            f"({trace_length} <= 2)."
        )

        assert trace_type in ("accumulating", "replacing")
        self.trace_type = trace_type

        super().__init__(
            trace_length=trace_length,
            *args,
            **kwargs,
        )

    def reset_params(self):
        super().reset_params()

        # Accumulated dQ(s, a; \theta)/d\theta.
        self.state_action_eligibility_trace = defaultdict(
            lambda: defaultdict(lambda: 0),
        )

    def reset(self):
        self.state_action_eligibility_trace = defaultdict(
            lambda: defaultdict(lambda: 0),
        )

        super().reset()

    def observe(self, observation, action, reward):
        super().observe(observation, action, reward)

        # We receive observations as S, A, R triples, so we delay learning by
        # one step and need two steps before starting.
        if len(self.trace) >= 2:
            self.learn(self.trace[-2:])

        self.update_eligibility_trace(observation, action)

    def update_eligibility_trace(self, observation, action):
        for state, action_trace in self.state_action_eligibility_trace.items():
            for action in action_trace.keys():
                action_trace[action] *= self.trace_lambda * self.gamma

                # Performance optimization.
                if action_trace[action] < self.trace_clip_min:
                    del action_trace[action]

        if self.trace_type == "accumulating":
            self.state_action_eligibility_trace[observation][action] += 1
        elif self.trace_type == "replacing":
            self.state_action_eligibility_trace[observation][action] = 1

    def learn(self, trace):
        assert len(trace) == 2
        prev_observation, prev_action,  prev_reward = trace[0]
        observation, action,  reward = trace[1]

        td_error = (
            reward
            + self.gamma * self.state_action_value[observation][action]
            - self.state_action_value[prev_observation][prev_action]
        )

        for state, action_value in self.state_action_eligibility_trace.items():
            for action, action_eligibility_trace in action_value.items():
                self.state_action_value[prev_observation][prev_action] += (
                    self.alpha * td_error * action_eligibility_trace
                )
