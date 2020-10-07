from argparse import ArgumentParser
from functools import partial
from math import sqrt
from pprint import pprint
from sys import argv
from time import sleep
from typing import Optional
import logging

import gym
import matplotlib.pyplot as plt
import numpy as np
from rich.progress import track, Progress

from rl_examples.agents import agent_name_func, discrete_agent_names, DiscritizingWrapperAgent
from rl_examples.discritizer import discritizer_name_func
from rl_examples.envs import name_grid_env

def single_episode(
        env,
        agent,
        render=False,
        sleep_duration: float = 0.05,
):
    """
    Note: We use S, A, R, all with t \in [0, ...). Reward is not shifted here.
    """
    logging.debug("Starting a new episode.")
    trace = []

    observation = env.reset()

    done = False
    while not done:
        action = agent.sample_action(observation)

        if render:
            sleep(sleep_duration)
            env.render()

        next_observation, reward, done, context = env.step(action)

        agent.observe(observation, action, reward)

        trace_tuple = (observation, action, reward)
        trace.append(trace_tuple)

        observation = next_observation

    agent.reset()

    return trace

def gym_returns(
        env_name="CartPole-v0",
        agent_name="sarsa",
        train_episode_count: int = 10000,
        test_episode_count: int = 200,
        agent_config=None,
        render_training: bool = False,
        render_testing: bool = False,
        discritizer_name="grid",
        sleep_duration: float = 0.05,
        **kwargs
):
    env = gym.make(env_name)

    logging.debug(f"Env action space: {env.action_space}")

    agent_func = agent_name_func[agent_name]

    if (agent_name in discrete_agent_names) and discritizer_name is not None:
        discritizer_func = discritizer_name_func[discritizer_name]

        agent_func = partial(
            DiscritizingWrapperAgent,
            wrapped_agent_func=agent_func,
            discritizer_func=discritizer_func,
        )

    agent = agent_func(
        observation_space=env.observation_space,
        action_space=env.action_space,
        env=env,
        **(agent_config or {}),
    )

    train_returns = []
    for episode_index in range(train_episode_count):
        trace = single_episode(
            env,
            agent,
            render_training,
            sleep_duration=sleep_duration,
        )
        train_returns.append(
            sum(reward for observation, action, reward in trace)
        )
        logging.debug(f"Train epsiode undiscounted return: {train_returns[-1]}")

    test_returns = []
    for display_index in range(test_episode_count):
        trace = single_episode(
            env,
            agent,
            render=render_testing,
        )
        test_returns.append(
            sum(reward for observation, action, reward in trace)
        )
        logging.debug(f"Test epsiode undiscounted return: {test_returns[-1]}")

    summary_doc = {
        "train_returns": train_returns,
        "test_returns": test_returns,
        "mean_train_returns": sum(train_returns) / len(train_returns),
        "mean_test_returns": sum(test_returns) / len(test_returns),
    }

    return summary_doc

def display_evaluation():
    logging.basicConfig(
        level="INFO",
    )

    episode_count = 100
    for env_name in ["CartPole-v1", "MountainCar-v0"]:
        for agent_name in ["montecarlo_q", "random", "sticky_random", "constant"]:

            if env_name.startswith("CartPole"):
                discritizer_name = "cartpole"
            else:
                discritizer_name = "grid"

            returns = gym_returns(
                env_name=env_name,
                agent_name=agent_name,
                episode_count=episode_count,
                discritizer_name=discritizer_name,
            )
            logging.info(f"[env={env_name}, agent={agent_name}] returns={returns}.")

def rolling_average(results, gamma=0.5):
    smoothed_results = []

    if len(results) == 0:
        return results

    average = results[0]
    for result in results:
        average = gamma * result + (1 - gamma) * average
        smoothed_results.append(average)

    return np.array(smoothed_results)

def smoothed_setting_runs(setting_runs, gamma=0.5):
    return np.array([
        [
            rolling_average(run_returns, gamma=gamma)
            for run_returns in setting_run_returns
        ]
        for setting_run_returns in setting_runs
    ])

def display_runs(
        setting_runs,
        labels=None,
        stdev=False,
        title=None,
        fig_name=None,
        smooth_gamma=None,
        show=False,
):
    n_settings, n_runs, n_steps = setting_runs.shape

    if smooth_gamma is not None:
        setting_runs = smoothed_setting_runs(setting_runs, smooth_gamma)

    for color_index, (label, runs) in enumerate(zip(labels, setting_runs)):
        color = f"C{color_index}"
        step_mean = runs.mean(axis=0)
        step_std = runs.std(axis=0)
        if stdev:
            step_std /= sqrt(n_runs)
        
        step_upper = step_mean + 1 * step_std
        step_lower = step_mean - 1 * step_std

        steps = np.arange(n_steps) + 1
        plt.plot(steps, step_mean, label=label, color=color, linestyle='-')
        plt.plot(steps, step_upper, color=color, linestyle='--')
        plt.plot(steps, step_lower, color=color, linestyle='--')
        plt.fill_between(
            steps,
            step_lower,
            step_upper,
            facecolor=color,
            alpha=0.15,
        )

    plt.legend()

    plt.xlabel("Step")
    plt.ylabel("Estimated Performance")

    if title:
        plt.title(title)

    if fig_name:
        plt.savefig(f"{fig_name}.png")
    
    if show:
        plt.show()

    
def display_run(
        progress_bar=None,
        **kwargs,
):
    logging.basicConfig(
        level="INFO",
    )

    if kwargs["env_name"].startswith("CartPole"):
        discritizer_name = "cartpole"
    elif "grid" in kwargs["env_name"].lower():
        discritizer_name = None
    else:
        discritizer_name = "grid"

    results = gym_returns(
        discritizer_name=discritizer_name,
        **kwargs,
    )

    logging.info(f"[env={kwargs['env_name']}, agent={kwargs['agent_name']}] returns={results['mean_test_returns']}.")

def display_group_performance(
        trial_count: int = 12,
        agent_names_str: Optional[str] = None,
        agent_name: Optional[str] = None,
        progress_bar = None,
        **kwargs,
):
    logging.basicConfig(
        level="WARN",
    )

    if kwargs["env_name"].startswith("CartPole"):
        discritizer_name = "cartpole"
    elif "grid" in kwargs["env_name"].lower():
        discritizer_name = None
    else:
        discritizer_name = "grid"

    if agent_names_str is None:
        agent_names_str = agent_name

    agent_names = agent_names_str.split(",")

    evaluate_agents_task = progress_bar.add_task(
        "[green]Evaluating agents...",
        total=len(agent_names),
    )

    results = []
    for agent_name in agent_names:

        agent_trials_task = progress_bar.add_task(
            f"  [red]Running {agent_name} trials...",
            total=trial_count,
        )

        agent_results = []
        for trial in range(trial_count):
            agent_results.append(
                gym_returns(
                    agent_name=agent_name,
                    discritizer_name=discritizer_name,
                    **kwargs,
                )
            )
            progress_bar.update(agent_trials_task, advance=1)

        results.append(agent_results)

        progress_bar.update(evaluate_agents_task, advance=1)

    for agent_name, agent_results in zip(agent_names, results):
        test_mean = sum(
            result["mean_test_returns"]
            for result in agent_results
        ) / len(results)

        logging.info(f"[{agent_name}] Mean test episode return: {test_mean}")

    display_runs(
        setting_runs=np.array([
            [
                list(result["train_returns"])
                for result in agent_results
            ]
            for agent_results in results
        ]),
        labels=agent_names,
        stdev=True,
        smooth_gamma=0.03,
        title=f"Estimated epoch return mean.",
        show=True,
    )
        


def rl_sim_arg_parser():
    agent_names = set(discrete_agent_names)
    env_names = {
        env.id
        for env in gym.envs.registry.all()
    }

    arg_parser = ArgumentParser(
        description="Simulate and analyze RL agents in various gym environments.",
    )
    arg_parser.add_argument(
        "--agent",
        dest="agent_name",
        type=str,
        choices=agent_names,
    )
    arg_parser.add_argument(
        "--agents",
        dest="agent_names_str",
        type=str,
    )
    arg_parser.add_argument(
        "--env",
        dest="env_name",
        type=str,
        required=True,
        # choices=env_names,
    )
    arg_parser.add_argument(
        "--sleep-duration",
        type=float,
        default=0.05,
    )
    arg_parser.add_argument(
        "--render-training",
        default=False,
        action="store_true",
    )
    arg_parser.add_argument(
        "--render-testing",
        default=False,
        action="store_true",
    )
    """
    arg_parser.add_argument(
        "--progress-bar",
        dest="display_progress_bar",
        default=False,
        action="store_true",
    )
    """
    arg_parser.add_argument(
        "--train-episodes",
        dest="train_episode_count",
        type=int,
        default=1000,
    )
    arg_parser.add_argument(
        "--test-episodes",
        dest="test_episode_count",
        type=int,
        default=200,
    )

    # Eval only.
    arg_parser.add_argument(
        "--trials",
        dest="trial_count",
        type=int,
        default=12,
    )

    return arg_parser

def rl_main_command_args():
    parser = rl_sim_arg_parser()
    args = dict(
        parser.parse_args()._get_kwargs()
    )
    if not (bool(args["agent_name"]) ^ bool(args["agent_names_str"])):
        parser.error("You must provide either --agent=agent1 or --agents=agent1,agent2,...")

    return args

def main():
    args = rl_main_command_args()

    command = "eval"

    command_func = {
        "run": display_run,
        "eval": display_group_performance,
    }[command]

    with Progress(transient=True) as progress_bar:
        plt.show()
        command_func(
            progress_bar=progress_bar,
            **args,
        )

if __name__ == "__main__":
    main()
