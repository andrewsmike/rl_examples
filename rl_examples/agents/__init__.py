from functools import partial

from rl_examples.agents.baseline import (
    Agent,
    ConstantAgent,
    RandomAgent,
    StickyManualAgent,
    StickyRandomAgent,
)
from rl_examples.agents.discrete import (
    DiscritizingWrapperAgent,
    ESARSAOffPolicyTDZeroAgent,
    ESARSAOffPolicyTDNStepAgent,
    FirstVisitOnPolicyMCControlAgent,
    QLearningOffPolicyTDNStepAgent,
    QLearningOffPolicyTDZeroAgent,
    SARSAOnPolicyTDLambdaAgent,
    SARSAOnPolicyTDNStepAgent,
    SARSAOnPolicyTDZeroAgent,
    TreeBackupOffPolicyTDNStepAgent,
)

from rl_examples.discritizer import GridDiscritizer

agent_name_func = {
    "constant": ConstantAgent,
    "random": RandomAgent,
    "sticky_manual": StickyManualAgent,
    "sticky_random": StickyRandomAgent,

    # MC control w/ e-greedy strategy.
    "first_visit_mc": FirstVisitOnPolicyMCControlAgent,

    # TD control w/ max(action_values) update rule.
    "qlearning": QLearningOffPolicyTDZeroAgent,
    "qlearning_nstep": QLearningOffPolicyTDNStepAgent,

    # TD control w/ expected(action_values) update rule.
    "esarsa": ESARSAOffPolicyTDZeroAgent,
    "esarsa_nstep": ESARSAOffPolicyTDNStepAgent,
    "tree_nstep": TreeBackupOffPolicyTDNStepAgent,

    # TD control w/ sampled(action_value) update rule.
    "sarsa": SARSAOnPolicyTDZeroAgent,
    "sarsa_nstep": SARSAOnPolicyTDNStepAgent,
    "sarsa_lambda": SARSAOnPolicyTDLambdaAgent,
}

discrete_agent_names = {
    "first_visit_mc",

    "qlearning",
    "qlearning_nstep",

    "esarsa",
    "esarsa_nstep",
    "tree_nstep",

    "sarsa",
    "sarsa_nstep",
    "sarsa_lambda",
}
