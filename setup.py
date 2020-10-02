from setuptools import find_packages, setup

setup(
    name='rl_examples',
    version='0.1.0',

    description="Example RL agents using OpenAI Gym.",

    packages=find_packages(),

    entry_points={
        "console_scripts": [
            "rlmain=rl_examples.main:main",
        ],
    }
)
