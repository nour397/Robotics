from gym.envs.registration import register

register(
    id="gym_examples/Poppy-v0",
    entry_point="gym_examples.envs:PoppyEnv",
)
