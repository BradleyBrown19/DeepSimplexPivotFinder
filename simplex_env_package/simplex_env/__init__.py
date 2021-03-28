from gym.envs.registration import register

register(
    id='simplex-v0',
    entry_point='simplex_env.simplex:SimplexEnv',
)