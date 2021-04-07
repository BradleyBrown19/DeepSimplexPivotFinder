from gym.envs.registration import register

register(
    id='spicy-v0',
    entry_point='spicy_env.env_v2:SpicyGym',
)