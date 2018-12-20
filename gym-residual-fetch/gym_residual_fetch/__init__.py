from gym.envs.registration import register

register(
    id='MyFetchPush-v0',
    entry_point='gym_residual_fetch.envs:MyFetchPushEnv',
    timestep_limit=50,
)
register(
    id='MyFetchPickAndPlace-v0',
    entry_point='gym_residual_fetch.envs:MyFetchPickAndPlaceEnv',
    timestep_limit=50,
)
register(
    id='ResidualFetchPickAndPlace-v0',
    entry_point='gym_residual_fetch.envs:ResidualFetchPickAndPlaceEnv',
    timestep_limit=50,
)

register(
    id='ResidualFetchPush-v0',
    entry_point='gym_residual_fetch.envs:ResidualFetchPushEnv',
    timestep_limit=50,
)

register(
    id='FetchPushHighFriction-v0',
    entry_point='gym_residual_fetch.envs:FetchPushHighFrictionEnv',
    timestep_limit=50,
)

register(
    id='FetchHook-v0',
    entry_point='gym_residual_fetch.envs:FetchHookEnv',
    timestep_limit=100,
)
register(
    id='NoisyFetchHook-v0',
    entry_point='gym_residual_fetch.envs:NoisyFetchHookEnv',
    timestep_limit=100,
)

register(
    id='ResidualFetchHook-v0',
    entry_point='gym_residual_fetch.envs:ResidualFetchHookEnv',
    timestep_limit=100,
)

register(
    id='NoisyResidualFetchHook-v0',
    entry_point='gym_residual_fetch.envs:NoisyResidualFetchHookEnv',
    timestep_limit=100,
)

register(
    id='TwoFrameResidualHookNoisy-v0',
    entry_point='gym_residual_fetch.envs:TwoFrameResidualHookNoisyEnv',
    timestep_limit=100,
)

register(
    id='TwoFrameHookNoisy-v0',
    entry_point='gym_residual_fetch.envs:TwoFrameHookNoisyEnv',
    timestep_limit=100,
)

register(
    id = 'ResidualMPCPush-v0',
    entry_point='gym_residual_fetch.envs:ResidualMPCPushEnv',
    timestep_limit=50,
)

register(
    id = 'MPCPush-v0',
    entry_point='gym_residual_fetch.envs:MPCPushEnv',
    timestep_limit=50,
)
