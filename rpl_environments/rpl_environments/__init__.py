from gym.envs.registration import register


register(
    id='ResidualFetchPickAndPlace-v0',
    entry_point='rpl_environments.envs:ResidualFetchPickAndPlaceEnv',
    timestep_limit=50,
)

register(
    id='ResidualFetchPush-v0',
    entry_point='rpl_environments.envs:ResidualFetchPushEnv',
    timestep_limit=50,
)

register(
    id='FetchPushHighFriction-v0',
    entry_point='rpl_environments.envs:FetchPushHighFrictionEnv',
    timestep_limit=50,
)

register(
    id='FetchHook-v0',
    entry_point='rpl_environments.envs:FetchHookEnv',
    timestep_limit=100,
)

register(
    id='ResidualFetchHook-v0',
    entry_point='rpl_environments.envs:ResidualFetchHookEnv',
    timestep_limit=100,
)

register(
    id='TwoFrameResidualHookNoisy-v0',
    entry_point='rpl_environments.envs:TwoFrameResidualHookNoisyEnv',
    timestep_limit=100,
)

register(
    id='TwoFrameHookNoisy-v0',
    entry_point='rpl_environments.envs:TwoFrameHookNoisyEnv',
    timestep_limit=100,
)

register(
    id = 'ResidualMPCPush-v0',
    entry_point='rpl_environments.envs:ResidualMPCPushEnv',
    timestep_limit=50,
)

register(
    id = 'MPCPush-v0',
    entry_point='rpl_environments.envs:MPCPushEnv',
    timestep_limit=50,
)

register(
    id='OtherPusherEnv-v0',
    entry_point='rpl_environments.envs:PusherEnv',
    timestep_limit=150,
)

register(
    id='ResidualOtherPusherEnv-v0',
    entry_point='rpl_environments.envs:ResidualPusherEnv',
    timestep_limit=150,
)

register(
    id='ComplexHookTrain-v0',
    entry_point='rpl_environments.envs:ComplexHookTrainEnv',
    timestep_limit=100,
)

register(
    id='ResidualComplexHookTrain-v0',
    entry_point='rpl_environments.envs:ResidualComplexHookTrainEnv',
    timestep_limit=100,
)

