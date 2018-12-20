from gym.envs.registration import register

register(
    id='ComplexHookTrain-v0',
    entry_point='gym_residual_complex_objects.envs:ComplexHookTrainEnv',
    timestep_limit=100,
)

register(
    id='ComplexHookTest-v0',
    entry_point='gym_residual_complex_objects.envs:ComplexHookTestEnv',
    timestep_limit=100,
)

register(
    id='ResidualComplexHookTrain-v0',
    entry_point='gym_residual_complex_objects.envs:ResidualComplexHookTrainEnv',
    timestep_limit=100,
)

register(
    id='ResidualComplexHookTest-v0',
    entry_point='gym_residual_complex_objects.envs:ResidualComplexHookTestEnv',
    timestep_limit=100,
)
