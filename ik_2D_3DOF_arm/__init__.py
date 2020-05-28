from gym.envs.registration import register

register(
    id='ik-2D-3DOF-arm-v0',
    entry_point='ik_2D_3DOF_arm.envs:IK2D3DOFARMEnv',
)

register(
    id='ik-2D-3DOF-arm-v1',
    entry_point='ik_2D_3DOF_arm.envs:IK2D3DOFARMEnv1',
)
