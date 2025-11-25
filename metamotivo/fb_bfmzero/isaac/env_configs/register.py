import gymnasium as gym

gym.register(
    id="FB-G129dof-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "metamotivo.fb_bfmzero.isaac.env_configs.isaac_g1_cfg:IsaacEnvCfg",
    },
)


gym.register(
    id="FB-G129dof-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "metamotivo.fb_bfmzero.isaac.env_configs.isaac_g1_cfg:IsaacEnvCfg_Play",
    },
)
