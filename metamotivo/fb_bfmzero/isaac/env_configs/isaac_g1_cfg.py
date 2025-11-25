from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg

##
# Pre-defined configs
##
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.config.spot.mdp as spot_mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.envs.mdp.commands import commands_cfg

from metamotivo.fb_bfmzero.isaac import configs as envs_configs
from metamotivo.fb_bfmzero.isaac.mdps.rewards import foot

from metamotivo.fb_bfmzero.isaac.robots import g1
from metamotivo.fb_bfmzero.isaac.mdps.events import events as mdp_events
from metamotivo.fb_bfmzero.isaac.mdps.events import motion_reset
from metamotivo.fb_bfmzero.isaac.mdps.obs import privileged
##
# Scene definition
##

VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
    )
    # robots
    robot: ArticulationCfg = MISSING
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True, force_threshold=10.0, debug_vis=True
    )

##
# MDP settings
##
@configclass
class CommandsCfg:
    """Command specifications for the MDP."""
    base_velocity = commands_cfg.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=False,
        ranges=commands_cfg.UniformVelocityCommandCfg.Ranges(
            #lin_vel_x=(0, 4.5), lin_vel_y=(-0.75, 0.75), ang_vel_z=(-2., 2.), heading=(0., 0)
            lin_vel_x=(0, 0.3), lin_vel_y=(-0.05, 0.05), ang_vel_z=(-0.05, 0.05), heading=(0., 0)
        )
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        # observation terms (order preserved)
        # base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.5, n_max=0.5))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        # actions = ObsTerm(func=mdp.last_action)

        history_length: int = 4
        flatten_history_dim: int = False

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    @configclass
    class PrivilegedCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_pos_z = ObsTerm(func=mdp.base_pos_z)

        local_body_lin_vel = ObsTerm(func=privileged.local_body_lin_vel,
                            params={"asset_cfg": SceneEntityCfg("robot")})

        local_body_ang_vel = ObsTerm(func=privileged.local_body_ang_vel,
                            params={"asset_cfg": SceneEntityCfg("robot")})

        local_body_pos = ObsTerm(func=privileged.local_body_pos,
                            params={"asset_cfg": SceneEntityCfg("robot")})

        local_body_quat = ObsTerm(func=privileged.local_body_quat,
                            params={"asset_cfg": SceneEntityCfg("robot")})

        joint_acc = ObsTerm(func=privileged.joint_acc,
                            params={"asset_cfg": SceneEntityCfg("robot")}
                            )
        joint_stiffness = ObsTerm(func=privileged.joint_stiffness,
                                  params={"asset_cfg": SceneEntityCfg("robot")}
                                  )
        joint_damping = ObsTerm(func=privileged.joint_damping,
                                params={"asset_cfg": SceneEntityCfg("robot")})

        friction_coeff = ObsTerm(func=privileged.joint_friction_coeff,
                                params={"asset_cfg": SceneEntityCfg("robot")})

        torques = ObsTerm(func=privileged.joint_torques,
                          params={"asset_cfg": SceneEntityCfg("robot")})

        feet_status = ObsTerm(func=privileged.feet_contact_status,
                              params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
                                      "threshold": 1.0})

        feet_forces = ObsTerm(func=privileged.feet_contact_forces,
                              params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: PrivilegedCfg = PrivilegedCfg()


@configclass
class EventCfg:
    """Configuration for events."""
    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.6),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )

    add_joint_default_pos = EventTerm(
        func=mdp_events.randomize_joint_default_pos,  ## TODO
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "pos_distribution_params": (-0.01, 0.01),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(6, 15.0),
        params={"velocity_range": VELOCITY_RANGE},
    )

    state_reset = EventTerm(
        func=motion_reset.StateInit,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "motions_root": envs_configs.motions_root,
            "motions_sample_ratio": 0.8
        },
    )

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-1)
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
                ],
            ),
            "threshold": 1.0,
        },
    )
    foot_slip = RewTerm(
        func=spot_mdp.foot_slip_penalty,
        weight=-2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": 1.0,
        },
    )
    ankle_roll_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-4,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_ankle_roll_joint")
        },
    )
    feet_orientation = RewTerm(
        func=foot.penalize_feet_orientation,
        weight=-0.4,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_ankle_roll_joint")
        },
    )




@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    pass


##
# Environment configuration
##

@configclass
class IsaacEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=1024, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 10.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # viewer settings
        self.viewer.eye = (1.5, 1.5, 1.5)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"

        self.scene.robot = g1.G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = g1.G1_ACTION_SCALE


@configclass
class IsaacEnvCfg_Play(IsaacEnvCfg):

    def __post_init__(self):
        super(IsaacEnvCfg_Play, self).__post_init__()
        self.scene.num_envs = 2


