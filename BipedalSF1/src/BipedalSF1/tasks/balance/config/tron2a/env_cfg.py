# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SF-TRON2A balance and velocity-tracking task on mixed terrain."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.physics import PhysxAutoCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import (
    HfRandomUniformTerrainCfg,
    MeshPlaneTerrainCfg,
    MeshPyramidStairsTerrainCfg,
    TerrainGeneratorCfg,
    TerrainImporterCfg,
)
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg
from isaaclab_newton.physics import (
    KaminoPADMMSolverCfg,
    MJWarpSolverCfg,
    NewtonCfg,
    NewtonCollisionPipelineCfg,
)
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab.visualizers import VisualizerCfg
from isaaclab_tasks.core.velocity import mdp
from isaaclab_tasks.utils import PresetCfg

from BipedalSF1.assets import TRON2A_USD_PATH


TRON2A_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=8,
    num_cols=8,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    sub_terrains={
        "flat": MeshPlaneTerrainCfg(proportion=0.25),
        "stairs": MeshPyramidStairsTerrainCfg(
            proportion=0.35,
            step_height_range=(0.025, 0.12),
            step_width=0.30,
            platform_width=2.0,
            border_width=0.5,
            holes=False,
        ),
        "uneven": HfRandomUniformTerrainCfg(
            proportion=0.40,
            noise_range=(0.01, 0.06),
            noise_step=0.01,
            border_width=0.25,
            downsampled_scale=0.2,
        ),
    },
)


@configclass
class Tron2APhysicsCfg(PresetCfg):
    """Physics presets for Newton MJWarp and Isaac Sim/PhysX."""

    isaacsim_physx: PhysxCfg = PhysxCfg()
    ovphysx: OvPhysxCfg = OvPhysxCfg()
    physx: PhysxAutoCfg = PhysxAutoCfg(isaacsim_physx=isaacsim_physx, ovphysx=ovphysx)
    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=512,
            nconmax=128,
            cone="pyramidal",
            impratio=1.0,
            integrator="implicitfast",
            use_mujoco_contacts=False,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(max_triangle_pairs=2_500_000),
        num_substeps=2,
        debug_mode=False,
    )
    newton_kamino: NewtonCfg = NewtonCfg(solver_cfg=KaminoPADMMSolverCfg())
    default: NewtonCfg = newton_mjwarp


@configclass
class Tron2ASceneCfg(InteractiveSceneCfg):
    """Curriculum terrain scene containing the twelve-joint biped."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TRON2A_TERRAINS_CFG,
        max_init_terrain_level=1,
        collision_group=-1,
        physics_material=sim_utils.PhysxRigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(TRON2A_USD_PATH),
            activate_contact_sensors=True,
            joint_drive_props=sim_utils.UsdPhysicsDriveCfg(),
            ensure_drives_exist=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.90),
            joint_pos={".*": 0.0},
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.9,
        actuators={
            "leg_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*_Joint"],
                stiffness=180.0,
                damping=8.0,
            ),
        },
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_Link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.15, size=(0.9, 0.6)),
        mesh_prim_paths=["/World/ground"],
        global_world_only=True,
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


@configclass
class CommandsCfg:
    """Walking commands; zero velocity commands train standing balance."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 8.0),
        rel_standing_envs=0.2,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 0.8),
            lin_vel_y=(-0.25, 0.25),
            ang_vel_z=(-0.5, 0.5),
        ),
    )


@configclass
class ActionsCfg:
    """Joint position targets around the robot's neutral pose."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*_Joint"],
        scale=0.35,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Proprioceptive observations for balance and locomotion."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=UniformNoiseCfg(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=UniformNoiseCfg(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventsCfg:
    """Reset perturbations and occasional pushes for robust balancing."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "yaw": (-0.15, 0.15)},
            "velocity_range": {
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1),
                "z": (-0.05, 0.05),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.1, 0.1),
            },
        },
    )
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.04, 0.04),
            "velocity_range": (-0.1, 0.1),
        },
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(8.0, 12.0),
        params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.2, 0.2)}},
    )


@configclass
class RewardsCfg:
    """Rewards for standing upright and tracking walking commands."""

    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    upright = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-5.0,
        params={"target_height": 0.88, "sensor_cfg": SceneEntityCfg("height_scanner")},
    )
    vertical_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    roll_pitch_rate = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    joint_acceleration = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    joint_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_Joint")},
    )
    foot_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="ankle_pitch_[LR]_Link"),
            "threshold": 0.3,
        },
    )


@configclass
class TerminationsCfg:
    """End an episode on a fall or after the time limit."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_Link"), "threshold": 1.0},
    )
    low_base = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.45},
    )


@configclass
class CurriculumCfg:
    """Advance environments to harder terrain levels as they learn to walk."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


@configclass
class Tron2ABalanceWalkEnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based balance and terrain walking environment for SF-TRON2A."""

    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(physics=Tron2APhysicsCfg())
    scene: Tron2ASceneCfg = Tron2ASceneCfg(num_envs=256, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics.default = self.sim.physics.newton_mjwarp
        self.sim.visualizer_cfgs = VisualizerCfg(eye=(3.5, 3.5, 2.5), lookat=(0.0, 0.0, 0.8))
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt
