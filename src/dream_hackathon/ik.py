"""Inverse kinematics for teleoperation: mink's differential IK on a MuJoCo model of one arm.

Each control tick moves the arm's joints toward the tool target by a bounded step: every arm joint stays inside its
limits (less a margin) and under a speed cap, so a target out of reach or a jump of the controller never makes the
arm lunge; it saturates and follows. An arm that starts outside its limits (resting against a stop) is eased back in
at the same speed cap, never snapped. The YAM's model is i2rt's own MuJoCo model (arms/yam.py); a MakerMods arm's is
built from its URDF here (urdf_model()), with a site at its measured tool center point.

@see ArmIK, urdf_model()
"""

import re
from collections.abc import Sequence

import mink
import mujoco
import numpy as np
import numpy.typing as npt

## Solver iterations per control tick; each moves at most the speed cap times its share of the tick.
_ITERATIONS_PER_TICK = 3
## Name of the tool site urdf_model() adds.
TCP_SITE = "tcp"


class ArmIK:
    """Differential IK for one six-joint arm, driving one site of its model to a target pose.

    Joint values go in and out in the arm's own convention: the model's joint is `sign * joint`.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        site: str,
        joint_names: Sequence[str],
        max_joint_speed_rad_s: float,
        joint_limit_margin_rad: float,
        joint_signs: Sequence[float] | None = None,
    ) -> None:
        """Builds the solver.

        @param model The arm's MuJoCo model.
        @param site The site to drive, e.g. the tool center point.
        @param joint_names The model's six arm joints, base to wrist.
        @param max_joint_speed_rad_s No joint moves faster than this.
        @param joint_limit_margin_rad Every joint stays this far inside its model limits.
        @param joint_signs Per joint, +1 or -1; None for all +1.
        """
        self._site = site
        self._configuration = mink.Configuration(model)
        joint_ids = [model.joint(name).id for name in joint_names]
        self._qpos_index = np.array([model.jnt_qposadr[i] for i in joint_ids])
        self._signs = np.ones(len(joint_names)) if joint_signs is None else np.asarray(joint_signs, dtype=np.float64)
        self._max_speed = max_joint_speed_rad_s
        # The arm joints' limits less the margin, in the model's convention; an unlimited joint is unbounded.
        low = np.array([model.jnt_range[i][0] if model.jnt_limited[i] else -np.inf for i in joint_ids])
        high = np.array([model.jnt_range[i][1] if model.jnt_limited[i] else np.inf for i in joint_ids])
        middle = np.where(np.isfinite(low + high), (low + high) / 2, 0.0)
        self._low = np.minimum(low + joint_limit_margin_rad, middle)
        self._high = np.maximum(high - joint_limit_margin_rad, middle)
        self._task = mink.FrameTask(site, "site", position_cost=1.0, orientation_cost=0.5, lm_damping=1e-3)
        # The margin is applied to the arm joints above, not here: a gripper's prismatic finger joints, also in the
        # model, travel less than two margins.
        self._limits = [
            mink.ConfigurationLimit(model, min_distance_from_limits=0.0),
            mink.VelocityLimit(model, {name: max_joint_speed_rad_s for name in joint_names}),
        ]

    def fk(self, joints_rad: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """The site's pose for the given joints.

        @param joints_rad The six joints, radians.
        @return `[4, 4]` pose in the arm's base frame.
        """
        self._set_joints(joints_rad)
        return self._configuration.get_transform_frame_to_world(self._site, "site").as_matrix()

    def step(
        self,
        target: npt.NDArray[np.float64],
        joints_rad: npt.NDArray[np.float64],
        dt_s: float,
    ) -> npt.NDArray[np.float64]:
        """One control tick's step from `joints_rad` toward the target pose.

        @param target `[4, 4]` site pose in the arm's base frame.
        @param joints_rad The six joints the step starts from, radians.
        @param dt_s The tick's length; the step is at most the speed cap times it.
        @return The six joints after the step, radians: the same joints when the solver finds no step.
        """
        seed = np.asarray(joints_rad, dtype=np.float64) * self._signs
        self._set_model_joints(np.clip(seed, self._low, self._high))
        self._task.set_target(mink.SE3.from_matrix(target))
        dt_iteration = dt_s / _ITERATIONS_PER_TICK
        try:
            for _ in range(_ITERATIONS_PER_TICK):
                velocity = mink.solve_ik(
                    self._configuration, [self._task], dt_iteration, "daqp", damping=1e-3, limits=self._limits
                )
                self._configuration.integrate_inplace(velocity, dt_iteration)
        except mink.NoSolutionFound:
            return np.asarray(joints_rad, dtype=np.float64).copy()
        solved = np.clip(self._configuration.q[self._qpos_index], self._low, self._high)
        max_step = self._max_speed * dt_s
        return np.clip(solved, seed - max_step, seed + max_step) * self._signs

    def _set_joints(self, joints_rad: npt.NDArray[np.float64]) -> None:
        self._set_model_joints(np.asarray(joints_rad, dtype=np.float64) * self._signs)

    def _set_model_joints(self, model_joints: npt.NDArray[np.float64]) -> None:
        q = self._configuration.q.copy()
        q[self._qpos_index] = model_joints
        self._configuration.update(q)


def urdf_model(urdf_text: str, flange_link: str, tcp_xyz_m: Sequence[float]) -> mujoco.MjModel:
    """A kinematic MuJoCo model of an arm from its URDF, with a `tcp` site on the flange.

    Meshes are not needed: visual and collision geometry is dropped, and fixed links are kept as bodies so the flange
    can carry the site.

    @param urdf_text The URDF's text.
    @param flange_link The link the gripper is mounted on.
    @param tcp_xyz_m The tool center point in the flange's frame, metres (measured on the real gripper).
    @return The compiled model.
    """
    text = re.sub(r"<(visual|collision)\b.*?</\1>", "", urdf_text, flags=re.DOTALL)
    compiler = (
        '<mujoco><compiler discardvisual="true" fusestatic="false" balanceinertia="true" '
        'boundmass="0.001" boundinertia="0.000001"/></mujoco>'
    )
    text = re.sub(r"(<robot\b[^>]*>)", rf"\1{compiler}", text, count=1)
    spec = mujoco.MjSpec.from_string(text)
    flange = spec.body(flange_link)
    if flange is None:
        raise ValueError(f"the URDF has no link {flange_link!r}")
    flange.add_site(name=TCP_SITE, pos=list(tcp_xyz_m))
    return spec.compile()
