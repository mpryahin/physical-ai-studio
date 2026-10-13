from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger
from physicalai.runtime import WorkerDiedError

from runtime.contract import (
    ErrorEvent,
    LoadModelCommand,
    ReturnHomeCommand,
    SetFollowerSourceCommand,
    SetInterpolationCommand,
    StartRecordingCommand,
    StartTaskCommand,
    StateData,
    StateEvent,
    StopTaskCommand,
)
from runtime.policy_loader import PolicyLoader, model_identity, requires_task

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from physicalai.capture.frame import Frame
    from physicalai.robot.interface import Robot, RobotObservation
    from physicalai.runtime import PolicySource

    from runtime.callbacks.recording import RecordingState
    from runtime.contract import CommandMailbox, EventSink, FollowerSource
    from runtime.interpolation import AdjustableLinearInterpolator
    from runtime.policy_loader import ModelIdentity, ObservationSnapshot

# Same duration and collision threshold as RobotRuntime's return_to_initial_state.
RETURN_HOME_DURATION_S = 2.5
RETURN_HOME_MAX_TRACKING_ERROR = 0.15


@dataclass
class _ReturnHome:
    start: np.ndarray
    target: np.ndarray
    travel: float
    steps: int
    last_command: np.ndarray
    step: int = 0


class StudioActionSource:
    """Select the action sent by RobotRuntime from Studio's current mode."""

    def __init__(  # noqa: PLR0913 — keyword-only collaborators
        self,
        *,
        follower: Robot,
        leader: Robot | None,
        mailbox: CommandMailbox,
        event_sink: EventSink,
        fps: float,
        camera_keys: Sequence[str] = (),
        models_dir: Path | None = None,
        interpolator: AdjustableLinearInterpolator | None = None,
    ) -> None:
        self._follower = follower
        self._leader = leader
        self._mailbox = mailbox
        self._event_sink = event_sink
        self._fps = fps
        self._interpolator = interpolator
        self._max_leader_failures = max(1, int(3 * fps))
        self._follower_source: FollowerSource = "hold"
        self._hold_target: np.ndarray | None = None
        self._home_target: np.ndarray | None = None
        self._return_home: _ReturnHome | None = None
        self._last_leader_action: np.ndarray | None = None
        self._last_leader_timestamp: float | None = None
        self._leader_failures = 0
        self._leader_reads_enabled = leader is not None
        self._failure_logged = False
        self._bus: object | None = None
        self._session_id: str | None = None
        self._policy: PolicySource | None = None
        self._policy_lock = threading.Lock()
        self._attached_generation = 0
        self._arm_generation = 0
        self._arming = False
        self._model_loaded = False
        self._loaded_identity: ModelIdentity | None = None
        self._task: str | None = None
        self._policy_error_emitted = False
        self._last_robot_state: RobotObservation | None = None
        self._last_camera_frames: Mapping[str, Frame] = {}
        self._loader = PolicyLoader(
            event_sink=event_sink,
            on_ready=self._set_policy,
            camera_keys=camera_keys,
            models_dir=models_dir,
        )
        self._recording: RecordingState | None = None

    @property
    def follower_source(self) -> FollowerSource:
        return self._follower_source

    def bind_recording(self, recording: RecordingState) -> None:
        """Attach session-owned recording flags so state events include them."""
        self._recording = recording

    def state_data(self) -> StateData:
        """Return the session state currently published to the browser."""
        recording = self._recording
        return StateData(
            connected=True,
            follower_source=self._follower_source,
            has_leader=self._leader_reads_enabled,
            model_loaded=self._model_loaded,
            task=self._task,
            dataset_loaded=None if recording is None else recording.dataset_loaded,
            is_recording=None if recording is None else recording.is_recording,
            episodes_recorded=None if recording is None else recording.episodes_recorded,
            returning_home=self._return_home is not None,
            interpolation_multiplier=None if self._interpolator is None else self._interpolator.requested_multiplier,
        )

    def connect(self, *, bus: object, session_id: str) -> None:
        """Connect the leader once and remember runtime context for policy loads."""
        self._bus = bus
        self._session_id = session_id
        if self._leader is not None and not self._leader.is_connected():
            self._leader.connect()
        if self._leader is not None and self._leader.joint_names != self._follower.joint_names:
            raise ValueError("Leader and follower joint names must match for teleoperation")

    def update(
        self,
        robot_state: RobotObservation,
        camera_frames: Mapping[str, Frame],
        step: int,
    ) -> np.ndarray:
        self._last_robot_state = robot_state
        self._last_camera_frames = camera_frames
        if self._home_target is None:
            self._home_target = np.array(robot_state.joint_positions, dtype=np.float32, copy=True)
        self._drain_commands(robot_state)
        if self._hold_target is None:
            self._hold_target = np.array(robot_state.joint_positions, dtype=np.float32, copy=True)

        leader_action = self._read_leader(robot_state) if self._leader_reads_enabled else self._last_leader_action
        policy_action = self._policy_action(robot_state, camera_frames, step)
        if self._return_home is not None:
            self._advance_return_home(robot_state)

        if self._follower_source == "teleop" and leader_action is not None:
            return leader_action
        if self._follower_source == "policy" and policy_action is not None:
            return policy_action
        return self._hold_target.copy()

    def disconnect(self) -> None:
        """Clear per-run state without stopping a loaded policy's execution worker."""
        self._bus = None
        self._session_id = None
        self._hold_target = None
        self._home_target = None
        self._return_home = None
        self._last_leader_action = None
        self._last_leader_timestamp = None
        self._leader_failures = 0
        self._leader_reads_enabled = self._leader is not None
        self._failure_logged = False
        self._last_robot_state = None
        self._last_camera_frames = {}

    def shutdown_policy(self) -> None:
        """Stop in-flight loads and the policy execution worker. Session teardown only."""
        self._loader.shutdown()
        with self._policy_lock:
            self._cancel_arming_locked()
            policy = self._policy
            self._policy = None
            self._model_loaded = False
            self._loaded_identity = None
        if policy is not None:
            policy.disconnect()

    def _set_policy(self, source: PolicySource, generation: int, identity: ModelIdentity | None = None) -> None:
        with self._policy_lock:
            if generation < self._attached_generation:
                stale = True
                previous = None
            else:
                stale = False
                self._cancel_arming_locked()
                previous = self._policy
                self._policy = source
                self._attached_generation = generation
                self._model_loaded = True
                self._loaded_identity = identity
                self._policy_error_emitted = False
        if stale:
            source.disconnect()
            return
        self._emit_state()
        if previous is not None and previous is not source:
            previous.disconnect()

    def _latest_observation(self) -> ObservationSnapshot:
        if self._last_robot_state is None:
            return None
        return self._last_robot_state, self._last_camera_frames

    def _policy_action(
        self,
        robot_state: RobotObservation,
        camera_frames: Mapping[str, Frame],
        step: int,
    ) -> np.ndarray | None:
        with self._policy_lock:
            policy = self._policy
            if policy is None or self._arming:
                return None
            try:
                return policy.update(robot_state, camera_frames, step)
            except WorkerDiedError:
                raise
            except Exception as exc:
                self._drop_policy_to_hold(robot_state, exc)
                return None

    def _drop_policy_to_hold(self, robot_state: RobotObservation, exc: Exception) -> None:
        self._hold_target = np.array(robot_state.joint_positions, dtype=np.float32, copy=True)
        changed = self._follower_source != "hold"
        self._follower_source = "hold"
        # A policy that raised is not a policy worth reusing, so stop matching
        # it: re-entering inference should rebuild rather than skip back onto
        # the one that just failed.
        self._loaded_identity = None
        if not self._policy_error_emitted:
            logger.exception("Policy update failed; switching to hold")
            self._policy_error_emitted = True
            self._event_sink.emit(
                ErrorEvent(
                    message=str(exc) or "Policy inference failed.",
                    error_code="policy_inference_failed",
                )
            )
        if changed:
            self._emit_state()

    def _drain_commands(self, robot_state: RobotObservation) -> None:
        for command in self._mailbox.drain():
            if isinstance(command, StartTaskCommand | StopTaskCommand | SetFollowerSourceCommand):
                # Any explicit mode change takes the arm away from a return in progress.
                self._cancel_return_home()
            if isinstance(command, LoadModelCommand):
                self._handle_load_model(command)
            elif isinstance(command, StartTaskCommand):
                self._handle_start_task(command)
            elif isinstance(command, StopTaskCommand):
                self._handle_stop_task(robot_state)
            elif isinstance(command, SetFollowerSourceCommand):
                self._handle_set_follower_source(command, robot_state)
            elif isinstance(command, StartRecordingCommand):
                self._handle_start_recording(command)
            elif isinstance(command, ReturnHomeCommand):
                self._handle_return_home(robot_state)
            elif isinstance(command, SetInterpolationCommand):
                self._handle_set_interpolation(command)

    def _handle_return_home(self, robot_state: RobotObservation) -> None:
        if self._home_target is None:
            return
        self._cancel_arming()
        self._follower_source = "hold"
        current = np.array(robot_state.joint_positions, dtype=np.float32, copy=True)
        travel = float(np.max(np.abs(self._home_target - current), initial=0.0))
        if travel == 0.0:
            self._return_home = None
            self._hold_target = self._home_target.copy()
        else:
            self._return_home = _ReturnHome(
                start=current,
                target=self._home_target.copy(),
                travel=travel,
                steps=max(int(RETURN_HOME_DURATION_S * self._fps), 1),
                last_command=current,
            )
            self._hold_target = current.copy()
            logger.info("Returning follower to its session start pose over {:.1f}s", RETURN_HOME_DURATION_S)
        self._emit_state()

    def _advance_return_home(self, robot_state: RobotObservation) -> None:
        """Step the hold target towards home, aborting if the arm stops following."""
        move = self._return_home
        if move is None:
            return
        measured = np.asarray(robot_state.joint_positions, dtype=np.float32)
        if move.step > 0:
            error = np.abs(measured - move.last_command) / move.travel
            worst = int(np.argmax(error))
            if error[worst] > RETURN_HOME_MAX_TRACKING_ERROR:
                self._abort_return_home(measured, joint_index=worst, error=float(error[worst]))
                return
        move.step += 1
        t = move.step / move.steps
        move.last_command = (move.start * (1.0 - t) + move.target * t).astype(np.float32)
        self._hold_target = move.last_command.copy()
        if move.step >= move.steps:
            self._return_home = None
            self._emit_state()

    def _abort_return_home(self, measured: np.ndarray, *, joint_index: int, error: float) -> None:
        self._return_home = None
        # Hold where the arm actually is so the servos stop pushing against whatever blocked it.
        self._hold_target = measured.copy()
        joint_names = list(self._follower.joint_names)
        joint = joint_names[joint_index] if joint_index < len(joint_names) else str(joint_index)
        message = (
            f"Return home stopped: joint '{joint}' is off by {error:.0%} of the move (possible collision). "
            "Holding the current pose."
        )
        logger.warning(message)
        self._event_sink.emit(ErrorEvent(message=message, error_code="return_home_aborted"))
        self._emit_state()

    def _cancel_return_home(self) -> None:
        if self._return_home is None:
            return
        self._return_home = None
        self._emit_state()

    def _handle_set_interpolation(self, command: SetInterpolationCommand) -> None:
        if self._interpolator is None:
            self._event_sink.emit(
                ErrorEvent(
                    message="This runtime session does not support interpolation.",
                    error_code="interpolation_unsupported",
                )
            )
            return
        self._interpolator.request_multiplier(command.multiplier)
        self._emit_state()

    def _handle_load_model(self, command: LoadModelCommand) -> None:
        if not command.force and self._loaded_identity == model_identity(command):
            # Rebuilding costs a compile, a warmup and a second copy of the
            # weights resident beside the running one, to arrive back where we
            # started. Return before touching anything: a policy already running
            # under this model must not be cancelled or dropped to hold, and the
            # client is past its readiness gate, so re-publishing model_loaded
            # False then True would only flash "Initializing" at someone already
            # watching a loaded model.
            logger.info("Model {} is already loaded; skipping the rebuild", command.model_id)
            return
        # Clear before the load, not after it fails: leaving the previous
        # identity in place would let a later request for that model match a
        # policy this load is about to replace.
        self._loaded_identity = None
        self._cancel_arming()
        self._model_loaded = False
        self._emit_state()
        if self._bus is None or self._session_id is None:
            self._event_sink.emit(
                ErrorEvent(
                    message="Cannot load a model before the runtime session is connected.",
                    error_code="model_load_failed",
                )
            )
            return
        self._loader.request(
            command,
            self._latest_observation,
            bus=self._bus,
            session_id=self._session_id,
        )

    def _handle_start_task(self, command: StartTaskCommand) -> None:
        if not self._arm_policy(task=command.task):
            return
        self._emit_state()

    def _handle_stop_task(self, robot_state: RobotObservation) -> None:
        self._cancel_arming()
        if self._follower_source == "hold":
            return
        self._follower_source = "hold"
        self._hold_target = np.array(robot_state.joint_positions, dtype=np.float32, copy=True)
        self._emit_state()

    def _handle_start_recording(self, command: StartRecordingCommand) -> None:
        if self._recording is None or not self._recording.start(command.task):
            self._event_sink.emit(
                ErrorEvent(
                    message="Load a dataset before starting a recording.",
                    error_code="dataset_not_loaded",
                )
            )
            return
        self._emit_state()

    def _handle_set_follower_source(self, command: SetFollowerSourceCommand, robot_state: RobotObservation) -> None:
        requested = command.follower_source
        if requested == "teleop" and self._leader is None:
            self._event_sink.emit(
                ErrorEvent(
                    message="Teleoperation requires a leader robot.",
                    error_code="leader_required",
                )
            )
            return
        if requested == "teleop" and not self._leader_reads_enabled:
            self._event_sink.emit(
                ErrorEvent(
                    message="The leader robot is not responding. Reconnect before resuming teleoperation.",
                    error_code="leader_connection_lost",
                )
            )
            return
        if requested == "policy":
            if not self._arm_policy(task=self._task):
                return
            self._emit_state()
            return
        self._cancel_arming()
        if requested == self._follower_source:
            return
        self._follower_source = requested
        if requested == "hold":
            self._hold_target = np.array(robot_state.joint_positions, dtype=np.float32, copy=True)
        self._emit_state()

    def _arm_policy(self, *, task: str | None) -> bool:
        """Reset the policy, then warm up off this thread. Return whether mode already switched."""
        with self._policy_lock:
            policy = self._policy
            if policy is None:
                self._event_sink.emit(
                    ErrorEvent(
                        message="No policy is loaded.",
                        error_code="policy_not_loaded",
                    )
                )
                return False
            try:
                if task is not None:
                    policy.set_task(task)
                policy.reset()
            except WorkerDiedError:
                raise
            except RuntimeError as exc:
                self._event_sink.emit(
                    ErrorEvent(
                        message=str(exc) or "Failed to arm the policy.",
                        error_code="policy_reset_failed",
                    )
                )
                return False
            self._task = task
            self._policy_error_emitted = False
            self._arm_generation += 1
            generation = self._arm_generation
            self._arming = True
        self._warn_if_prompt_missing(policy, task)
        snapshot = self._latest_observation()
        if snapshot is None:
            with self._policy_lock:
                if generation != self._arm_generation or self._policy is not policy:
                    return False
                self._arming = False
                self._follower_source = "policy"
            return True
        thread = threading.Thread(
            target=self._finish_arm,
            name=f"policy-arm-{generation}",
            args=(policy, generation, snapshot),
            daemon=True,
        )
        thread.start()
        return False

    @staticmethod
    def _warn_if_prompt_missing(policy: PolicySource, task: str | None) -> None:
        """Log when a language-conditioned policy starts with an empty prompt.

        An empty prompt is allowed (the model then runs on its prompt template
        alone), but it is rarely intended and otherwise invisible.
        """
        if task is not None and task.strip():
            return
        model = getattr(policy, "_model", None)
        if model is not None and requires_task(model):
            logger.warning("Starting a policy that expects a task prompt with an empty prompt")

    def _finish_arm(self, policy: PolicySource, generation: int, snapshot: ObservationSnapshot) -> None:
        """Warm the current observation, then switch into policy mode if still current."""
        if snapshot is None:
            return
        robot_state, camera_frames = snapshot
        try:
            policy.warmup(policy.to_model_input(robot_state, camera_frames))
        except Exception as exc:
            with self._policy_lock:
                if generation != self._arm_generation:
                    return
                self._arming = False
            logger.exception("Policy warmup failed; staying in hold")
            self._event_sink.emit(
                ErrorEvent(
                    message=str(exc) or "Failed to warm up the policy.",
                    error_code="policy_warmup_failed",
                )
            )
            return
        with self._policy_lock:
            if generation != self._arm_generation or self._policy is not policy:
                return
            self._arming = False
            self._follower_source = "policy"
        self._emit_state()

    def _cancel_arming(self) -> None:
        with self._policy_lock:
            self._cancel_arming_locked()

    def _cancel_arming_locked(self) -> None:
        self._arm_generation += 1
        self._arming = False

    def _read_leader(self, robot_state: RobotObservation) -> np.ndarray | None:
        if self._leader is None:
            return None
        try:
            observation = self._leader.get_observation()
            if self._last_leader_timestamp is not None and observation.timestamp <= self._last_leader_timestamp:
                raise ConnectionError("Leader observation did not advance")
            action = np.array(observation.joint_positions, dtype=np.float32, copy=True)
            if action.shape != robot_state.joint_positions.shape:
                raise ValueError(
                    f"Leader action shape {action.shape} does not match "
                    f"follower shape {robot_state.joint_positions.shape}"
                )
        except Exception as exc:
            self._leader_failures += 1
            if not self._failure_logged:
                logger.warning("Leader observation failed; using the last safe action: {}", exc)
                self._failure_logged = True
            if self._leader_failures >= self._max_leader_failures:
                self._disable_failed_leader(robot_state)
            return self._last_leader_action if self._last_leader_action is not None else self._hold_target

        self._leader_failures = 0
        self._failure_logged = False
        self._last_leader_timestamp = observation.timestamp
        self._last_leader_action = action
        return action

    def _disable_failed_leader(self, robot_state: RobotObservation) -> None:
        self._leader_reads_enabled = False
        self._hold_target = np.array(robot_state.joint_positions, dtype=np.float32, copy=True)
        changed = self._follower_source != "hold"
        self._follower_source = "hold"
        logger.error("Leader failed for {} consecutive ticks; switching to hold", self._leader_failures)
        if changed:
            self._emit_state()
        self._event_sink.emit(
            ErrorEvent(
                message="The leader robot stopped responding. The follower switched to hold.",
                error_code="leader_connection_lost",
            )
        )

    def _emit_state(self) -> None:
        self._event_sink.emit(StateEvent(data=self.state_data()))
