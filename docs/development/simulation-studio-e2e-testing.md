# Simulation Studio E2E testing

The opt-in test in `application/backend/tests/integration/test_simulation_studio_e2e.py` exercises the Studio API and runtime WebSocket with a MuJoCo robot: import a genuine task recording, train and export a policy (or import an earlier export), load it for live inference, and read the scene's outcome. The separate [library E2E test](simulation-e2e-testing.md) produces the weekly exports used by inference-only Studio runs. These expensive tests are excluded from normal backend PR integration checks.

## Modes and current coverage

- `full` imports the pinned dataset through Studio endpoints, trains ACT through the jobs API, exports it, then checks policy loading and live control. It does **not** yet collect fresh Studio episodes. One CPU epoch over the conveyor dataset can take a long time: the train-job poll permits 70 hours and its workflow job permits 72 hours.
- `inference` imports the most recent successful, unexpired weekly library artifact for the selected policy and scenario. The workflow verifies checksums, policy, scene, and the **same pinned dataset revision** before loading it. This checks Studio inference, not PR training changes.

Both modes configure the project, robot, cameras, and environment through Studio APIs, follow the runtime WebSocket's model-loaded and policy-control states, and score completed simulator trials. A completed rollout or a 0/10 result is a wiring smoke test, not evidence of task proficiency; use `SIM_E2E_MIN_SUCCESS` only after establishing a baseline. The backend test's offline mocks cover `full` and `inference`; live ACT conveyor smoke runs used the previous Runtime pin, and the current Runtime revision has not been live-validated.

The conveyor dataset is pinned to `Albertvh97/conveyor-sort-sim-demos` revision `6a949360761fee17089fbaaf8bc53f59715cb575` and its belt is set to the recording's 0.04 m/s. Studio and the MuJoCo plugin use Runtime revision `91338224`. `single_pick_place` needs genuine published demonstrations before a full live run. Cosmos3 requires embodiment configuration and Studio model-import support, so neither matrix includes it.

## On-demand PR workflow

`.github/workflows/simulation-studio-e2e.yml` accepts manual inputs or an exact maintainer PR comment such as:

```text
/e2e studio policy=act scenario=conveyor_sort mode=inference
/e2e studio policy=act scenario=conveyor_sort mode=full
```

The authorizing job verifies write-or-higher permission, PR status and head SHA, an allowlisted policy/scenario/mode, and—for inference—a successful weekly artifact. The test job rechecks the PR SHA, runs untrusted PR code only on a **disposable isolated** self-hosted runner, uses read-only permissions and no checkout credentials, and cleans up services. The workflow definition for `issue_comment` must exist on the default branch before PR comments can trigger it.

**Do not enable this for fork PRs yet:** the `TEMP_EPHEMERAL_SIM` runner is not provisioned and `SIM_E2E_EPHEMERAL_RUNNER_READY` must stay false until the disposable runner and its network isolation are in place. The opt-in Studio workflow has not run in CI. It is not a required check for every PR. Publish a pick-place dataset and set its immutable repo/revision variables before requesting that scenario; do not substitute random-weight inference or call a smoke run a performance gate.
