# Simulation library E2E testing

The opt-in test in `library/tests/integration/test_simulation_training_e2e.py` checks the library path from a genuine simulation recording through policy training, checkpointing, OpenVINO export, and live Runtime inference in MuJoCo. It does not start Studio. The weekly workflow is `.github/workflows/simulation-library-e2e.yml`; expensive simulation runs are excluded from the usual PR tests.

## Scenarios and data

The scheduled scenario is `conveyor_sort`. Its pinned 50-episode [recording](https://huggingface.co/datasets/Albertvh97/conveyor-sort-sim-demos) is revision `6a949360761fee17089fbaaf8bc53f59715cb575`, with overview and wrist cameras and a 0.04 m/s belt speed. The test rejects a mismatched scene or incompatible dataset features. It scores **all completed conveyor episodes** using `correct / (correct + wrong + missed)`; at least one full scored episode must finish. The `single_pick_place` path is covered offline, but is not scheduled until genuine pick-place demonstrations have been published.

The library and MuJoCo plugin both use Runtime revision `91338224`. The weekly matrix selects `act`, `molmoact2`, `pi05`, `rldx1`, `smolvla`, and `xr0` for the conveyor. Cosmos3 is omitted until its required `embodiment` can be configured; the remaining policies have not all been validated live. The weekly matrix runs on the existing GPU runner with one job at a time.

## Running the test

Install the library with the `tests`, `policies`, `sim-e2e`, and appropriate hardware extras. Start the MuJoCo owner for the chosen scene, then set `SIM_E2E_DATASET_DIR`, `SIM_E2E_HTTP_URL`, `SIM_E2E_OWNER`, `SIM_E2E_POLICY`, `SIM_E2E_SCENE`, and `SIM_E2E_OUTPUT_DIR`. For the published conveyor dataset, set the owner's belt speed to 0.04 m/s and map the dataset gripper camera to the simulator wrist camera with `SIM_E2E_CAMERA_MAP='{"gripper":"wrist"}'`. Run `pytest tests/integration/test_simulation_training_e2e.py -k conveyor_sort` from `library/`.

The test constructs each selected policy using its own defaults via `get_policy(policy_name, source="physicalai")`. Training defaults to a two-step **smoke** budget; override it with `SIM_E2E_TRAIN_STEPS`. Inference stops after `SIM_E2E_ROLLOUT_SECONDS` of wall time (120 seconds by default). A smoke pass requires sent policy actions, observable joint motion, and—for conveyor—a scored episode. `result.json` includes every completed episode, its aggregate score, sent-action count, and maximum joint movement. Checkpoints and exported weights remain in `SIM_E2E_OUTPUT_DIR`.

Set `SIM_E2E_RECORD_VIDEO=true` (or the weekly workflow's manual `record_video` input) to record simulator cameras side by side during inference; this requires FFmpeg and is off by default. Set `SIM_E2E_MIN_SUCCESS` only after measuring a policy/scenario baseline. A 0/10 outcome can pass the wiring smoke test and does **not** establish task performance. Previous live ACT smoke runs used the earlier Runtime revision; the new pin has not been live-validated yet.
