"""Opt-in Studio HTTP/WebSocket train-or-import -> live MuJoCo inference.

Requires a running, isolated Studio backend and simulation owner. The dataset
archive and inference-only export are provided by the dedicated Actions lane.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Self
from uuid import UUID, uuid4

import httpx
import pytest
from websockets.sync.client import connect

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _response(response: httpx.Response) -> dict:
    response.raise_for_status()
    return response.json()


def _job(client: httpx.Client, job_id: str, *, step: str | None = None, timeout_s: float = 900) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        job = _response(client.get(f"/api/jobs/{job_id}"))
        if job["status"] in {"failed", "canceled"}:
            pytest.fail(f"Studio job failed: {job}")
        if step is not None and job["payload"]["step"] == step:
            return job
        if step is None and job["status"] == "completed":
            return job
        time.sleep(1)
    raise AssertionError(f"Studio job timed out: {job_id}")


def _import_export(export_dir: Path, project_id: UUID, dataset_id: UUID) -> UUID:
    """Seed the weekly export through Studio's existing internal import service."""
    from db import get_async_db_session_ctx
    from schemas import Model, TrainJob
    from schemas.dataset import Dataset
    from services.dataset_service import DatasetService
    from services.job_service import JobService
    from services.model_import_service import ModelImportService
    from services.model_service import ModelService

    async def get_dataset(identifier: UUID) -> Dataset:
        async with get_async_db_session_ctx() as session:
            return await DatasetService(session).get_dataset_by_id(identifier)

    async def persist(job: TrainJob, model: Model) -> Model:
        async with get_async_db_session_ctx() as session:
            saved = await JobService(session).create_job(job)
            return await ModelService(session).create_model(model.model_copy(update={"train_job_id": saved.id}))

    async def run() -> UUID:
        service = ModelImportService(get_dataset=get_dataset, persist_import=persist)
        # An export-only import expects exports/<backend>/manifest.json.
        with TemporaryDirectory(prefix="studio-sim-import-") as tmp:
            source = Path(tmp) / "studio-import"
            shutil.copytree(export_dir, source / "openvino")
            model = await service.import_model_directory(
                source_dir=source, project_id=project_id, dataset_id=dataset_id, model_name="Weekly simulation export"
            )
            return model.id

    return asyncio.run(run())


def _wait_state(ws, **expected: object) -> None:
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        message = json.loads(ws.recv(timeout=max(1, deadline - time.monotonic())))
        if message["event"] == "error":
            pytest.fail(f"Studio runtime error: {message}")
        if message["event"] == "state" and all(message["data"].get(k) == v for k, v in expected.items()):
            return
    pytest.fail(f"Studio runtime state not reached: {expected}")


def _wait_policy_action(ws) -> None:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        event = json.loads(ws.recv(timeout=max(1, deadline - time.monotonic())))
        if event["event"] == "error":
            pytest.fail(f"Studio runtime error: {event}")
        if event["event"] == "observation" and event.get("actions"):
            return
    pytest.fail("Studio did not send a policy action")


def test_runtime_state_gate() -> None:
    """An observation event cannot be mistaken for a loaded policy state."""

    class Socket:
        def __init__(self, messages: list[dict]) -> None:
            self.messages = iter(messages)

        def recv(self, *, timeout: float) -> str:
            return json.dumps(next(self.messages))

    ws = Socket([{"event": "observation", "data": {}}, {"event": "state", "data": {"model_loaded": True}}])
    _wait_state(ws, model_loaded=True)
    _wait_policy_action(Socket([{"event": "observation", "actions": {"gripper.pos": 0.0}}]))
    with pytest.raises(pytest.fail.Exception, match="Studio runtime error"):
        _wait_state(Socket([{"event": "error", "message": "bad model"}]), model_loaded=True)


def test_weekly_export_import_can_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Importing the same export twice must not reuse a stale staging directory."""
    from types import SimpleNamespace

    from services.model_import_service import ModelImportService

    export_dir = tmp_path / "export"
    export_dir.mkdir()
    (export_dir / "manifest.json").write_text("{}")
    model_id = uuid4()
    sources: list[Path] = []

    async def import_model(self: ModelImportService, *, source_dir: Path, **kwargs: object) -> SimpleNamespace:
        assert (source_dir / "openvino/manifest.json").is_file()
        sources.append(source_dir)
        return SimpleNamespace(id=model_id)

    monkeypatch.setattr(ModelImportService, "import_model_directory", import_model)
    for _ in range(2):
        assert _import_export(export_dir, uuid4(), uuid4()) == model_id
    assert sources[0] != sources[1]
    assert all(not source.exists() for source in sources)


@pytest.mark.parametrize("mode", ("inference", "full"))
def test_studio_orchestration_with_mock_endpoints(  # noqa: C901 - two end-to-end branches in one check
    mode: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Exercise both API branches and the WS command sequence without hardware or training."""
    import sys

    module = sys.modules[__name__]
    model_id, dataset_id = uuid4(), uuid4()
    calls: list[tuple[str, str, dict | None]] = []
    import_polls = 0
    sim_reads = 0

    class Client:
        def __init__(self, *, base_url: str, **kwargs: object) -> None:
            self.sim = base_url.endswith("8080")

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def request(self, method: str, path: str, **kwargs: object) -> httpx.Response:
            nonlocal import_polls, sim_reads
            payload = kwargs.get("json")
            calls.append((method, path, payload if isinstance(payload, dict) else None))
            body: dict | list[dict]
            if self.sim:
                if path == "/health":
                    sim_reads += 1
                    body = {
                        "scene": "conveyor_sort",
                        "autopilot": {"mode": "off"},
                        "episode": {
                            "episode_count": int(sim_reads > 2),
                            "last_episode": {"correct": 7, "wrong": 2, "missed": 1},
                        },
                    }
                else:
                    body = {"status": "queued"}
            elif path.endswith("datasets:prepare"):
                body = {"id": "import-job"}
            elif path == "/api/jobs/import-job":
                import_polls += 1
                body = {
                    "status": "running" if import_polls == 1 else "completed",
                    "payload": {
                        "step": "awaiting_user_review" if import_polls == 1 else "completed",
                        "result_dataset_id": str(dataset_id),
                    },
                }
            elif path == "/api/jobs:train":
                body = {"id": "train-job"}
            elif path == "/api/jobs/train-job":
                body = {"status": "completed", "payload": {}}
            elif path.endswith("/models"):
                body = [{"id": str(model_id)}]
            elif path.endswith("/episodes"):
                body = [{"episode_index": 0}]
            elif path.startswith("/api/models/"):
                body = {"exports": [{"type": "openvino"}]}
            else:
                body = {}
            return httpx.Response(200, json=body, request=httpx.Request(method, f"http://localhost{path}"))

        def get(self, path: str, **kwargs: object) -> httpx.Response:
            return self.request("GET", path, **kwargs)

        def post(self, path: str, **kwargs: object) -> httpx.Response:
            return self.request("POST", path, **kwargs)

        def put(self, path: str, **kwargs: object) -> httpx.Response:
            return self.request("PUT", path, **kwargs)

    class Socket:
        def __init__(self) -> None:
            self.sent: list[dict] = []
            self.events = iter(
                [
                    {"event": "state", "data": {"connected": True}},
                    {"event": "state", "data": {"model_loaded": True}},
                    {"event": "state", "data": {"follower_source": "policy"}},
                    {"event": "observation", "actions": {"gripper.pos": 0.0}},
                ]
            )

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def send(self, message: str) -> None:
            self.sent.append(json.loads(message))

        def recv(self, *, timeout: float) -> str:
            return json.dumps(next(self.events))

    socket = Socket()
    monkeypatch.setattr(module.httpx, "Client", Client)
    monkeypatch.setattr(module, "connect", lambda *args, **kwargs: socket)
    monkeypatch.setattr(module, "_import_export", lambda *args: model_id)
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    monkeypatch.setenv("SIM_E2E_SCENE", "conveyor_sort")
    monkeypatch.setenv("SIM_E2E_MODE", mode)
    monkeypatch.setenv("SIM_E2E_POLICY", "act")
    monkeypatch.setenv("SIM_E2E_OWNER", "sim-test")
    monkeypatch.setenv("SIM_E2E_HTTP_URL", "http://localhost:8080")
    monkeypatch.setenv("SIM_E2E_STUDIO_URL", "http://localhost:7860")
    monkeypatch.setenv("SIM_E2E_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("SIM_E2E_EXPORT_DIR", str(tmp_path / "export"))
    archive = tmp_path / "dataset.zip"
    archive.write_bytes(b"mock archive; no backend receives this")
    monkeypatch.setenv("SIM_E2E_DATASET_ARCHIVE", str(archive))

    test_studio_simulation_inference("conveyor_sort")

    events = [message["event"] for message in socket.sent if "event" in message]
    assert events == ["load_model", "set_follower_source", "disconnect"]
    assert socket.sent[1]["data"]["inference_device"] == {"backend": "openvino", "device": "CPU"}
    assert (("POST", "/api/jobs:train") in [(method, path) for method, path, _ in calls]) == (mode == "full")
    assert json.loads((tmp_path / "studio-result.json").read_text())["score"] == 0.7


@pytest.mark.parametrize("scene", ("conveyor_sort", "single_pick_place"))
def test_studio_simulation_inference(scene: str) -> None:
    """Exercise Studio endpoints and runtime WS against a real plugin owner."""
    selected = os.environ["SIM_E2E_SCENE"]
    if scene != selected:
        pytest.skip(f"Selected scenario is {selected}")
    mode = os.environ["SIM_E2E_MODE"]
    assert mode in {"inference", "full"}
    policy = os.environ["SIM_E2E_POLICY"]
    owner = os.environ["SIM_E2E_OWNER"]
    sim_url = os.environ["SIM_E2E_HTTP_URL"].rstrip("/")
    studio_url = os.environ["SIM_E2E_STUDIO_URL"].rstrip("/")
    with httpx.Client(base_url=sim_url, timeout=10) as sim, httpx.Client(base_url=studio_url, timeout=900) as client:
        health = _response(sim.get("/health"))
        assert health["scene"] == scene
        assert health["autopilot"]["mode"] == "off", "Disable simulator autopilot before policy evaluation"
        project_id, environment_id, robot_id, dataset_id = (uuid4() for _ in range(4))
        _response(client.post("/api/projects", json={"id": str(project_id), "name": "Simulation E2E"}))
        robots = f"/api/projects/{project_id}/robots"
        _response(
            client.post(
                robots,
                json={
                    "id": str(robot_id),
                    "type": "MuJoCo_SO101_Follower",
                    "name": "Sim follower",
                    "payload": {"name": owner},
                },
            )
        )
        camera_ids = []
        camera_map = json.loads(os.environ.get("SIM_E2E_CAMERA_MAP", "{}"))
        for key in os.environ.get("SIM_E2E_CAMERAS", "overview,gripper").split(","):
            camera_id = uuid4()
            camera_ids.append(camera_id)
            stream = camera_map.get(key, key)
            url = f"{sim_url}/cameras/{stream}/mjpeg"
            _response(
                client.post(
                    f"/api/projects/{project_id}/cameras",
                    json={
                        "id": str(camera_id),
                        "driver": "ipcam",
                        "name": key,
                        "fingerprint": {"url": url},
                        "hardware_name": f"MuJoCo {key}",
                        "payload": {"url": url, "fps": 30},
                    },
                )
            )
        _response(
            client.post(
                f"/api/projects/{project_id}/environments",
                json={
                    "id": str(environment_id),
                    "name": "MuJoCo",
                    "robots": [{"robot_id": str(robot_id), "tele_operator": {"type": "none"}}],
                    "cameras": [{"camera_id": str(c)} for c in camera_ids],
                },
            )
        )
        if mode == "full":
            archive = Path(os.environ["SIM_E2E_DATASET_ARCHIVE"])
            prepared = _response(
                client.post(
                    f"/api/projects/{project_id}/imports/datasets:prepare",
                    data={"format_hint": "auto", "dataset_name": "Simulation demonstrations"},
                )
            )
            import_id = prepared["id"]
            with archive.open("rb") as stream:
                _response(
                    client.put(
                        f"/api/projects/{project_id}/imports/datasets/{import_id}:upload",
                        files={"archive": ("dataset.zip", stream, "application/zip")},
                    )
                )
            _job(client, import_id, step="awaiting_user_review")
            _response(
                client.post(
                    f"/api/projects/{project_id}/imports/datasets/{import_id}:finalize",
                    json={"environment_id": str(environment_id), "default_task": "Complete the simulated task"},
                )
            )
            dataset_id = UUID(_job(client, import_id)["payload"]["result_dataset_id"])
            assert _response(client.get(f"/api/dataset/{dataset_id}/episodes"))
            train = _response(
                client.post(
                    "/api/jobs:train",
                    json={
                        "project_id": str(project_id),
                        "dataset_id": str(dataset_id),
                        "policy": policy,
                        "model_name": "Simulation E2E policy",
                        "max_epochs": int(os.environ.get("SIM_E2E_TRAIN_EPOCHS", "1")),
                        "batch_size": 2,
                        "num_workers": 0,
                        "val_split": 0.0,
                        "device": {"type": "cpu"},
                        "precision": "32-true",
                        "export_backends": ["openvino"],
                    },
                )
            )
            _job(client, train["id"], timeout_s=70 * 3600)
            models = _response(client.get(f"/api/projects/{project_id}/models"))
            assert len(models) == 1
            model_id = UUID(models[0]["id"])
        else:
            _response(
                client.post(
                    "/api/dataset",
                    json={
                        "id": str(dataset_id),
                        "name": "Weekly model dataset",
                        "default_task": "Complete the simulated task",
                        "project_id": str(project_id),
                        "environment_id": str(environment_id),
                    },
                )
            )
            model_id = _import_export(Path(os.environ["SIM_E2E_EXPORT_DIR"]), project_id, dataset_id)
        detail = _response(client.get(f"/api/models/{model_id}"))
        assert any(e["type"] == "openvino" for e in detail["exports"])

        ws_url = studio_url.replace("http", "ws", 1) + f"/api/projects/{project_id}/runtime/ws"
        with connect(ws_url, open_timeout=30, close_timeout=5, ping_interval=None) as ws, ExitStack() as cleanup:
            cleanup.callback(ws.send, json.dumps({"event": "disconnect"}))
            ws.send(json.dumps({"follower_id": str(robot_id), "camera_ids": [str(c) for c in camera_ids]}))
            _wait_state(ws, connected=True)
            ws.send(
                json.dumps(
                    {
                        "event": "load_model",
                        "data": {
                            "model_id": str(model_id),
                            "inference_device": {"backend": "openvino", "device": "CPU"},
                        },
                    }
                )
            )
            _wait_state(ws, model_loaded=True)
            _response(sim.post("/reset"))
            time.sleep(0.5)
            before = _response(sim.get("/health"))["episode"]["episode_count"]
            ws.send(json.dumps({"event": "set_follower_source", "data": {"follower_source": "policy"}}))
            _wait_state(ws, follower_source="policy")
            _wait_policy_action(ws)
            deadline = time.monotonic() + float(os.environ.get("SIM_E2E_ROLLOUT_SECONDS", "120"))
            while time.monotonic() < deadline:
                time.sleep(1)
                episode = _response(sim.get("/health"))["episode"]
                if episode["episode_count"] > before:
                    break
            if scene == "conveyor_sort":
                assert episode["episode_count"] > before, "Policy did not finish a conveyor episode"
                result = episode["last_episode"]
                assert sum(result.values()) > 0
                score = result["correct"] / sum(result.values())
            else:
                score = float(episode["episode_count"] > before)
            output = Path(os.environ["SIM_E2E_OUTPUT_DIR"])
            output.mkdir(parents=True, exist_ok=True)
            (output / "studio-result.json").write_text(
                json.dumps(
                    {
                        "mode": mode,
                        "policy": policy,
                        "scene": scene,
                        "score": score,
                        "model_id": str(model_id),
                        "pr_sha": os.environ.get("SIM_E2E_PR_SHA"),
                        "weekly_run": os.environ.get("SIM_E2E_WEEKLY_RUN"),
                        "weekly_sha": os.environ.get("SIM_E2E_WEEKLY_SHA"),
                    }
                )
            )
            assert score >= float(os.environ.get("SIM_E2E_MIN_SUCCESS", "0"))
