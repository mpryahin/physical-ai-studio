"""AWS Batch trainer health probe."""

from __future__ import annotations

import asyncio
import json
from time import perf_counter
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlencode, urlsplit

from loguru import logger

from schemas.remote_trainer import (
    AwsBatchConnection,
    AwsBatchProviderConfiguration,
    AwsBatchResourceConfiguration,
    HealthStatus,
    RemoteTrainerHealth,
)
from training.policy_catalog import PolicyCatalog

if TYPE_CHECKING:
    from datetime import datetime
    from uuid import UUID

_PROBE_TIMEOUT_S = 10.0
_MAX_CONFIGURATION_BYTES = 65536

POLICY_INSTANCE_TYPES = {
    "act": "g4dn.2xlarge",
    "pi05": "g6e.2xlarge",
    "smolvla": "g4dn.2xlarge",
    "molmoact2": "g7e.8xlarge",
}


def aws_batch_trainer_url(connection: AwsBatchConnection) -> str:
    """Encode resolved resources in a restart-stable trainer URL."""
    query = urlencode(
        {
            "studio_role_arn": connection.studio_role_arn,
            "targets": json.dumps({key: target.model_dump() for key, target in connection.targets.items()}),
            **({"configuration_uri": connection.configuration_uri} if connection.configuration_uri else {}),
        }
    )
    return f"aws-batch://{connection.region}/{connection.bucket}?{query}"


def aws_batch_connection_from_url(url: str) -> AwsBatchConnection:
    """Restore pinned Batch resources without consulting mutable trainer configuration."""
    location = urlsplit(url)
    if location.scheme != "aws-batch":
        raise ValueError("Expected an aws-batch trainer URL")
    query = parse_qs(location.query, strict_parsing=True)
    try:
        return AwsBatchConnection(
            region=location.netloc,
            bucket=location.path.removeprefix("/"),
            studio_role_arn=query["studio_role_arn"][0],
            targets=json.loads(query["targets"][0]),
            configuration_uri=query.get("configuration_uri", [None])[0],
        )
    except KeyError as exc:
        raise ValueError(f"Batch trainer URL is missing {exc.args[0]}") from exc


def batch_policy_catalog(connection: AwsBatchConnection) -> PolicyCatalog:
    """Declare capabilities of the provisioned CUDA trainer image."""
    from training.policy_catalog import declared_policy_catalog

    return declared_policy_catalog(
        {
            policy: ["torch", "openvino", "onnx", "executorch"] if policy == "act" else ["torch", "openvino"]
            for policy, instance_type in POLICY_INSTANCE_TYPES.items()
            if instance_type in connection.targets
        }
    )


def _make_session(connection: AwsBatchConnection) -> Any:
    """Assume the stack's Studio role with the ambient credential chain."""
    import boto3

    sts = boto3.client("sts", region_name=connection.region)
    creds = sts.assume_role(RoleArn=connection.studio_role_arn, RoleSessionName="physicalai-studio")["Credentials"]
    return boto3.Session(
        aws_access_key_id=creds["AccessKeyId"],
        aws_secret_access_key=creds["SecretAccessKey"],
        aws_session_token=creds["SessionToken"],
        region_name=connection.region,
    )


def resolve_aws_batch_configuration(configuration: AwsBatchProviderConfiguration) -> AwsBatchConnection:
    """Load and validate configuration using the ambient AWS credential chain."""
    import boto3

    location = urlsplit(configuration.configuration_uri)
    session = boto3.Session()
    response = session.client("s3").get_object(Bucket=location.netloc, Key=location.path[1:])
    body = response["Body"]
    try:
        payload = body.read(_MAX_CONFIGURATION_BYTES + 1)
    finally:
        body.close()
    if len(payload) > _MAX_CONFIGURATION_BYTES:
        raise ValueError("AWS configuration exceeds 64 KiB.")
    resources = AwsBatchResourceConfiguration.model_validate_json(payload)
    return AwsBatchConnection(**configuration.model_dump(), **resources.model_dump())


def _probe_sync(connection: AwsBatchConnection) -> tuple[HealthStatus, str | None]:
    from botocore.exceptions import BotoCoreError, ClientError

    try:
        session = _make_session(connection)
        batch = session.client("batch")
        queues = batch.describe_job_queues(jobQueues=[t.queue for t in connection.targets.values()])["jobQueues"]
    except (BotoCoreError, ClientError) as exc:
        code = getattr(exc, "response", {}).get("Error", {}).get("Code") if hasattr(exc, "response") else None
        logger.debug("AWS Batch probe failed: {}", exc)
        return "unreachable", code or "aws_unavailable"
    if len(queues) != len(connection.targets):
        return "degraded", "job_queue_missing"
    if any(q.get("state") != "ENABLED" or q.get("status") != "VALID" for q in queues):
        return "degraded", "job_queue_disabled"
    return "healthy", None


async def probe_aws_batch(
    remote_trainer_id: UUID, connection: AwsBatchConnection, checked_at: datetime
) -> RemoteTrainerHealth:
    """Check credentials and job queues without representing queues as physical GPUs."""
    started = perf_counter()
    try:
        status, reason_code = await asyncio.wait_for(asyncio.to_thread(_probe_sync, connection), _PROBE_TIMEOUT_S)
    except TimeoutError:
        status, reason_code = "unreachable", "timeout"
    return RemoteTrainerHealth(
        remote_trainer_id=remote_trainer_id,
        status=status,
        checked_at=checked_at,
        latency_ms=round((perf_counter() - started) * 1000),
        devices=[],
        storage=None,
        reason_code=reason_code,
    )
