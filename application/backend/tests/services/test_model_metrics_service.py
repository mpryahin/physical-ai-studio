# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock

from botocore.exceptions import ClientError

from schemas.remote_trainer import AwsBatchConnection
from services.aws_batch import aws_batch_trainer_url
from services.model_metrics_service import ModelMetricsService


async def test_batch_metrics_wait_for_s3_and_emit_only_new_complete_rows(monkeypatch) -> None:
    connection = AwsBatchConnection(
        region="us-east-1",
        bucket="job-bucket",
        studio_role_arn="arn:aws:iam::123:role/studio",
        targets={"g4dn.2xlarge": {"queue": "queue", "job_definition": "definition"}},
    )
    client = MagicMock()
    first = BytesIO(b"step,train/loss\n0,2.5\n1,")
    unchanged = BytesIO(b"step,train/loss\n0,2.5\n")
    second = BytesIO(b"step,train/loss\n0,2.5\n1,1.25\n")
    client.get_object.side_effect = [
        ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject"),
        {"Body": first},
        {"Body": unchanged},
        {"Body": second},
    ]
    session = MagicMock()
    session.client.return_value = client
    make_session = MagicMock(return_value=session)
    monkeypatch.setattr("services.aws_batch._make_session", make_session)
    monkeypatch.setattr("services.model_metrics_service.asyncio.sleep", AsyncMock())
    stream = ModelMetricsService.tail_batch_csv_file(aws_batch_trainer_url(connection), "studio-job-id")
    try:
        assert json.loads((await anext(stream)).data) == {"step": 0, "train_loss": 2.5}
        assert json.loads((await anext(stream)).data) == {"step": 1, "train_loss": 1.25}
    finally:
        await stream.aclose()

    make_session.assert_called_once_with(connection)
    client.get_object.assert_called_with(Bucket="job-bucket", Key="jobs/studio-job-id/metrics.csv")
    assert first.closed and unchanged.closed and second.closed
