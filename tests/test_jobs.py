from unittest.mock import MagicMock, patch

import pytest


def _mock_client():
    mock_doc_ref = MagicMock()
    mock_client = MagicMock()
    mock_client.collection.return_value.document.return_value = mock_doc_ref
    return mock_client, mock_doc_ref


def test_create_job_unconfigured_raises():
    from app.ingestion import jobs
    with patch("app.ingestion.jobs._get_client", return_value=None):
        with pytest.raises(RuntimeError):
            jobs.create_job(["a.txt"])


def test_get_job_unconfigured_raises():
    from app.ingestion import jobs
    with patch("app.ingestion.jobs._get_client", return_value=None):
        with pytest.raises(RuntimeError):
            jobs.get_job("job-1")


def test_create_job_writes_pending_status_with_files():
    from app.ingestion import jobs
    mock_client, mock_doc_ref = _mock_client()
    with patch("app.ingestion.jobs._get_client", return_value=mock_client):
        job_id = jobs.create_job(["a.txt", "b.txt"])

    assert job_id  # a non-empty generated id
    written = mock_doc_ref.set.call_args[0][0]
    assert written["status"] == "pending"
    assert written["files"] == ["a.txt", "b.txt"]
    assert written["expires_at"] > written["created_at"]  # TTL is a future timestamp


def test_get_job_returns_none_when_missing():
    from app.ingestion import jobs
    mock_client, mock_doc_ref = _mock_client()
    mock_doc_ref.get.return_value.exists = False
    with patch("app.ingestion.jobs._get_client", return_value=mock_client):
        assert jobs.get_job("nonexistent") is None


def test_get_job_returns_data_when_found():
    from app.ingestion import jobs
    mock_client, mock_doc_ref = _mock_client()
    mock_snap = MagicMock()
    mock_snap.exists = True
    mock_snap.to_dict.return_value = {"status": "done"}
    mock_doc_ref.get.return_value = mock_snap
    with patch("app.ingestion.jobs._get_client", return_value=mock_client):
        assert jobs.get_job("job-1") == {"status": "done"}


def test_process_job_success_marks_done():
    from app.ingestion import jobs
    mock_client, mock_doc_ref = _mock_client()
    with patch("app.ingestion.jobs._get_client", return_value=mock_client):
        with patch("app.ingestion.jobs.ingest.run", return_value={"added": ["a.txt"]}):
            jobs.process_job("job-1")

    statuses = [call.args[0]["status"] for call in mock_doc_ref.update.call_args_list]
    assert statuses == ["processing", "done"]


def test_process_job_failure_marks_failed_and_reraises():
    from app.ingestion import jobs
    mock_client, mock_doc_ref = _mock_client()
    with patch("app.ingestion.jobs._get_client", return_value=mock_client):
        with patch("app.ingestion.jobs.ingest.run", side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError):
                jobs.process_job("job-1")

    statuses = [call.args[0]["status"] for call in mock_doc_ref.update.call_args_list]
    assert statuses == ["processing", "failed"]
    last_update = mock_doc_ref.update.call_args_list[-1].args[0]
    # The recorded error is classified, NOT str(exc): this record is returned
    # verbatim by /jobs/{id} and rendered by ui.html, so the raw text reached
    # an anonymous browser. See app/ingestion/errors.py. The original is in
    # the logs, keyed by job id.
    assert "boom" not in last_update["error"]
    assert last_update["error_code"] == "internal"
    assert last_update["error"]


def test_enqueue_cloud_task_calls_cloud_tasks_client(monkeypatch):
    from app import config
    from app.ingestion import jobs
    monkeypatch.setattr(config, "GCP_PROJECT_ID", "test-project")
    monkeypatch.setattr(config, "GCP_LOCATION", "us-central1")
    monkeypatch.setattr(config, "CLOUD_TASKS_QUEUE", "ingest-queue")
    monkeypatch.setattr(config, "INGEST_TARGET_URL", "https://example.run.app")

    with patch("google.cloud.tasks_v2.CloudTasksClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.queue_path.return_value = (
            "projects/test-project/locations/us-central1/queues/ingest-queue"
        )
        mock_client_cls.return_value = mock_client

        jobs.enqueue_cloud_task("job-1")

    mock_client.create_task.assert_called_once()
    request = mock_client.create_task.call_args.kwargs["request"]
    assert request.parent == "projects/test-project/locations/us-central1/queues/ingest-queue"
    assert request.task.http_request.url == "https://example.run.app/internal/process-ingest-job"
