import sys
from unittest.mock import MagicMock

# Mock dependencies before importing the code under test
mock_milvus = MagicMock()
sys.modules["pymilvus"] = mock_milvus
sys.modules["langchain_openai"] = MagicMock()
sys.modules["prometheus_client"] = MagicMock()

import pytest  # noqa: E402
import os  # noqa: E402
from unittest.mock import patch  # noqa: E402
from src.ingest import worker  # noqa: E402
from src.config import Config  # noqa: E402


@pytest.fixture
def mock_env():
    # Ensure RUNNING starts True for tests
    worker.RUNNING = True
    with patch.dict(
        os.environ, {"REDIS_URL": "redis://localhost:6379", "OPENAI_API_KEY": "test"}
    ):
        yield


@patch("src.ingest.worker.signal.signal")
@patch("src.ingest.worker._redis")
@patch("src.ingest.worker.insert_rows")
@patch("src.ingest.worker.OpenAIEmbeddings")
@patch("src.ingest.worker._ensure_group")
def test_worker_pel_recovery(
    mock_ensure, mock_embed_cls, mock_insert, mock_redis_func, mock_signal, mock_env
):
    r = MagicMock()
    # Important: sismember must return False to allow processing
    r.sismember.return_value = False
    mock_redis_func.return_value = r

    mock_embed = MagicMock()
    mock_embed.embed_query.return_value = [0.1] * 3072
    mock_embed_cls.return_value = mock_embed

    calls = []

    def xreadgroup_side_effect(*args, **kwargs):
        calls.append(args)
        # Iteration 1: pending
        if len(calls) == 1:
            return [
                (
                    "ingest:chunks",
                    [("msg_pending", {"text": "pending doc", "source": "p.txt"})],
                )
            ]
        # Iteration 2: pending (None)
        if len(calls) == 2:
            return None
        # Iteration 2: new msgs (">")
        if len(calls) == 3:
            return [
                ("ingest:chunks", [("msg_new", {"text": "new doc", "source": "n.txt"})])
            ]
        # Finish
        worker.RUNNING = False
        return None

    r.xreadgroup.side_effect = xreadgroup_side_effect

    worker.run_worker()

    # Verify both types of reads
    r.xreadgroup.assert_any_call(
        Config.INGEST_GROUP,
        f"c-{os.getpid()}",
        streams={Config.INGEST_STREAM: "0"},
        count=Config.INGEST_CONCURRENCY,
    )

    # Verify 2 insertions
    assert mock_insert.call_count >= 2


@patch("src.ingest.worker.signal.signal")
@patch("src.ingest.worker._redis")
@patch("src.ingest.worker.insert_rows")
@patch("src.ingest.worker.OpenAIEmbeddings")
@patch("src.ingest.worker._ensure_group")
def test_worker_batch_fallback(
    mock_ensure, mock_embed_cls, mock_insert, mock_redis_func, mock_signal, mock_env
):
    r = MagicMock()
    r.sismember.return_value = False
    mock_redis_func.return_value = r

    mock_embed = MagicMock()
    mock_embed.embed_query.return_value = [0.1] * 3072
    mock_embed_cls.return_value = mock_embed

    calls = []

    def xreadgroup_side_effect(*args, **kwargs):
        calls.append(args)
        if len(calls) == 1:
            return None  # Pending
        if len(calls) == 2:
            return [
                (
                    "ingest:chunks",
                    [
                        ("msg1", {"text": "good doc", "source": "g.txt"}),
                        ("msg2", {"text": "bad doc", "source": "b.txt"}),
                    ],
                )
            ]  # New
        worker.RUNNING = False
        return None

    r.xreadgroup.side_effect = xreadgroup_side_effect

    def insert_side_effect(rows, partition=None):
        if len(rows) > 1:
            raise RuntimeError("Batch fail")
        if rows[0][2] == "bad doc":
            raise ValueError("Individual fail")
        return None

    mock_insert.side_effect = insert_side_effect

    worker.run_worker()

    r.xack.assert_any_call(Config.INGEST_STREAM, Config.INGEST_GROUP, "msg1")
    # Verify xadd for the failed item - order of fields in dict doesn't matter for assert_any_call
    # but we'll use the exact match if possible.
    called_args = [call.args for call in r.xadd.call_args_list]
    assert any(
        args[0] == Config.INGEST_STREAM and args[1].get("text") == "bad doc"
        for args in called_args
    )
    r.xack.assert_any_call(Config.INGEST_STREAM, Config.INGEST_GROUP, "msg2")
