from unittest.mock import MagicMock, patch


def test_add_to_history_noop_without_session_id():
    # No session_id -- must never touch Firestore at all.
    from app.retrieval import memory
    with patch("app.retrieval.memory._get_client") as mock_get_client:
        memory.add_to_history(None, "question", "answer")
        mock_get_client.assert_not_called()


def test_contextualize_question_noop_without_session_id():
    from app.retrieval import memory
    with patch("app.retrieval.memory._get_client") as mock_get_client:
        result = memory.contextualize_question(None, "What is X?")
        assert result == "What is X?"
        mock_get_client.assert_not_called()


def test_add_to_history_fails_open_on_firestore_error():
    # A transient Firestore failure must not propagate -- the caller
    # already has a usable answer to return.
    from app.retrieval import memory
    with patch("app.retrieval.memory._get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.collection.side_effect = RuntimeError("Firestore unavailable")
        mock_get_client.return_value = mock_client

        memory.add_to_history("session-1", "question", "answer")  # must not raise


def test_contextualize_question_fails_open_on_firestore_error():
    from app.retrieval import memory
    with patch("app.retrieval.memory._get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.collection.side_effect = RuntimeError("Firestore unavailable")
        mock_get_client.return_value = mock_client

        result = memory.contextualize_question("session-1", "What is X?")
        assert result == "What is X?"


def test_contextualize_question_unconfigured_returns_question_unchanged():
    # _get_client() returns None when Firestore isn't configured at all.
    from app.retrieval import memory
    with patch("app.retrieval.memory._get_client", return_value=None):
        result = memory.contextualize_question("session-1", "What is X?")
        assert result == "What is X?"


def test_history_capped_at_five_turns():
    from app.retrieval import memory
    existing_turns = [{"question": f"q{i}", "answer": f"a{i}"} for i in range(5)]

    mock_snap = MagicMock()
    mock_snap.exists = True
    mock_snap.to_dict.return_value = {"turns": existing_turns}

    mock_doc_ref = MagicMock()
    mock_doc_ref.get.return_value = mock_snap

    mock_client = MagicMock()
    mock_client.collection.return_value.document.return_value = mock_doc_ref

    # The write goes through the transaction, not doc_ref.set -- see
    # memory._apply_turn. Driving it directly keeps this test about the
    # capping rule rather than about Firestore's transaction protocol.
    txn = MagicMock()
    memory._apply_turn(txn, mock_doc_ref, "new question", "new answer")

    written = txn.set.call_args[0][1]
    assert len(written["turns"]) == 5
    assert written["turns"][-1] == {"question": "new question", "answer": "new answer"}
    assert written["turns"][0]["question"] == "q1"  # oldest (q0) evicted
    # Read must join the transaction, or the append races exactly as before.
    mock_doc_ref.get.assert_called_once_with(transaction=txn)


def test_add_to_history_commits_through_a_transaction():
    """add_to_history swallows every exception, so a transaction that never
    commits would be indistinguishable from success. Assert the wrapper
    actually runs the callback and writes."""
    from unittest.mock import ANY

    from app.retrieval import memory

    snap = MagicMock()
    snap.exists = False
    doc_ref = MagicMock()
    doc_ref.get.return_value = snap

    txn = MagicMock()
    client = MagicMock()
    client.collection.return_value.document.return_value = doc_ref
    client.transaction.return_value = txn

    # Stand in for firestore.transactional: return a callable that invokes
    # the wrapped function, which is what the real decorator does once the
    # transaction is open.
    fake_firestore = MagicMock()
    fake_firestore.transactional = lambda fn: fn

    with patch("app.retrieval.memory._get_client", return_value=client), \
         patch.dict("sys.modules", {"google.cloud.firestore": fake_firestore}), \
         patch("google.cloud.firestore", fake_firestore, create=True):
        memory.add_to_history("session-1", "q", "a")

    txn.set.assert_called_once_with(doc_ref, ANY)
    written = txn.set.call_args[0][1]
    assert written["turns"] == [{"question": "q", "answer": "a"}]


def test_contextualize_question_rewrites_with_history():
    from app.retrieval import memory
    mock_snap = MagicMock()
    mock_snap.exists = True
    mock_snap.to_dict.return_value = {
        "turns": [{"question": "What is the refund policy?", "answer": "30 days."}]
    }
    mock_doc_ref = MagicMock()
    mock_doc_ref.get.return_value = mock_snap
    mock_client = MagicMock()
    mock_client.collection.return_value.document.return_value = mock_doc_ref

    with patch("app.retrieval.memory._get_client", return_value=mock_client):
        with patch("app.retrieval.memory.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_response = MagicMock()
            mock_response.content = "How long is the refund policy window?"
            mock_llm.invoke.return_value = mock_response
            mock_get_llm.return_value = mock_llm

            result = memory.contextualize_question("session-1", "How long is it?")

    assert result == "How long is the refund policy window?"
