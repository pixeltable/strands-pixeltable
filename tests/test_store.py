"""PixeltableMemoryStore against a real local catalog."""

from __future__ import annotations

import uuid

import pixeltable as pxt
import pytest
from strands.memory import AddMessagesContext, MemoryManager

from strands_pixeltable import PixeltableMemoryStore
from tests.embed import tiny_embed


def _root() -> str:
    return f"test_strands_{uuid.uuid4().hex[:8]}"


def _drop(table: str | None, directory: str) -> None:
    if table is not None:
        try:
            pxt.drop_table(table, force=True)
        except pxt.Error:
            pass
    try:
        pxt.drop_dir(directory, force=True)
    except pxt.Error:
        pass


@pytest.fixture
def root() -> str:
    name = _root()
    yield name
    _drop(None, name)


def _store(root: str, **kwargs: object) -> PixeltableMemoryStore:
    params: dict[str, object] = {"table_name": f"{root}.memory", "embedding": tiny_embed}
    params.update(kwargs)
    return PixeltableMemoryStore(**params)  # type: ignore[arg-type]


async def test_add_and_search_ranks_exact_text_first(root: str) -> None:
    store = _store(root)
    await store.add("alpha cats sit on mats", {"source": "user"})
    await store.add("unrelated boats and weather")

    found = await store.search("alpha cats sit on mats")
    assert found[0].content == "alpha cats sit on mats"
    assert found[0].metadata is not None
    assert found[0].metadata["kind"] == "note"
    assert found[0].metadata["metadata"] == {"source": "user"}
    assert found[0].metadata["score"] >= found[1].metadata["score"]  # type: ignore[index]
    assert store.table.count() == 2


async def test_search_limit_empty_query_and_duplicate_id(root: str) -> None:
    store = _store(root)
    await store.add("same text", {"memory_id": "fixed"})
    await store.add("same text")
    assert await store.search("   ") == []
    assert len(await store.search("same text", {"max_search_results": 1})) == 1
    with pytest.raises(ValueError, match="max_search_results"):
        await store.search("same text", {"max_search_results": 0})
    with pytest.raises(ValueError, match="already exists"):
        await store.add("other", {"memory_id": "fixed"})


async def test_add_messages_are_idempotent_on_tracking_id(root: str) -> None:
    store = _store(root)
    message = {"role": "user", "content": [{"text": "I like tea"}], "tracking_id": "track-1"}
    first = await store.add_messages([message], AddMessagesContext(sequence_numbers=[4]))
    second = await store.add_messages([message])
    assert first == ["msg:track-1"]
    assert second == []

    skipped = await store.add_messages(
        [
            {"role": "assistant", "content": [{"toolUse": {"name": "noop"}}]},
            {"role": "assistant", "content": [{"text": "noted"}]},
        ]
    )
    assert len(skipped) == 1
    rows = list(
        store.table.where(store.table.kind == "message").select(store.table.content, store.table.metadata).collect()
    )
    by_content = {row["content"]: row["metadata"] for row in rows}
    assert by_content["user: I like tea"]["sequence_number"] == 4
    assert "assistant: noted" in by_content
    assert store.table.count() == 2


async def test_messages_without_tracking_id_are_stored_again(root: str) -> None:
    store = _store(root)
    message = {"role": "user", "content": [{"text": "ok"}]}
    await store.add_messages([message])
    await store.add_messages([message])
    assert store.table.count() == 2


async def test_session_filter_and_blank_override(root: str) -> None:
    store = _store(root, session_id="s1")
    await store.add("alpha cats")
    await store.add("beta dogs", {"session_id": "s2"})

    # The default session keeps the s2 row out. Nearest-neighbor search still returns the s1 row.
    assert [entry.content for entry in await store.search("beta dogs")] == ["alpha cats"]
    found = await store.search("beta dogs", {"session_id": ""})
    assert [entry.content for entry in found] == ["beta dogs", "alpha cats"]
    scoped = await store.search("alpha cats", {"session_id": "s1"})
    assert [entry.content for entry in scoped] == ["alpha cats"]
    other = await store.search("alpha cats", {"session_id": "s2"})
    assert [entry.content for entry in other] == ["beta dogs"]


async def test_reopen_keeps_rows_and_does_not_reload_default_embedding(
    root: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(root)
    await store.add("alpha cats sit on mats")

    def fail() -> None:
        raise AssertionError("default embedding should not load when the index exists")

    monkeypatch.setattr("strands_pixeltable.store.sentence_transformer_embedding", fail)
    reopened = PixeltableMemoryStore(table_name=store.table_name, writable=False)
    found = await reopened.search("alpha cats sit on mats")
    assert found[0].content == "alpha cats sit on mats"
    with pytest.raises(ValueError, match="not writable"):
        await reopened.add("nope")


async def test_read_only_missing_table(root: str) -> None:
    store = PixeltableMemoryStore(table_name=f"{root}.missing", writable=False, embedding=tiny_embed)
    with pytest.raises(ValueError, match="read-only"):
        await store.search("anything")
    assert root not in pxt.list_dirs()


async def test_default_embedding_is_used_when_the_index_is_missing(root: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("strands_pixeltable.store.sentence_transformer_embedding", lambda: tiny_embed)
    store = PixeltableMemoryStore(table_name=f"{root}.memory")
    memory_id = await store.add("hello from the default embedder")
    found = await store.search("hello from the default embedder")
    assert found[0].metadata is not None
    assert found[0].metadata["memory_id"] == memory_id


async def test_existing_table_with_the_wrong_schema_is_rejected(root: str) -> None:
    pxt.create_dir(root, if_exists="ignore")
    pxt.create_table(f"{root}.memory", {"memory_id": pxt.Int, "content": pxt.String})
    store = _store(root)
    with pytest.raises(ValueError, match="memory_id"):
        _ = store.table


async def test_if_exists_error_and_replace(root: str) -> None:
    store = _store(root)
    await store.add("keep me")
    again = _store(root, if_exists="error")
    with pytest.raises(ValueError, match="already exists"):
        _ = again.table

    replaced = _store(root, if_exists="replace")
    assert replaced.table.count() == 0


async def test_view_is_rejected(root: str) -> None:
    store = _store(root)
    await store.add("a row")
    view_name = f"{root}.memory_view"
    pxt.create_view(view_name, store.table)
    view_store = PixeltableMemoryStore(table_name=view_name, embedding=tiny_embed)
    with pytest.raises(ValueError, match="view"):
        _ = view_store.table


async def test_memory_manager_search_sets_store_name(root: str) -> None:
    store = _store(root, name="notes")
    manager = MemoryManager(stores=[store], add_tool_config=True, injection=False)
    assert {tool.tool_name for tool in manager.tools} == {"search_memory", "add_memory"}
    await manager.add("alpha cats sit on mats")
    found = await manager.search("alpha cats sit on mats")
    assert found[0].store_name == "notes"
    assert found[0].content == "alpha cats sit on mats"


def test_extraction_true_is_accepted_without_a_model(root: str) -> None:
    store = _store(root, extraction=True)
    manager = MemoryManager(stores=[store], injection=False)
    assert manager is not None


def test_constructor_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError, match="table path"):
        PixeltableMemoryStore(table_name="bad..name", embedding=tiny_embed)
    with pytest.raises(ValueError, match="version handle"):
        PixeltableMemoryStore(table_name="mem:3", embedding=tiny_embed)
    with pytest.raises(ValueError, match="metric"):
        PixeltableMemoryStore(embedding=tiny_embed, metric="manhattan")  # type: ignore[arg-type]


async def test_blank_content_and_non_json_metadata(root: str) -> None:
    store = _store(root)
    with pytest.raises(ValueError, match="content"):
        await store.add("   ")
    with pytest.raises(ValueError, match="JSON-serializable"):
        await store.add("ok", {"blob": object()})
