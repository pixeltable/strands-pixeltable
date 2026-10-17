"""Read-only catalog tools and remember/recall."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pixeltable as pxt
import pytest
from PIL import Image

from strands_pixeltable import PixeltableMemoryStore, pixeltable_tools
from tests.embed import tiny_embed


def _root() -> str:
    return f"test_tools_{uuid.uuid4().hex[:8]}"


def _drop(directory: str) -> None:
    try:
        pxt.drop_dir(directory, force=True)
    except pxt.Error:
        pass


@pytest.fixture
def catalog() -> str:
    root = _root()
    pxt.create_dir(root, if_exists="ignore")
    docs = pxt.create_table(
        f"{root}.docs",
        {
            "topic": pxt.String,
            "text": pxt.String,
            "status": pxt.String,
            "pos": pxt.Int,
            "active": pxt.Bool,
        },
    )
    secret = pxt.create_table(f"{root}.secret", {"text": pxt.String})
    docs.insert(
        [
            {
                "topic": "expenses",
                "text": "meals are reimbursed up to 60 EUR",
                "status": "open",
                "pos": 1,
                "active": True,
            },
            {"topic": "travel", "text": "trains need a receipt", "status": "closed", "pos": 2, "active": False},
            {
                "topic": "meals",
                "text": "meals during travel are reimbursed up to 60 EUR",
                "status": "open",
                "pos": 3,
                "active": True,
            },
        ]
    )
    secret.insert([{"text": "do not show this"}])
    docs.add_embedding_index("text", string_embed=tiny_embed, if_exists="ignore")
    yield root
    _drop(root)


def _tool(tools: list, name: str):
    return next(tool for tool in tools if tool.tool_name == name)


async def _run(tool, payload: dict) -> dict:
    last = None
    async for event in tool.stream({"toolUseId": "t", "name": tool.tool_name, "input": payload}, {}):
        last = event
    assert last is not None
    return last["tool_result"]


async def test_allowlist_query_and_similarity(catalog: str) -> None:
    tools = pixeltable_tools([catalog])
    names = {tool.tool_name for tool in tools}
    assert names == {"list_tables", "describe_table", "query_table", "similarity_search"}
    assert "remember" not in names

    listed = _tool(tools, "list_tables")()
    assert listed["tables"] == [f"{catalog}.docs", f"{catalog}.secret"]

    described = _tool(tools, "describe_table")(f"{catalog}.docs")
    assert any(index["index_type"] == "embedding" and index["columns"] == ["text"] for index in described["indexes"])
    assert {column["name"] for column in described["columns"]} >= {"topic", "text", "status", "pos", "active"}

    queried = _tool(tools, "query_table")(
        f"{catalog}.docs",
        columns=["topic", "status"],
        where=json.dumps({"status": "open", "active": True, "pos": 1}),
        limit=10,
    )
    assert queried["truncated"] is False
    assert queried["rows"] == [{"topic": "expenses", "status": "open"}]

    found = _tool(tools, "similarity_search")(
        f"{catalog}.docs",
        column="text",
        query="meals are reimbursed up to 60 EUR",
        limit=2,
    )
    assert found["rows"][0]["text"] == "meals are reimbursed up to 60 EUR"
    assert found["rows"][0]["score"] >= found["rows"][1]["score"]


async def test_allowlist_rejects_other_tables_and_version_handles(catalog: str) -> None:
    tools = pixeltable_tools([f"{catalog}.docs"])
    listed = _tool(tools, "list_tables")()
    assert listed["tables"] == [f"{catalog}.docs"]
    query = _tool(tools, "query_table")
    with pytest.raises(ValueError, match="allowlist"):
        query(f"{catalog}.secret")
    with pytest.raises(ValueError, match="version handle"):
        query(f"{catalog}.docs:1")
    with pytest.raises(ValueError, match="Unknown column"):
        query(f"{catalog}.docs", where=json.dumps({"missing": 1}))
    with pytest.raises(ValueError, match="Int"):
        query(f"{catalog}.docs", where=json.dumps({"pos": "1"}))


async def test_prefix_allowlist_and_star(catalog: str) -> None:
    parent = catalog.split(".", 1)[0]
    prefixed = pixeltable_tools([parent.upper()])
    assert f"{catalog}.secret" in _tool(prefixed, "list_tables")()["tables"]

    starred = pixeltable_tools(["*"])
    assert f"{catalog}.docs" in _tool(starred, "list_tables")()["tables"]
    with pytest.raises(ValueError, match="only allowlist"):
        pixeltable_tools(["*", f"{catalog}.docs"])


async def test_query_caps_and_timestamp_filter(catalog: str) -> None:
    tools = pixeltable_tools([catalog], max_rows=1, max_chars=80)
    queried = _tool(tools, "query_table")(f"{catalog}.docs", columns=["text"], limit=10)
    assert queried["truncated"] is True
    assert len(queried["rows"]) == 1

    memory = PixeltableMemoryStore(table_name=f"{catalog}.memory", embedding=tiny_embed)
    await memory.add("hello")
    created = list(memory.table.select(memory.table.created_at).collect())[0]["created_at"]
    assert isinstance(created, datetime)
    stamped = _tool(pixeltable_tools(["*"], max_rows=5, max_chars=4000), "query_table")(
        memory.table_name,
        columns=["content"],
        where=json.dumps({"created_at": created.isoformat()}),
    )
    assert stamped["rows"] == [{"content": "hello"}]
    assert created.isoformat()


async def test_similarity_requires_one_modality(catalog: str, tmp_path: Path) -> None:
    image = tmp_path / "cat.png"
    Image.new("RGB", (2, 2), color="red").save(image)
    search = _tool(pixeltable_tools([catalog]), "similarity_search")
    with pytest.raises(ValueError, match="exactly one"):
        search(f"{catalog}.docs", column="text")
    with pytest.raises(ValueError, match="exactly one"):
        search(f"{catalog}.docs", column="text", query="meals", image=str(image))
    with pytest.raises(ValueError, match="image embedding"):
        search(f"{catalog}.docs", column="text", image=str(image))
    with pytest.raises(ValueError, match="No such file"):
        search(f"{catalog}.docs", column="text", image="cat.jpg")


async def test_remember_and_recall_round_trip(catalog: str) -> None:
    memory = PixeltableMemoryStore(table_name=f"{catalog}.memory", embedding=tiny_embed, session_id="trip")
    tools = pixeltable_tools([f"{catalog}.docs"], memory=memory)
    remember = _tool(tools, "remember")
    recall = _tool(tools, "recall")

    failed = await _run(remember, {"content": "   "})
    assert failed["status"] == "error"

    stored = await _run(remember, {"content": "The user travels monthly.", "session_id": "trip"})
    assert stored["status"] == "success"
    body = json.loads(stored["content"][0]["text"])
    assert body["table"] == memory.table_name

    found = await recall(query="The user travels monthly.", limit=3)
    assert found["memories"][0]["content"] == "The user travels monthly."
    assert found["memories"][0]["session_id"] == "trip"
    assert found["truncated"] is False

    required = remember.tool_spec["inputSchema"]["json"]["required"]
    assert "content" in required
    assert "session_id" not in required


def test_null_where_matches_null_role(catalog: str) -> None:
    memory = PixeltableMemoryStore(table_name=f"{catalog}.memory", embedding=tiny_embed)
    _ = memory.table
    now = datetime.now(UTC)
    memory.table.insert(
        [
            {
                "memory_id": "n1",
                "content": "no role",
                "kind": "note",
                "role": None,
                "metadata": None,
                "session_id": None,
                "created_at": now,
            }
        ]
    )
    # The insert above does not go through add(), but the index still updates.
    query = _tool(pixeltable_tools(["*"]), "query_table")
    result = query(memory.table_name, columns=["content", "role"], where=json.dumps({"role": None}))
    assert result["rows"] == [{"content": "no role", "role": None}]
