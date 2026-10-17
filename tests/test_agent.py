"""The store and tools drop into Agent and MemoryManager."""

from __future__ import annotations

import uuid

import pixeltable as pxt
from strands import Agent
from strands.memory import MemoryManager

from strands_pixeltable import PixeltableMemoryStore, pixeltable_tools
from tests.embed import tiny_embed


def test_agent_registers_memory_and_catalog_tools() -> None:
    root = f"test_agent_{uuid.uuid4().hex[:8]}"
    try:
        store = PixeltableMemoryStore(table_name=f"{root}.memory", embedding=tiny_embed, name="notes")
        agent = Agent(
            model="us.amazon.nova-micro-v1:0",
            tools=pixeltable_tools([root], memory=store),
            system_prompt="Use the handbook and the memory tools.",
            memory_manager=MemoryManager(stores=[store], add_tool_config=True, injection=False),
            callback_handler=None,
            load_tools_from_directory=False,
        )
        names = set(agent.tool_names)
        assert {"remember", "recall", "query_table", "similarity_search", "search_memory", "add_memory"} <= names
        assert agent.memory_manager is not None
        assert agent.memory_manager._stores[0] is store
    finally:
        try:
            pxt.drop_dir(root, force=True)
        except pxt.Error:
            pass
