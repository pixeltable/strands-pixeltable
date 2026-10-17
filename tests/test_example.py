"""The documented handbook setup runs without a model or downloaded embeddings."""

import pixeltable as pxt

from examples.agent_with_memory import build_handbook
from strands_pixeltable import pixeltable_tools
from tests.embed import tiny_embed


def test_handbook_setup_is_repeatable() -> None:
    try:
        build_handbook(tiny_embed)
        build_handbook(tiny_embed)
        assert pxt.get_table("examples.handbook").count() == 1
        search = next(tool for tool in pixeltable_tools(["examples.handbook"]) if tool.tool_name == "similarity_search")
        found = search("examples.handbook", column="text", query="Meals during business travel")
        assert found["rows"][0]["topic"] == "expenses"
    finally:
        pxt.drop_dir("examples", force=True)
