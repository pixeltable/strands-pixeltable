"""Strands agent with Pixeltable memory and a handbook table.

Setup::

    pip install "strands-pixeltable[local]"

The script writes to the local Pixeltable catalog (``~/.pixeltable`` by default).
It stores a note and prints a semantic search without calling a model.
Set ``RUN_AGENT=1`` to invoke the agent. That needs credentials for the model
provider. The default model is Amazon Bedrock. Set ``STRANDS_MODEL`` to a Bedrock
model id to override it.
"""

from __future__ import annotations

import asyncio
import os

import pixeltable as pxt
from strands import Agent
from strands.memory import MemoryManager

from strands_pixeltable import PixeltableMemoryStore, pixeltable_tools, sentence_transformer_embedding


def build_handbook(embedding: object) -> None:
    """Create a one-row handbook and an embedding index, if they are not there yet."""
    pxt.create_dir("examples", if_exists="ignore")
    docs = pxt.create_table(
        "examples.handbook",
        {"topic": pxt.String, "text": pxt.String},
        if_exists="ignore",
    )
    if docs.count() == 0:
        docs.insert(
            [
                {
                    "topic": "expenses",
                    "text": "Meals during business travel are reimbursed up to 60 EUR per day.",
                }
            ]
        )
    docs.add_embedding_index("text", embedding=embedding, if_exists="ignore")


async def remember_and_search(store: PixeltableMemoryStore) -> None:
    memory_id = await store.add("The user travels monthly.", {"source": "example"})
    hits = await store.search("How often does the user travel?")
    print(f"stored {memory_id}")
    for hit in hits:
        score = (hit.metadata or {}).get("score")
        print(f"  {score}: {hit.content}")


def main() -> None:
    embedding = sentence_transformer_embedding()
    build_handbook(embedding)
    store = PixeltableMemoryStore(table_name="examples.memory", embedding=embedding)
    asyncio.run(remember_and_search(store))

    agent = Agent(
        model=os.environ.get("STRANDS_MODEL"),
        system_prompt=(
            "Answer from the handbook. Use similarity_search or query_table for policy text. "
            "Use remember for durable facts about the user."
        ),
        tools=pixeltable_tools(["examples.handbook"], memory=store),
        memory_manager=MemoryManager(stores=[store], add_tool_config=True),
        callback_handler=None,
    )
    if os.environ.get("RUN_AGENT") == "1":
        print(agent("Can I expense a 75 EUR dinner? Remember that I travel monthly."))
    else:
        print("Agent is ready. Set RUN_AGENT=1 to call the model.")
        print("Tools:", ", ".join(sorted(agent.tool_names)))


if __name__ == "__main__":
    main()
