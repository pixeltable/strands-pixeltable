# strands-pixeltable

[![CI](https://github.com/pixeltable/strands-pixeltable/actions/workflows/ci.yml/badge.svg)](https://github.com/pixeltable/strands-pixeltable/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/strands-pixeltable.svg)](https://pypi.org/project/strands-pixeltable/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/pixeltable/strands-pixeltable/blob/main/LICENSE)

Pixeltable memory and read-only table tools for [Strands Agents](https://strandsagents.com). Pixeltable is the table and embedding store ([docs](https://docs.pixeltable.com), [source](https://github.com/pixeltable/pixeltable)). Strands tool and memory interfaces are documented at [strandsagents.com](https://strandsagents.com/docs/).

- `PixeltableMemoryStore` implements `strands.memory.MemoryStore`. `MemoryManager` searches it with the store's embedding index and can write notes and messages into it.
- `pixeltable_tools` builds `@tool` functions: `remember`, `recall`, `list_tables`, `describe_table`, `query_table`, and `similarity_search`.

Requires Python 3.11+ (Pixeltable's floor), strands-agents 1.61.0+, and Pixeltable 0.7.8+.

## Install

```bash
pip install strands-pixeltable
```

Local embeddings, with no API key:

```bash
pip install "strands-pixeltable[local]"
```

## Memory

```python
import asyncio

from strands import Agent
from strands.memory import MemoryManager
from strands_pixeltable import PixeltableMemoryStore, sentence_transformer_embedding

embed = sentence_transformer_embedding()  # sentence-transformers/all-MiniLM-L6-v2
store = PixeltableMemoryStore(table_name="strands.memory", embedding=embed)


async def main() -> None:
    await store.add("The user prefers aisle seats.")
    for hit in await store.search("seat preference"):
        print(hit.metadata["score"], hit.content)


asyncio.run(main())

agent = Agent(
    memory_manager=MemoryManager(stores=[store], add_tool_config=True),
)
```

`MemoryManager` registers `search_memory`. `add_tool_config=True` also registers `add_memory`. With the default `injection=True`, retrieved rows are prepended to the model input and are not written into the conversation history.

The table is created on first use. Columns are `memory_id` (primary key), `content`, `kind` (`note` or `message`), `role`, `metadata`, `session_id`, and `created_at`. The store adds a btree index on `session_id` and an embedding index named `content_embedding` on `content`. `store.table` is that table.

`add_messages` stores each message containing text as `"<role>: <text>"`, joining multiple text blocks with newlines. A message `tracking_id` becomes the primary key `msg:<tracking_id>`, so a retry does not insert a second row. Tool-only messages are skipped.

`extraction=True` opts into Strands background extraction. This store implements `add_messages`, so the manager writes the raw messages here and does not make a separate extraction model call. The default is off.

`session_id=` on the store filters search to that session. Pass `session_id=""` in the search options to search every session. A note's `metadata["session_id"]` overrides the store default for that row.

`embedding` must be a module-level Pixeltable UDF or a `.using()` binding. Omit it to use the local MiniLM model. Reopening a table that already has `content_embedding` does not load the model. The index is fixed at creation; `if_exists="replace"` builds a new table. `replace` fails when the table has views or snapshots.

## Tools

```python
import pixeltable as pxt
from strands import Agent
from strands.memory import MemoryManager
from strands_pixeltable import PixeltableMemoryStore, pixeltable_tools, sentence_transformer_embedding

embed = sentence_transformer_embedding()
pxt.create_dir("handbook", if_exists="ignore")
docs = pxt.create_table(
    "handbook.docs",
    {"topic": pxt.String, "text": pxt.String},
    if_exists="ignore",
)
if docs.count() == 0:
    docs.insert([{"topic": "expenses", "text": "Meals during business travel are reimbursed up to 60 EUR per day."}])
docs.add_embedding_index("text", string_embed=embed, if_exists="ignore")

memory = PixeltableMemoryStore(table_name="handbook.memory", embedding=embed)
agent = Agent(
    system_prompt="Answer from the handbook. Remember durable facts about the user.",
    tools=pixeltable_tools(["handbook.docs"], memory=memory),
    memory_manager=MemoryManager(stores=[memory], add_tool_config=True),
)
print(agent("Can I expense a 75 EUR dinner? Remember that I travel monthly."))
```

`tables` is required. It is an allowlist of table paths or directory prefixes, matched case-insensitively. `["*"]` is the whole catalog. `["*", "handbook.docs"]` is rejected.

- `remember` / `recall` write and search the memory store only.
- `list_tables` / `describe_table` report allowlisted tables, columns, and indexes.
- `query_table` selects rows. `where` is a JSON object of equality filters (`{"status": "open"}`). Timestamp, date, and UUID values are ISO strings.
- `similarity_search` runs `column.similarity`. Pass `query` for text or `image` for a file path or URL. The column needs an embedding index for that modality. Pass `idx` when the column has more than one embedding index.

`query_table` and `similarity_search` do not insert, update, or delete. Default columns skip media, array, binary, and computed columns that are not stored. Explicitly selecting an unstored computed column is rejected so reading does not execute its function. A named media column is returned as a file URL. Version handles (`docs:3`) are refused.

`max_rows` (20) and `max_chars` (8000) bound query, similarity, and recall results, including their JSON envelope and metadata. `truncated` is true when a cap drops data; if even a shortened row cannot fit, the result list is empty. A budget too small for the result envelope raises `ValueError`.

[`examples/agent_with_memory.py`](examples/agent_with_memory.py) builds the same agent. It stores a note and searches without a model. `RUN_AGENT=1` invokes the model.

## Development

```bash
pip install -e ".[dev]"
pytest tests/
ruff check . && ruff format --check .
mypy strands_pixeltable
```

Tests use a hash embedding and do not need an API key.

## License

Apache 2.0
