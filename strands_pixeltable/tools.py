"""Read-only Pixeltable tools, plus remember/recall for a memory store."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import date, datetime
from typing import Any, cast
from uuid import UUID

import pixeltable as pxt
from strands import tool
from strands.types.tools import AgentTool

from strands_pixeltable._types import column_base
from strands_pixeltable.store import PixeltableMemoryStore

_MEDIA = frozenset({"Image", "Video", "Audio", "Document"})
_SKIPPED = frozenset({"Array", "Binary"})
_ALL = "*"

_WHERE_VALUE_TYPES: dict[str, type | tuple[type, ...]] = {
    "String": str,
    "Int": int,
    "Float": (int, float),
    "Bool": bool,
    "Timestamp": str,
    "Date": str,
    "UUID": str,
}
_WHERE_PARSERS: dict[str, Callable[[str], Any]] = {
    "Timestamp": datetime.fromisoformat,
    "Date": date.fromisoformat,
    "UUID": UUID,
}


def _norm(path: str) -> str:
    return path.replace("/", ".").lower()


def _allowed(path: str, tables: list[str]) -> bool:
    if ":" in path:
        return False
    if tables == [_ALL]:
        return True
    normalized = _norm(path)
    return any(normalized == _norm(entry) or normalized.startswith(f"{_norm(entry)}.") for entry in tables)


def _is_media(type_: object) -> bool:
    return column_base(type_) in _MEDIA


def _is_skipped(type_: object) -> bool:
    return column_base(type_) in _SKIPPED


def _cell(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _cell(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_cell(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            converted = item()
        except (ValueError, TypeError):
            converted = None
        else:
            if isinstance(converted, (str, int, float, bool)):
                return converted
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return isoformat()
    return str(value)


def _row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: _cell(value) for key, value in row.items()}


def _parse_where(where: str | dict[str, Any] | None) -> dict[str, Any] | None:
    if where is None or where == "":
        return None
    if isinstance(where, dict):
        return where
    if isinstance(where, str):
        try:
            parsed = json.loads(where)
        except json.JSONDecodeError as exc:
            raise ValueError(f"where must be a JSON object: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("where must be a JSON object")
        return parsed
    raise ValueError("where must be a JSON object")


def _bounded(table: str, rows: list[dict[str, Any]], *, has_more: bool, max_chars: int) -> dict[str, Any]:
    """Keep leading rows while the JSON payload fits in ``max_chars``."""
    empty = {"table": table, "rows": [], "truncated": True}
    if len(json.dumps(empty, ensure_ascii=False)) > max_chars:
        return empty
    kept: list[dict[str, Any]] = []
    truncated = has_more
    for row in rows:
        trial = {"table": table, "rows": [*kept, row], "truncated": truncated}
        if len(json.dumps(trial, ensure_ascii=False)) <= max_chars:
            kept.append(row)
            continue
        truncated = True
        if not kept:
            kept.append(_shrink_row(row, max_chars - len(json.dumps(empty, ensure_ascii=False))))
        break
    return {"table": table, "rows": kept, "truncated": truncated}


def _shrink_row(row: dict[str, Any], budget: int) -> dict[str, Any]:
    clipped = dict(row)
    for _ in range(32):
        if len(json.dumps(clipped, ensure_ascii=False)) <= max(budget, 8):
            return clipped
        strings = [key for key, value in clipped.items() if isinstance(value, str) and len(value) > 3]
        if not strings:
            return {key: ("..." if isinstance(value, str) else None) for key, value in clipped.items()}
        key = max(strings, key=lambda name: len(str(clipped[name])))
        text = str(clipped[key])
        clipped[key] = text[: max(0, len(text) // 2)] + "..."
    return clipped


class _Catalog:
    """Allowlisted, read-only view of a Pixeltable catalog."""

    def __init__(self, tables: list[str], *, max_rows: int, max_chars: int) -> None:
        self._tables = tables
        self._max_rows = max_rows
        self._max_chars = max_chars

    def list_tables(self) -> dict[str, list[str]]:
        try:
            names = pxt.list_tables()
        except pxt.Error as exc:
            raise ValueError(str(exc)) from exc
        return {"tables": sorted(_norm(name) for name in names if _allowed(name, self._tables))}

    def describe_table(self, table: str) -> dict[str, Any]:
        opened, metadata = self._open(table)
        del opened
        indexes = metadata.get("indexes") or {}
        described: list[dict[str, Any]] = []
        for info in indexes.values():
            if not isinstance(info, dict):
                continue
            item: dict[str, Any] = {
                "name": info.get("name"),
                "index_type": info.get("index_type"),
                "columns": info.get("columns"),
            }
            parameters = info.get("parameters") or {}
            if isinstance(parameters, dict) and parameters.get("metric"):
                item["metric"] = parameters["metric"]
            described.append(item)
        return {
            "table": _norm(str(metadata.get("path") or table)),
            "kind": metadata.get("kind"),
            "comment": metadata.get("comment"),
            "columns": [
                {
                    "name": info.get("name"),
                    "type": info.get("type_"),
                    "is_computed": info.get("is_computed", False),
                    "is_stored": info.get("is_stored", True),
                }
                for info in (metadata.get("columns") or {}).values()
                if isinstance(info, dict)
            ],
            "indexes": described,
        }

    def query_table(
        self,
        table: str,
        columns: list[str] | None = None,
        where: str | dict[str, Any] | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        opened, metadata = self._open(table)
        try:
            projected = self._project(opened, columns, metadata)
        except (pxt.Error, TypeError, ValueError) as exc:
            raise ValueError(str(exc)) from exc
        projected = self._where(projected, opened, _parse_where(where), metadata)
        return self._collect(table, projected, limit)

    def similarity_search(
        self,
        table: str,
        column: str,
        query: str | None = None,
        image: str | None = None,
        columns: list[str] | None = None,
        limit: int = 5,
        idx: str | None = None,
    ) -> dict[str, Any]:
        text = query.strip() if isinstance(query, str) else ""
        image_path = image.strip() if isinstance(image, str) else ""
        if bool(text) == bool(image_path):
            raise ValueError("similarity_search needs exactly one of query or image")
        opened, metadata = self._open(table)
        column_md = metadata.get("columns") or {}
        if column not in column_md:
            raise ValueError(
                f"Unknown column {column!r} on {_norm(str(metadata.get('path') or table))!r}. Call describe_table."
            )
        selected = self._default_columns(metadata) if columns is None else list(columns)
        column_type = column_md[column].get("type_") if isinstance(column_md[column], dict) else None
        if column not in selected and not _is_skipped(column_type):
            selected = [column, *[name for name in selected if name != column]]
        score_name = "score"
        while score_name in column_md:
            score_name = f"similarity_{score_name}"
        try:
            if text:
                similarity = (
                    opened[column].similarity(string=text, idx=idx)
                    if idx is not None
                    else opened[column].similarity(string=text)
                )
            else:
                similarity = (
                    opened[column].similarity(image=image_path, idx=idx)
                    if idx is not None
                    else opened[column].similarity(image=image_path)
                )
            projected = self._project(opened, selected, metadata, **{score_name: similarity})
            ranked = projected.order_by(similarity, asc=False)
        except (pxt.Error, TypeError, ValueError, OSError) as exc:
            raise ValueError(str(exc)) from exc
        return self._collect(table, ranked, limit)

    def _open(self, table: str) -> tuple[Any, dict[str, Any]]:
        if not isinstance(table, str) or not table.strip():
            raise ValueError("table must be a non-empty string")
        if ":" in table:
            raise ValueError(f"Table {table!r} is a version handle. Query the current table.")
        if not _allowed(table, self._tables):
            raise ValueError(f"Table {table!r} is not in the Pixeltable allowlist. Call list_tables.")
        try:
            opened = pxt.get_table(table)
        except pxt.Error as exc:
            raise ValueError(f"Cannot open table {table!r}: {exc}") from exc
        if opened is None:
            raise ValueError(f"Cannot open table {table!r}.")
        try:
            raw = opened.get_metadata()
        except pxt.Error as exc:
            raise ValueError(str(exc)) from exc
        if not isinstance(raw, dict) or "columns" not in raw:
            raise ValueError(f"unexpected metadata for {table!r}")
        return opened, cast(dict[str, Any], raw)

    def _default_columns(self, metadata: dict[str, Any]) -> list[str]:
        names: list[str] = []
        for name, info in (metadata.get("columns") or {}).items():
            if not isinstance(info, dict):
                continue
            type_ = info.get("type_")
            if _is_media(type_) or _is_skipped(type_):
                continue
            if info.get("is_computed") and not info.get("is_stored"):
                continue
            names.append(str(name))
        return names

    def _project(self, table: Any, columns: list[str] | None, metadata: dict[str, Any], **extra: Any) -> Any:
        path = _norm(str(metadata.get("path") or ""))
        column_md = metadata.get("columns") or {}
        if columns is None:
            names = self._default_columns(metadata)
        else:
            names = list(dict.fromkeys(columns))
            if not names:
                raise ValueError("query_table needs at least one column")
        items: list[Any] = []
        named: dict[str, Any] = dict(extra)
        for name in names:
            if name in named:
                continue
            if name not in column_md:
                raise ValueError(f"Unknown column {name!r} on {path!r}. Call describe_table.")
            info = column_md[name]
            type_ = info.get("type_") if isinstance(info, dict) else None
            if _is_skipped(type_):
                continue
            ref = table[name]
            if _is_media(type_):
                named[name] = ref.fileurl
            else:
                items.append(ref)
        if not items and not named:
            raise ValueError("No selectable columns. Name a non-array column, or call describe_table.")
        return table.select(*items, **named)

    def _where(self, query: Any, table: Any, where: dict[str, Any] | None, metadata: dict[str, Any]) -> Any:
        if not where:
            return query
        column_md = metadata.get("columns") or {}
        predicate = None
        for name, value in where.items():
            if not isinstance(name, str) or name not in column_md:
                raise ValueError(f"Unknown column {name!r} in where. Call describe_table.")
            info = column_md[name]
            if not isinstance(info, dict):
                raise ValueError(f"Unknown column {name!r} in where. Call describe_table.")
            type_ = info.get("type_")
            if info.get("is_computed") and not info.get("is_stored"):
                raise ValueError(f"Column {name!r} is computed on read and cannot be filtered.")
            if value is not None and (_is_skipped(type_) or _is_media(type_)):
                raise ValueError(f"Column {name!r} does not support equality filters.")
            if value is not None and not isinstance(value, (str, int, float, bool)):
                raise ValueError(f"where value for {name!r} must be a scalar, got {type(value).__name__}")
            base = column_base(type_)
            expected = _WHERE_VALUE_TYPES.get(base)
            if (
                value is not None
                and expected is not None
                and (not isinstance(value, expected) or (isinstance(value, bool) and base != "Bool"))
            ):
                raise ValueError(f"where value for {name!r} must be {base}, got {type(value).__name__}")
            parser = _WHERE_PARSERS.get(base)
            if parser is not None and isinstance(value, str):
                try:
                    value = parser(value)
                except ValueError as exc:
                    raise ValueError(f"where value for {name!r} must be an ISO-format {base}: {exc}") from exc
            try:
                clause = table[name] == value
                predicate = clause if predicate is None else predicate & clause
            except (pxt.Error, TypeError, ValueError) as exc:
                raise ValueError(f"Cannot filter {name!r} by {value!r}: {exc}") from exc
        try:
            return query.where(predicate)
        except (pxt.Error, TypeError, ValueError) as exc:
            raise ValueError(str(exc)) from exc

    def _collect(self, table: str, query: Any, limit: int) -> dict[str, Any]:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be an integer >= 1")
        count = min(limit, self._max_rows)
        try:
            fetched = list(query.limit(count + 1).collect())
        except pxt.Error as exc:
            raise ValueError(str(exc)) from exc
        has_more = len(fetched) > count
        rows = [_row(dict(row)) for row in fetched[:count]]
        return _bounded(_norm(table), rows, has_more=has_more, max_chars=self._max_chars)


def pixeltable_tools(
    tables: list[str],
    *,
    memory: PixeltableMemoryStore | None = None,
    max_rows: int = 20,
    max_chars: int = 8000,
) -> list[AgentTool]:
    """Build Strands tools for a Pixeltable catalog.

    ``tables`` is an allowlist of table paths or directory prefixes, matched
    case-insensitively the way Pixeltable folds names. ``["*"]`` allows every
    table. ``query_table`` and ``similarity_search`` are read-only: equality
    filters only, no inserts, updates, or deletes. Version handles (``tbl:3``)
    are refused.

    When ``memory`` is set, the list also contains ``remember`` and ``recall``.
    Those write and search that store only.

    Args:
        tables: Allowlist. Use ``["*"]`` for the whole catalog.
        memory: Optional memory store backing ``remember`` and ``recall``.
        max_rows: Cap on rows returned by one call.
        max_chars: Cap on the JSON size of one result.
    """
    if not isinstance(tables, list) or not tables:
        raise ValueError("tables must be a non-empty allowlist; pass ['*'] to allow the whole catalog")
    normalized = []
    for entry in tables:
        if not isinstance(entry, str) or not entry.strip():
            raise ValueError("allowlist entries must be non-empty strings")
        normalized.append(entry.strip())
    if _ALL in normalized and normalized != [_ALL]:
        raise ValueError("'*' must be the only allowlist entry")
    if isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows < 1:
        raise ValueError("max_rows must be an integer >= 1")
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars < 1:
        raise ValueError("max_chars must be an integer >= 1")
    if memory is not None and not isinstance(memory, PixeltableMemoryStore):
        raise ValueError("memory must be a PixeltableMemoryStore")

    catalog = _Catalog(normalized, max_rows=max_rows, max_chars=max_chars)
    tools: list[AgentTool] = []
    if memory is not None:
        tools.append(_remember_tool(memory))
        tools.append(_recall_tool(memory, max_rows=max_rows, max_chars=max_chars))
    tools.append(_list_tables_tool(catalog))
    tools.append(_describe_table_tool(catalog))
    tools.append(_query_table_tool(catalog))
    tools.append(_similarity_search_tool(catalog))
    return tools


def _remember_tool(memory: PixeltableMemoryStore) -> AgentTool:
    @tool(
        description=(
            f"Store one durable note in the Pixeltable memory table {memory.table_name}. "
            "Does not modify any other table."
        )
    )
    async def remember(content: str, session_id: str | None = None) -> dict[str, str]:
        """Store a note for later semantic recall.

        Args:
            content: The fact or note to remember.
            session_id: Optional session id stored with the note.

        Returns:
            The memory id and table path.
        """
        metadata: dict[str, Any] | None = {"session_id": session_id} if session_id else None
        memory_id = await memory.add(content, metadata)
        return {"memory_id": memory_id, "table": memory.table_name}

    return remember


def _recall_tool(memory: PixeltableMemoryStore, *, max_rows: int, max_chars: int) -> AgentTool:
    @tool(
        description=(
            f"Semantic search over the Pixeltable memory table {memory.table_name}. "
            "Read-only. Returns the closest notes and messages."
        )
    )
    async def recall(query: str, limit: int = 5, session_id: str | None = None) -> dict[str, Any]:
        """Search memory by meaning.

        Args:
            query: Text to embed and compare with stored notes and messages.
            limit: Maximum number of hits. Capped by the tool's row limit.
            session_id: When set, only rows from this session are searched.

        Returns:
            Ranked memories. ``truncated`` is true when a hit was shortened.
        """
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be an integer >= 1")
        fetch = min(limit, max_rows)
        options: dict[str, Any] = {"max_search_results": fetch}
        if session_id:
            options["session_id"] = session_id
        entries = await memory.search(query, options)  # type: ignore[arg-type]
        memories: list[dict[str, Any]] = []
        truncated = False
        used = 0
        for entry in entries:
            content = entry.content
            if len(content) > max_chars:
                content = content[: max(0, max_chars - 3)] + "..."
                truncated = True
            if used + len(content) > max_chars and memories:
                truncated = True
                break
            meta = entry.metadata or {}
            memories.append(
                {
                    "content": content,
                    "score": round(float(meta.get("score", 0.0)), 6),
                    "memory_id": meta.get("memory_id"),
                    "kind": meta.get("kind"),
                    "role": meta.get("role"),
                    "session_id": meta.get("session_id"),
                }
            )
            used += len(content)
        return {"query": query, "memories": memories, "truncated": truncated}

    return recall


def _list_tables_tool(catalog: _Catalog) -> AgentTool:
    @tool
    def list_tables() -> dict[str, list[str]]:
        """List Pixeltable tables this agent may read.

        Returns:
            The allowed table paths.
        """
        return catalog.list_tables()

    return list_tables


def _describe_table_tool(catalog: _Catalog) -> AgentTool:
    @tool
    def describe_table(table: str) -> dict[str, Any]:
        """Describe a table's columns and indexes. Read-only.

        Args:
            table: Pixeltable table path, for example ``handbook.docs``.

        Returns:
            Kind, comment, columns, and indexes.
        """
        return catalog.describe_table(table)

    return describe_table


def _query_table_tool(catalog: _Catalog) -> AgentTool:
    @tool
    def query_table(
        table: str,
        columns: list[str] | None = None,
        where: str | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Select rows from a table. Read-only. ``where`` is equality only.

        Args:
            table: Pixeltable table path.
            columns: Columns to return. Omit to skip media, array, binary, and
                computed columns that are not stored. A named media column is
                returned as a file URL.
            where: JSON object of equality filters, for example
                ``{"status": "open"}``. Timestamp, date, and UUID values are
                ISO strings. Null matches null rows.
            limit: Maximum rows to return, capped by the tool.

        Returns:
            ``{"table", "rows", "truncated"}``. ``truncated`` is true when the
            row cap or the character cap dropped data.
        """
        return catalog.query_table(table, columns=columns, where=where, limit=limit)

    return query_table


def _similarity_search_tool(catalog: _Catalog) -> AgentTool:
    @tool
    def similarity_search(
        table: str,
        column: str,
        query: str | None = None,
        image: str | None = None,
        columns: list[str] | None = None,
        limit: int = 5,
        idx: str | None = None,
    ) -> dict[str, Any]:
        """Nearest-neighbor search on a column that has an embedding index. Read-only.

        Pass ``query`` for text, or ``image`` for a local path or URL. The
        column's embedding index must support that modality. Pass ``idx`` when
        the column has more than one embedding index.

        Args:
            table: Pixeltable table path.
            column: Column with an embedding index.
            query: Text to embed and search for.
            image: Image file path or URL to search for.
            columns: Extra columns to return with the match.
            limit: Maximum rows to return, capped by the tool.
            idx: Embedding index name.

        Returns:
            Rows ordered by similarity, each with a score field.
        """
        return catalog.similarity_search(
            table,
            column,
            query=query,
            image=image,
            columns=columns,
            limit=limit,
            idx=idx,
        )

    return similarity_search
