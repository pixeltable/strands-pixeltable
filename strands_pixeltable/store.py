"""Pixeltable backend for a Strands ``MemoryStore``."""

from __future__ import annotations

import asyncio
import json
import threading
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, Literal, cast

import pixeltable as pxt
from strands.memory import AddMessagesContext, ExtractionConfig, MemoryEntry, MemoryStore, SearchOptions
from strands.memory.types import Metadata
from strands.types.content import Message

from strands_pixeltable._types import column_base, is_nullable_type
from strands_pixeltable.embeddings import sentence_transformer_embedding

_KIND_NOTE = "note"
_KIND_MESSAGE = "message"
_CONTENT_INDEX = "content_embedding"
_SESSION_INDEX = "session_id_idx"
_DEFAULT_MAX_SEARCH_RESULTS = 5
_TABLE_COMMENT = "Memories for a Strands agent. Managed by strands-pixeltable."
_MAX_ID_CHARS = 200

# Base columns the store owns. Indexes are added on first writable open.
_BASE_COLUMNS: dict[str, tuple[str, bool]] = {
    "memory_id": ("String", False),
    "content": ("String", False),
    "kind": ("String", False),
    "role": ("String", True),
    "metadata": ("Json", True),
    "session_id": ("String", True),
    "created_at": ("Timestamp", False),
}

_METRICS = ("cosine", "ip", "l2")
_PRECISIONS = ("fp16", "fp32")
_IF_EXISTS = ("error", "ignore", "replace")


def _schema() -> dict[str, Any]:
    return {
        "memory_id": pxt.String,
        "content": pxt.String,
        "kind": pxt.String,
        "role": pxt.String | None,
        "metadata": pxt.Json | None,
        "session_id": pxt.String | None,
        "created_at": pxt.Timestamp,
    }


def _is_duplicate_key(exc: BaseException) -> bool:
    return isinstance(exc, pxt.Error) and "Duplicate primary key" in str(exc)


def _json_object(metadata: Metadata | None) -> dict[str, Any] | None:
    if not metadata:
        return None
    try:
        encoded = json.dumps(metadata)
    except (TypeError, ValueError) as exc:
        raise ValueError("metadata must be JSON-serializable") from exc
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):
        raise ValueError("metadata must be a JSON object")
    return decoded


def _block_text(block: object) -> str:
    if isinstance(block, dict):
        text = block.get("text")
        return text.strip() if isinstance(text, str) else ""
    text = getattr(block, "text", None)
    return text.strip() if isinstance(text, str) else ""


def _message_text(message: Message) -> str:
    content = message.get("content")
    if not isinstance(content, list):
        raise ValueError("message content must be a list of content blocks")
    parts = [_block_text(block) for block in content]
    return "\n".join(part for part in parts if part).strip()


class PixeltableMemoryStore(MemoryStore):
    """Strands ``MemoryStore`` persisted in a Pixeltable table.

    ``add`` inserts a note. ``add_messages`` inserts conversation text, one row per
    message that has text. ``search`` orders rows by ``content.similarity`` against
    the ``content_embedding`` index.

    The first writable open creates the table, a btree on ``session_id``, and the
    embedding index on ``content``.
    Pass ``embedding`` as a module-level Pixeltable UDF or a ``.using()`` binding.
    Omit it to use :func:`strands_pixeltable.sentence_transformer_embedding`
    (install the ``local`` extra). A later open reuses the index and does not
    reload that default.

    The index is on the string column, not on a computed array column.
    ``Table.get_metadata()`` in current Pixeltable releases raises if an embedding
    index sits on an array column that has only a string query embedder.

    ``extraction=True`` turns on Strands background extraction. This store
    implements ``add_messages``, so the manager stores the raw messages here and
    does not make a separate extraction model call.

    Args:
        table_name: Pixeltable table path, for example ``strands.memory``.
        name: Store name ``MemoryManager`` uses to address this store.
        description: Shown in the manager's tool descriptions.
        embedding: Embedding UDF used to build the index. Required only when the
            index does not already exist and the local model should not be used.
        writable: When false, ``add`` and ``add_messages`` raise and a missing
            table or index is not created.
        max_search_results: Default ``search`` limit. A call can override it.
        extraction: Passed through to ``MemoryManager``. ``None`` leaves
            extraction off.
        session_id: Default session stamped on writes and applied as a search
            filter. A write's ``metadata['session_id']`` overrides it. A search
            passes ``session_id=''`` to search every session.
        metric: Embedding index distance. One of ``cosine``, ``ip``, ``l2``.
        precision: Embedding index precision. One of ``fp16``, ``fp32``.
        if_exists: ``ignore`` opens an existing table, ``error`` refuses one,
            ``replace`` drops and recreates it. ``replace`` fails if the table
            has views or snapshots.
    """

    def __init__(
        self,
        table_name: str = "strands.memory",
        *,
        name: str = "pixeltable",
        description: str | None = None,
        embedding: Any | None = None,
        writable: bool = True,
        max_search_results: int | None = _DEFAULT_MAX_SEARCH_RESULTS,
        extraction: ExtractionConfig | bool | None = None,
        session_id: str | None = None,
        metric: Literal["cosine", "ip", "l2"] = "cosine",
        precision: Literal["fp16", "fp32"] = "fp16",
        if_exists: Literal["error", "ignore", "replace"] = "ignore",
    ) -> None:
        if (
            not isinstance(table_name, str)
            or not table_name.strip()
            or any(part == "" for part in table_name.split("."))
        ):
            raise ValueError(f"invalid Pixeltable table path: {table_name!r}")
        if ":" in table_name:
            raise ValueError(f"table path {table_name!r} must not be a version handle")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string")
        if max_search_results is not None and (
            isinstance(max_search_results, bool) or not isinstance(max_search_results, int) or max_search_results < 1
        ):
            raise ValueError("max_search_results must be an integer >= 1")
        if session_id is not None and (not isinstance(session_id, str) or not session_id.strip()):
            raise ValueError("session_id must be a non-empty string")
        if metric not in _METRICS:
            raise ValueError(f"metric must be one of {_METRICS}")
        if precision not in _PRECISIONS:
            raise ValueError(f"precision must be one of {_PRECISIONS}")
        if if_exists not in _IF_EXISTS:
            raise ValueError(f"if_exists must be one of {_IF_EXISTS}")

        self.table_name = table_name
        self.name = name
        self.description = (
            description
            if description is not None
            else "Long-term notes and messages stored in a Pixeltable table and retrieved by semantic similarity."
        )
        self.writable = writable
        self.max_search_results = max_search_results
        self.extraction = extraction
        self.session_id = session_id.strip() if isinstance(session_id, str) else None
        self._embedding = embedding
        self._metric = metric
        self._precision = precision
        self._if_exists = if_exists
        self._lock = threading.Lock()
        self._table: pxt.Table | None = None

    def __repr__(self) -> str:
        return f"PixeltableMemoryStore(name={self.name!r}, table_name={self.table_name!r}, writable={self.writable})"

    @property
    def table(self) -> pxt.Table:
        """The Pixeltable table. Available for queries the store does not wrap."""
        return self._locked(self._ensure_table)

    async def search(self, query: str, options: SearchOptions | None = None) -> list[MemoryEntry]:
        """Return notes and messages ranked by embedding similarity to ``query``.

        ``options`` may include ``max_search_results`` and, for this store,
        ``session_id``. ``session_id=''`` searches every session.
        """
        raw: dict[str, Any] | None = dict(options) if options is not None else None
        return await asyncio.to_thread(self._locked, self._search_locked, query, raw)

    async def add(self, content: str, metadata: Metadata | None = None) -> str:
        """Insert one note and return its ``memory_id``.

        ``metadata['memory_id']`` is used as the primary key when it is a
        non-empty string. Inserting that id again raises ``ValueError``.
        ``metadata['session_id']`` overrides the store default.
        """
        return await asyncio.to_thread(self._locked, self._add_locked, content, metadata)

    async def add_messages(self, messages: list[Message], context: AddMessagesContext | None = None) -> list[str]:
        """Insert one row per message that contains text. Return the new ids.

        A message with a ``tracking_id`` is stored as ``msg:<tracking_id>``.
        Inserting that id again is a no-op, so a retried batch is not duplicated.
        Messages with no text (tool-only turns) are skipped. The stored text is
        ``"<role>: <text>"``.
        """
        return await asyncio.to_thread(self._locked, self._add_messages_locked, messages, context)

    def _locked(self, fn: Any, *args: Any) -> Any:
        with self._lock:
            try:
                return fn(*args)
            except pxt.NotFoundError:
                self._table = None
                return fn(*args)

    def _ensure_dirs(self) -> None:
        if "." not in self.table_name:
            return
        parent = self.table_name.rsplit(".", 1)[0]
        parts: list[str] = []
        for part in parent.split("."):
            parts.append(part)
            pxt.create_dir(".".join(parts), if_exists="ignore")

    def _ensure_table(self) -> pxt.Table:
        if self._table is not None:
            return self._table
        if self.writable:
            self._ensure_dirs()
        table: pxt.Table | None
        if self._if_exists == "replace":
            if not self.writable:
                raise ValueError("if_exists='replace' requires a writable store")
            table = pxt.create_table(
                self.table_name,
                _schema(),
                primary_key="memory_id",
                comment=_TABLE_COMMENT,
                custom_metadata={"component": "strands-pixeltable", "role": "memory"},
                if_exists="replace",
            )
        else:
            table = pxt.get_table(self.table_name, if_not_exists="ignore")
            if table is None:
                if not self.writable:
                    raise ValueError(f"Pixeltable table {self.table_name!r} does not exist and the store is read-only")
                table = pxt.create_table(
                    self.table_name,
                    _schema(),
                    primary_key="memory_id",
                    comment=_TABLE_COMMENT,
                    custom_metadata={"component": "strands-pixeltable", "role": "memory"},
                    if_exists="ignore",
                )
            elif self._if_exists == "error":
                raise ValueError(f"Pixeltable table {self.table_name!r} already exists")
        assert table is not None
        self._check_schema(table)
        self._ensure_derived(table)
        self._table = table
        return table

    def _check_schema(self, table: pxt.Table) -> None:
        metadata = table.get_metadata()
        problems: list[str] = []
        kind = metadata.get("kind")
        if kind != "table":
            problems.append(f"kind is {kind!r}, expected 'table'")
        primary_key = list(metadata.get("primary_key") or [])
        if primary_key != ["memory_id"]:
            problems.append(f"primary key is {primary_key}, expected ['memory_id']")
        columns = metadata.get("columns") or {}
        for name, (expected_base, nullable) in _BASE_COLUMNS.items():
            info = columns.get(name)
            if not isinstance(info, dict):
                problems.append(f"column {name!r} is missing")
                continue
            if info.get("is_computed"):
                problems.append(f"column {name!r} must be stored, not computed")
            actual = info.get("type_")
            if column_base(actual) != expected_base:
                problems.append(f"column {name!r} has type {actual!r}, expected {expected_base}")
            if is_nullable_type(actual) != nullable:
                problems.append(f"column {name!r} has type {actual!r}, expected nullable={nullable}")
        if problems:
            detail = "; ".join(problems)
            raise ValueError(f"Pixeltable table {self.table_name!r} cannot be used as a memory store: {detail}")

    def _index_ok(self, metadata: Mapping[str, Any]) -> bool:
        info = (metadata.get("indexes") or {}).get(_CONTENT_INDEX)
        if not isinstance(info, dict):
            return False
        if info.get("index_type") != "embedding":
            return False
        return "content" in (info.get("columns") or [])

    def _ensure_derived(self, table: pxt.Table) -> None:
        metadata = _metadata_dict(table)
        if self._index_ok(metadata):
            if self.writable:
                self._ensure_session_index(table, metadata)
            return
        if not self.writable:
            raise ValueError(
                f"Pixeltable table {self.table_name!r} has no {_CONTENT_INDEX!r} embedding index "
                "and the store is read-only"
            )
        embedding = self._embedding if self._embedding is not None else sentence_transformer_embedding()
        self._embedding = embedding
        self._ensure_session_index(table, metadata)
        table.add_embedding_index(
            "content",
            string_embed=embedding,
            idx_name=_CONTENT_INDEX,
            metric=self._metric,
            precision=self._precision,
            if_exists="ignore",
        )
        if not self._index_ok(_metadata_dict(table)):
            raise RuntimeError(f"Failed to create embedding index {_CONTENT_INDEX!r} on {self.table_name!r}")

    def _ensure_session_index(self, table: pxt.Table, metadata: Mapping[str, Any]) -> None:
        indexes = metadata.get("indexes") or {}
        if _SESSION_INDEX not in indexes:
            table.add_btree_index("session_id", idx_name=_SESSION_INDEX, if_exists="ignore")

    def _result_limit(self, options: dict[str, Any] | None) -> int:
        raw = options.get("max_search_results") if options else None
        if raw is None:
            raw = self.max_search_results if self.max_search_results is not None else _DEFAULT_MAX_SEARCH_RESULTS
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
            raise ValueError("max_search_results must be an integer >= 1")
        return raw

    def _session_filter(self, options: dict[str, Any] | None) -> str | None:
        if options is not None and "session_id" in options:
            raw = options.get("session_id")
            if raw is None or raw == "":
                return None
            if not isinstance(raw, str):
                raise ValueError("session_id must be a string")
            return raw
        return self.session_id

    def _session_for_write(self, metadata: dict[str, Any] | None) -> str | None:
        if metadata is not None:
            raw = metadata.get("session_id")
            if isinstance(raw, str) and raw.strip():
                return raw.strip()
        return self.session_id

    def _search_locked(self, query: str, options: dict[str, Any] | None) -> list[MemoryEntry]:
        if not isinstance(query, str) or not query.strip():
            return []
        limit = self._result_limit(options)
        session_id = self._session_filter(options)
        table = self._ensure_table()
        similarity = table.content.similarity(string=query.strip(), idx=_CONTENT_INDEX)
        ranked = table.select(
            table.memory_id,
            table.content,
            table.kind,
            table.role,
            table.metadata,
            table.session_id,
            table.created_at,
            score=similarity,
        )
        if session_id is not None:
            ranked = ranked.where(table.session_id == session_id)
        rows = list(ranked.order_by(similarity, asc=False).limit(limit).collect())
        entries: list[MemoryEntry] = []
        for row in rows:
            data = dict(row)
            score = data.get("score")
            content = data.get("content")
            if score is None or not isinstance(content, str):
                continue
            entries.append(MemoryEntry(content=content, metadata=_entry_metadata(data, float(score))))
        return entries

    def _add_locked(self, content: str, metadata: Metadata | None) -> str:
        if not self.writable:
            raise ValueError(f"memory store {self.name!r} is not writable")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("content must be a non-empty string")
        stored_metadata = _json_object(metadata)
        memory_id = _explicit_id(stored_metadata)
        row = _note_row(
            memory_id=memory_id,
            content=content,
            metadata=stored_metadata,
            session_id=self._session_for_write(stored_metadata),
        )
        table = self._ensure_table()
        try:
            table.insert([row])
        except pxt.Error as exc:
            if _is_duplicate_key(exc):
                raise ValueError(f"memory_id {memory_id!r} already exists") from exc
            raise
        return memory_id

    def _add_messages_locked(self, messages: list[Message], context: AddMessagesContext | None) -> list[str]:
        if not self.writable:
            raise ValueError(f"memory store {self.name!r} is not writable")
        if not isinstance(messages, list):
            raise ValueError("messages must be a list")
        rows: list[dict[str, Any]] = []
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                raise ValueError("each message must be a dict")
            text = _message_text(message)
            if not text:
                continue
            role = message.get("role")
            if not isinstance(role, str) or not role.strip():
                raise ValueError("message role must be a non-empty string")
            tracking_id = message.get("tracking_id")
            if isinstance(tracking_id, str) and tracking_id.strip():
                memory_id = f"msg:{tracking_id.strip()}"
                if len(memory_id) > _MAX_ID_CHARS:
                    raise ValueError("tracking_id is too long")
            else:
                memory_id = str(uuid.uuid4())
            sequence: int | None = None
            if context is not None and context.sequence_numbers is not None and index < len(context.sequence_numbers):
                sequence = context.sequence_numbers[index]
            meta: dict[str, Any] = {}
            if isinstance(tracking_id, str) and tracking_id.strip():
                meta["tracking_id"] = tracking_id.strip()
            if sequence is not None:
                meta["sequence_number"] = sequence
            rows.append(
                _message_row(
                    memory_id=memory_id,
                    content=f"{role}: {text}",
                    role=role,
                    metadata=meta or None,
                    session_id=self.session_id,
                )
            )
        if not rows:
            return []
        table = self._ensure_table()
        return _insert_new(table, rows)


def _explicit_id(metadata: dict[str, Any] | None) -> str:
    if metadata is not None:
        raw = metadata.get("memory_id")
        if isinstance(raw, str) and raw.strip():
            memory_id = raw.strip()
            if len(memory_id) > _MAX_ID_CHARS:
                raise ValueError("memory_id is too long")
            return memory_id
    return str(uuid.uuid4())


def _note_row(
    *,
    memory_id: str,
    content: str,
    metadata: dict[str, Any] | None,
    session_id: str | None,
) -> dict[str, Any]:
    return {
        "memory_id": memory_id,
        "content": content,
        "kind": _KIND_NOTE,
        "role": None,
        "metadata": metadata,
        "session_id": session_id,
        "created_at": datetime.now(UTC),
    }


def _message_row(
    *,
    memory_id: str,
    content: str,
    role: str,
    metadata: dict[str, Any] | None,
    session_id: str | None,
) -> dict[str, Any]:
    return {
        "memory_id": memory_id,
        "content": content,
        "kind": _KIND_MESSAGE,
        "role": role,
        "metadata": metadata,
        "session_id": session_id,
        "created_at": datetime.now(UTC),
    }


def _metadata_dict(table: pxt.Table) -> dict[str, Any]:
    return cast(dict[str, Any], table.get_metadata())


def _entry_metadata(data: dict[str, Any], score: float) -> dict[str, Any]:
    user_meta = data.get("metadata")
    created = data.get("created_at")
    created_at = created.isoformat() if isinstance(created, datetime) else created
    metadata: dict[str, Any] = {
        "memory_id": data.get("memory_id"),
        "kind": data.get("kind"),
        "role": data.get("role"),
        "session_id": data.get("session_id"),
        "created_at": created_at,
        "score": score,
    }
    if isinstance(user_meta, dict) and user_meta:
        metadata["metadata"] = user_meta
    return metadata


def _insert_new(table: pxt.Table, rows: list[dict[str, Any]]) -> list[str]:
    try:
        table.insert(rows)
    except pxt.Error as exc:
        if not _is_duplicate_key(exc) or len(rows) == 1:
            if _is_duplicate_key(exc):
                return []
            raise
        stored: list[str] = []
        for row in rows:
            try:
                table.insert([row])
            except pxt.Error as row_exc:
                if _is_duplicate_key(row_exc):
                    continue
                raise
            stored.append(str(row["memory_id"]))
        return stored
    return [str(row["memory_id"]) for row in rows]
