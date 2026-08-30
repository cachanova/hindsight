"""End-to-end retain coverage for metadata-based LLM routing."""

import dataclasses
import uuid

import pytest

from hindsight_api.config import LLMMetadataRoute, LLMStrategyConfig, _get_raw_config
from hindsight_api.engine import memory_engine as engine_module
from hindsight_api.engine.db_utils import acquire_with_retry
from hindsight_api.engine.llm_wrapper import LLMProvider
from hindsight_api.engine.multi_llm import MultiLLMProvider
from hindsight_api.engine.schema import fq_table


@dataclasses.dataclass
class _CallCounts:
    primary: int = 0
    secondary: int = 0

    def reset(self) -> None:
        self.primary = 0
        self.secondary = 0


def _install_metadata_router(
    memory,
    monkeypatch,
    *,
    operation: str = "retain",
    key: str = "tags",
    value: str = "sensitive",
) -> _CallCounts:
    counts = _CallCounts()
    primary = LLMProvider(provider="mock", api_key="", base_url="", model="primary")
    secondary = LLMProvider(provider="mock", api_key="", base_url="", model="secondary")
    primary_call = primary.call
    secondary_call = secondary.call
    primary_call_with_tools = primary.call_with_tools
    secondary_call_with_tools = secondary.call_with_tools

    async def record_primary(*args, **kwargs):
        counts.primary += 1
        return await primary_call(*args, **kwargs)

    async def record_secondary(*args, **kwargs):
        counts.secondary += 1
        return await secondary_call(*args, **kwargs)

    async def record_primary_with_tools(*args, **kwargs):
        counts.primary += 1
        return await primary_call_with_tools(*args, **kwargs)

    async def record_secondary_with_tools(*args, **kwargs):
        counts.secondary += 1
        return await secondary_call_with_tools(*args, **kwargs)

    monkeypatch.setattr(primary, "call", record_primary)
    monkeypatch.setattr(secondary, "call", record_secondary)
    monkeypatch.setattr(primary, "call_with_tools", record_primary_with_tools)
    monkeypatch.setattr(secondary, "call_with_tools", record_secondary_with_tools)
    setattr(
        memory,
        f"_{operation}_llm_config",
        MultiLLMProvider(
            [primary, secondary],
            LLMStrategyConfig(
                mode="metadata",
                routes=[LLMMetadataRoute(key=key, value=value, member=1)],
            ),
        ),
    )
    return counts


def _install_ambiguous_metadata_router(memory, monkeypatch) -> None:
    members = [
        LLMProvider(provider="mock", api_key="", base_url="", model="primary"),
        LLMProvider(provider="mock", api_key="", base_url="", model="internal"),
        LLMProvider(provider="mock", api_key="", base_url="", model="sensitive"),
    ]

    async def unexpected_call(*args, **kwargs):
        pytest.fail("ambiguous retain reached an LLM provider")

    for member in members:
        monkeypatch.setattr(member, "call", unexpected_call)
        monkeypatch.setattr(member, "call_with_tools", unexpected_call)

    memory._retain_llm_config = MultiLLMProvider(
        members,
        LLMStrategyConfig(
            mode="metadata",
            routes=[
                LLMMetadataRoute(key="tags", value="internal", member=1),
                LLMMetadataRoute(key="tags", value="sensitive", member=2),
            ],
        ),
    )


async def test_http_preserves_explicit_empty_classification(api_client, memory, monkeypatch) -> None:
    captured: list[dict] = []

    async def capture_retain(*args, **kwargs):
        captured.extend(kwargs["contents"])
        return [[]], None

    monkeypatch.setattr(memory, "retain_batch_async", capture_retain)
    response = await api_client.post(
        "/v1/default/banks/metadata-routing-http/memories",
        json={
            "items": [
                {
                    "content": "Explicitly declassified append.",
                    "document_id": "document-1",
                    "update_mode": "append",
                    "tags": [],
                    "metadata": {},
                }
            ]
        },
    )

    assert response.status_code == 200
    assert captured[0]["tags"] == []
    assert captured[0]["metadata"] == {}


async def test_split_sync_retain_rejects_cross_member_classification_before_llm(
    memory_no_llm_verify, request_context, monkeypatch
) -> None:
    _install_ambiguous_metadata_router(memory_no_llm_verify, monkeypatch)
    narrowed = dataclasses.replace(_get_raw_config(), retain_batch_tokens=20)
    monkeypatch.setattr(engine_module, "get_config", lambda: narrowed)
    bank_id = f"metadata-routing-mixed-sync-{uuid.uuid4().hex[:8]}"
    contents = [
        {"content": " ".join(["internal"] * 80), "tags": ["internal"]},
        {"content": " ".join(["sensitive"] * 80), "tags": ["sensitive"]},
    ]

    with pytest.raises(ValueError, match="select multiple members"):
        await memory_no_llm_verify.retain_batch_async(
            bank_id,
            contents,
            request_context=request_context,
        )


async def test_queued_retain_rejects_cross_member_classification_before_children(
    memory_no_llm_verify, request_context, monkeypatch
) -> None:
    _install_ambiguous_metadata_router(memory_no_llm_verify, monkeypatch)
    narrowed = dataclasses.replace(_get_raw_config(), retain_batch_tokens=20)
    monkeypatch.setattr(engine_module, "get_config", lambda: narrowed)
    bank_id = f"metadata-routing-mixed-queued-{uuid.uuid4().hex[:8]}"
    contents = [
        {"content": " ".join(["internal"] * 80), "tags": ["internal"]},
        {"content": " ".join(["sensitive"] * 80), "tags": ["sensitive"]},
    ]

    with pytest.raises(ValueError, match="select multiple members"):
        await memory_no_llm_verify.submit_async_retain(
            bank_id,
            contents,
            request_context=request_context,
        )

    operations = await memory_no_llm_verify.list_operations(bank_id, request_context=request_context)
    assert operations["total"] == 0


async def test_tag_routed_reflect_stays_on_secondary_for_every_scope(
    memory_no_llm_verify, request_context, monkeypatch
) -> None:
    bank_id = f"metadata-routing-reflect-{uuid.uuid4().hex[:8]}"
    await memory_no_llm_verify.retain_batch_async(
        bank_id,
        [{"content": "The restricted launch code is 2468.", "tags": ["sensitive"]}],
        request_context=request_context,
    )
    await memory_no_llm_verify.retain_batch_async(
        bank_id,
        [{"content": "The public office opens at nine.", "tags": ["public"]}],
        request_context=request_context,
    )

    calls = _install_metadata_router(memory_no_llm_verify, monkeypatch, operation="reflect")
    await memory_no_llm_verify.reflect_async(
        bank_id,
        "Summarize the available information.",
        request_context=request_context,
    )
    assert calls.secondary > 0
    assert calls.primary == 0

    # Expand can turn a public fact into its full mixed-classification document,
    # so even an exact public scope must stay on the protected lane.
    calls.reset()
    await memory_no_llm_verify.reflect_async(
        bank_id,
        "When does the office open?",
        tags=["public"],
        tags_match="exact",
        request_context=request_context,
    )
    assert calls.secondary > 0
    assert calls.primary == 0

    calls.reset()
    await memory_no_llm_verify.reflect_async(
        bank_id,
        "What is the launch code?",
        tags=["sensitive"],
        tags_match="exact",
        request_context=request_context,
    )
    assert calls.secondary > 0
    assert calls.primary == 0


async def test_sensitive_pending_facts_route_consolidation_to_secondary(
    memory_no_llm_verify, request_context, monkeypatch
) -> None:
    from hindsight_api.engine.consolidation import consolidator

    calls = _install_metadata_router(memory_no_llm_verify, monkeypatch, operation="consolidation")

    async def capture_run(memory_engine, bank_id, context, config, llm_config, *args):
        await llm_config.call(messages=[{"role": "user", "content": "Sensitive facts"}], scope="consolidation")
        return {"status": "complete"}

    monkeypatch.setattr(consolidator, "_run_consolidation_job", capture_run)
    result = await consolidator.run_consolidation_job(
        memory_no_llm_verify,
        "metadata-routing-consolidation",
        request_context,
    )

    assert result == {"status": "complete"}
    assert calls.secondary == 1
    assert calls.primary == 0


async def test_sensitive_retain_uses_secondary_without_touching_primary(
    memory_no_llm_verify, request_context, monkeypatch
) -> None:
    calls = _install_metadata_router(memory_no_llm_verify, monkeypatch)

    bank_id = f"metadata-routing-{uuid.uuid4().hex[:8]}"
    document_id = "private-account"
    unit_ids = await memory_no_llm_verify.retain_batch_async(
        bank_id,
        [
            {
                "content": "Alice's private account number is 1234.",
                "document_id": document_id,
                "tags": ["sensitive"],
            }
        ],
        request_context=request_context,
    )

    assert unit_ids[0]
    assert calls.secondary > 0
    assert calls.primary == 0

    memories = await memory_no_llm_verify.list_memory_units(
        bank_id,
        fact_type=["world", "experience"],
        tags=["sensitive"],
        tags_match="all_strict",
        request_context=request_context,
    )
    assert memories["total"] == len(unit_ids[0])

    calls.reset()
    narrowed = dataclasses.replace(_get_raw_config(), retain_batch_tokens=20)
    monkeypatch.setattr(engine_module, "get_config", lambda: narrowed)
    sub_batch_count = 0
    real_iter_sub_batches = engine_module.iter_sub_batches

    def count_sub_batches(*args, **kwargs):
        nonlocal sub_batch_count
        for sub_batch in real_iter_sub_batches(*args, **kwargs):
            sub_batch_count += 1
            yield sub_batch

    monkeypatch.setattr(engine_module, "iter_sub_batches", count_sub_batches)
    await memory_no_llm_verify.retain_batch_async(
        bank_id,
        [
            {
                "content": " ".join(
                    f"The private account review entry number {index} was completed today." for index in range(120)
                ),
                "document_id": document_id,
                "update_mode": "append",
            }
        ],
        request_context=request_context,
    )

    assert sub_batch_count > 1
    assert calls.secondary > 0
    assert calls.primary == 0

    # Omitted tags inherit the stored classification, including for later
    # appends after the first append has replaced the document record.
    calls.reset()
    await memory_no_llm_verify.retain_batch_async(
        bank_id,
        [{"content": "A second review was recorded.", "document_id": document_id, "update_mode": "append"}],
        request_context=request_context,
    )
    assert calls.secondary > 0
    assert calls.primary == 0
    document = await memory_no_llm_verify.get_document(document_id, bank_id, request_context=request_context)
    assert document is not None and document["tags"] == ["sensitive"]

    # Adding an unrelated tag must not implicitly remove the classifier.
    calls.reset()
    await memory_no_llm_verify.retain_batch_async(
        bank_id,
        [
            {
                "content": "Customer scope added.",
                "document_id": document_id,
                "update_mode": "append",
                "tags": ["customer"],
            }
        ],
        request_context=request_context,
    )
    assert calls.secondary > 0
    assert calls.primary == 0
    document = await memory_no_llm_verify.get_document(document_id, bank_id, request_context=request_context)
    assert document is not None and set(document["tags"]) == {"customer", "sensitive"}

    # Explicitly clearing tags declassifies future appends, but this operation
    # still reprocesses the old sensitive body on the protected lane.
    calls.reset()
    await memory_no_llm_verify.retain_batch_async(
        bank_id,
        [{"content": "Classification cleared.", "document_id": document_id, "update_mode": "append", "tags": []}],
        request_context=request_context,
    )
    assert calls.secondary > 0
    assert calls.primary == 0

    calls.reset()
    await memory_no_llm_verify.retain_batch_async(
        bank_id,
        [{"content": "A public follow-up.", "document_id": document_id, "update_mode": "append"}],
        request_context=request_context,
    )
    assert calls.primary > 0
    assert calls.secondary == 0


async def test_custom_metadata_is_inherited_for_append_routing(
    memory_no_llm_verify, request_context, monkeypatch
) -> None:
    calls = _install_metadata_router(
        memory_no_llm_verify,
        monkeypatch,
        key="metadata.classification",
        value="restricted",
    )
    bank_id = f"metadata-routing-custom-{uuid.uuid4().hex[:8]}"
    document_id = "restricted-document"

    await memory_no_llm_verify.retain_batch_async(
        bank_id,
        [
            {
                "content": "Restricted source material.",
                "document_id": document_id,
                "metadata": {"classification": "restricted"},
            }
        ],
        request_context=request_context,
    )
    assert calls.secondary > 0
    assert calls.primary == 0

    for content in ("First append without metadata.", "Second append without metadata."):
        calls.reset()
        await memory_no_llm_verify.retain_batch_async(
            bank_id,
            [{"content": content, "document_id": document_id, "update_mode": "append"}],
            request_context=request_context,
        )
        assert calls.secondary > 0
        assert calls.primary == 0

    # Adding an unrelated key retains the stored classifier.
    calls.reset()
    await memory_no_llm_verify.retain_batch_async(
        bank_id,
        [
            {
                "content": "Source metadata added.",
                "document_id": document_id,
                "update_mode": "append",
                "metadata": {"source": "crm"},
            }
        ],
        request_context=request_context,
    )
    assert calls.secondary > 0
    assert calls.primary == 0
    document = await memory_no_llm_verify.get_document(document_id, bank_id, request_context=request_context)
    assert document is not None
    assert document["document_metadata"] == {"classification": "restricted", "source": "crm"}

    # An explicit empty map clears custom routing metadata for later appends.
    calls.reset()
    await memory_no_llm_verify.retain_batch_async(
        bank_id,
        [{"content": "Classification cleared.", "document_id": document_id, "update_mode": "append", "metadata": {}}],
        request_context=request_context,
    )
    assert calls.secondary > 0
    assert calls.primary == 0

    calls.reset()
    await memory_no_llm_verify.retain_batch_async(
        bank_id,
        [{"content": "Public follow-up.", "document_id": document_id, "update_mode": "append"}],
        request_context=request_context,
    )
    assert calls.primary > 0
    assert calls.secondary == 0


async def test_store_owned_sensitive_append_uses_authoritative_document_tags(
    memory_no_llm_verify, request_context, monkeypatch
) -> None:
    from hindsight_api.engine.memories import set_memories
    from tests.test_memories_extension import InMemoryMemories

    class MetadataAwareStore(InMemoryMemories):
        async def get_document_record(self, *, bank_id, document_id, include_text=False):
            record = await super().get_document_record(
                bank_id=bank_id,
                document_id=document_id,
                include_text=include_text,
            )
            if record is not None:
                record["metadata"] = dict(self.documents[document_id]["metadata"])
            return record

    store = MetadataAwareStore({})
    set_memories(store)
    try:

        async def clear_sql_classification(document_id: str) -> None:
            # Deliberately create SQL/store divergence that no public API can
            # express, proving append routing treats the store-owned record as
            # authoritative instead of accidentally passing via the SQL mirror.
            backend = await memory_no_llm_verify._get_backend()
            async with acquire_with_retry(backend) as conn:
                await conn.execute(
                    f"UPDATE {fq_table('documents')} SET tags = $1, retain_params = $2 WHERE id = $3 AND bank_id = $4",
                    [],
                    None,
                    document_id,
                    bank_id,
                )
                row = await conn.fetchrow(
                    f"SELECT tags, retain_params FROM {fq_table('documents')} WHERE id = $1 AND bank_id = $2",
                    document_id,
                    bank_id,
                )
                if row is not None:
                    assert list(conn.parse_json(row["tags"]) or []) == []
                    assert row["retain_params"] is None

        calls = _install_metadata_router(memory_no_llm_verify, monkeypatch)
        bank_id = f"metadata-routing-store-{uuid.uuid4().hex[:8]}"
        document_id = "store-owned-sensitive"
        await memory_no_llm_verify.retain_batch_async(
            bank_id,
            [{"content": "Sensitive store-owned content.", "document_id": document_id, "tags": ["sensitive"]}],
            request_context=request_context,
        )
        assert store.documents[document_id]["tags"] == ["sensitive"]
        await clear_sql_classification(document_id)

        calls.reset()
        await memory_no_llm_verify.retain_batch_async(
            bank_id,
            [{"content": "An append without tags.", "document_id": document_id, "update_mode": "append"}],
            request_context=request_context,
        )
        assert calls.secondary > 0
        assert calls.primary == 0

        calls = _install_metadata_router(
            memory_no_llm_verify,
            monkeypatch,
            key="metadata.classification",
            value="restricted",
        )
        metadata_document_id = "store-owned-restricted"
        await memory_no_llm_verify.retain_batch_async(
            bank_id,
            [
                {
                    "content": "Restricted store-owned content.",
                    "document_id": metadata_document_id,
                    "metadata": {"classification": "restricted"},
                }
            ],
            request_context=request_context,
        )
        await clear_sql_classification(metadata_document_id)

        calls.reset()
        await memory_no_llm_verify.retain_batch_async(
            bank_id,
            [
                {
                    "content": "An append without metadata.",
                    "document_id": metadata_document_id,
                    "update_mode": "append",
                }
            ],
            request_context=request_context,
        )
        assert calls.secondary > 0
        assert calls.primary == 0
    finally:
        set_memories(None)
