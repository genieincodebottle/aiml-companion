"""Tenant scoping.

The post's rule is not "filter the results", it is "put the tenant id inside the
search query". These tests are written to fail if anyone ever weakens that,
including by adding a parameter that widens the scope for convenience.
"""

from __future__ import annotations

import inspect

import pytest

from app.tools import policy_index
from app.tools.catalog import get_order, list_orders

NORTHWIND = "tenant-northwind"
CONTOSO = "tenant-contoso"


def test_contoso_passage_exists_so_the_test_means_something():
    """An empty result from an empty index proves nothing. The other tenant's
    data has to actually be in the corpus for isolation to be testable."""
    assert any(p.tenant_id == CONTOSO for p in policy_index.CORPUS)


@pytest.mark.parametrize(
    "query",
    [
        "refund window",
        "return policy",
        # Asking for the other tenant by name.
        "Contoso returns policy",
        # Asking for the exact text of the other tenant's private note.
        "Contoso internal margin note fraud desk 300 USD",
        "90 days return window all retailers",
    ],
)
def test_no_cross_tenant_passage_is_ever_retrieved(query):
    result = policy_index.search(tenant_id=NORTHWIND, query=query)
    for passage in result["passages"]:
        assert passage["chunk_id"].startswith("nw-"), (
            f"cross-tenant passage {passage['chunk_id']} reached a "
            f"{NORTHWIND} query"
        )
    assert result["tenant_id"] == NORTHWIND


def test_scope_is_not_a_parameter_a_caller_can_widen():
    """There must be no argument that lets a caller search another tenant. The
    scope comes from the request envelope, not from anything the query can say."""
    params = set(inspect.signature(policy_index.search).parameters)
    for forbidden in {"force_tenant", "all_tenants", "tenant_ids", "skip_scope"}:
        assert forbidden not in params


def test_search_never_loads_the_other_tenant_before_filtering():
    """Scoping late is a breach whether or not the text reaches the answer. The
    source of `search` must scope the candidate list before it ranks anything."""
    src = inspect.getsource(policy_index.search)
    scope_line = src.index("p.tenant_id == scoped_tenant")
    rank_line = src.index("_score(query, p)")
    assert scope_line < rank_line


@pytest.mark.asyncio
async def test_sql_tools_are_scoped_in_the_where_clause():
    other_tenants_order = await get_order(NORTHWIND, "ORD-7777")
    assert other_tenants_order.get("error") == "not_found"

    mine = await list_orders(NORTHWIND, "cust-1001")
    assert mine and all(row["tenant_id"] == NORTHWIND for row in mine)
