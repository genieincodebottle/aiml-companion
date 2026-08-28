"""Load Northwind Freight's operational exports.

YOU IMPLEMENT THIS. See tests/test_ingest.py for the contract.

The temptation is `pd.read_csv(path)` and move on. Do that and you will silently
lose rows, and you will not find out until a customer asks why their shipment is
missing from your dashboard. Silent row loss is the single most common way an FDE
loses credibility in week two, because the system looks like it works.

The exports have real defects. Some you will find by reading the files, some only
by counting what came out against what went in. That counting discipline is the
actual lesson here.

The contract the tests enforce:

1. `load_shipments` returns every data row, or explicitly accounts for the ones it
   could not parse. Nothing disappears without a record.
2. Rows that cannot be parsed go into `RejectedRow` entries with a reason and the
   raw line, not into a log line nobody reads and not into an exception that kills
   the run. This is the collect-and-continue pattern: a batch job that dies on row
   4,000 of 4,001 has wasted the whole run.
3. The 2024 and 2025 exports unify into one schema. They do not have the same
   columns, and the join key does not have the same type in both.
4. Nulls arrive in several disguises. They normalise to one representation.

None of that is difficult. All of it is the job.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RejectedRow:
    """A row that could not be parsed, kept rather than dropped.

    `raw` is the original line so a human can look at it. `reason` is short and
    specific: "6 fields, header has 5" beats "parse error".
    """

    source: str
    line_number: int
    raw: str
    reason: str


@dataclass
class IngestResult:
    """What came out of an ingest run.

    `rows_in` is what the source contained. `rows_out` plus `len(rejected)` must
    equal it. If that identity does not hold, you lost data, and the test says so.
    """

    records: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[RejectedRow] = field(default_factory=list)
    rows_in: int = 0

    @property
    def rows_out(self) -> int:
        return len(self.records)

    def reconciles(self) -> bool:
        """The identity that makes silent loss impossible."""
        return self.rows_in == self.rows_out + len(self.rejected)


def load_shipments(path: Path) -> IngestResult:
    """Parse one shipments export into normalised records.

    Args:
        path: a CSV under customer/data/.

    Returns:
        IngestResult where `reconciles()` is True.

    Notes for the implementer:
        - Count the source rows BEFORE parsing, or you cannot prove reconciliation.
        - A quoted field may contain the delimiter. The stdlib `csv` module already
          handles this correctly; hand-rolled `line.split(",")` does not.
        - One file starts with a byte-order mark. `encoding="utf-8-sig"` exists for
          exactly this and costs nothing to use.
        - Some rows are not valid utf-8. Decide what you do, and be able to say why
          in a demo: reject them, or decode with a fallback and flag them.
        - Rows with the wrong field count are the interesting ones. Rejecting them
          is defensible. Silently truncating them is not.
    """
    raise NotImplementedError(
        "Implement load_shipments. Run `pytest tests/test_ingest.py -v` to see the contract."
    )


def unify(results: list[IngestResult]) -> list[dict[str, Any]]:
    """Merge exports from different years into one schema.

    The 2024 and 2025 files disagree about the name of the customer column and
    about its type. A naive concat produces a frame with both columns half-populated
    and a join that quietly matches nothing.

    Decide the canonical schema, convert into it, and be prepared to defend the
    choice. There is no single right answer, only answers you can explain.
    """
    raise NotImplementedError(
        "Implement unify. See tests/test_ingest.py::test_unify_preserves_every_record."
    )
