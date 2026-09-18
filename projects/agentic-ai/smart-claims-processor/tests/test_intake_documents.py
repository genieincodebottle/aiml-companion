"""Intake document matching.

Intake used to append every missing required document to the provided list
and hardcode missing_docs = [], so no claim ever had missing documents.
"""

import pytest

from src.agents.intake_agent import find_missing_documents


@pytest.mark.parametrize("required, provided, missing", [
    (["photos"], ["scratch_photo.jpg"], []),
    (["police_report"], ["police_report.pdf"], []),
    (["repair_estimate_authorized_garage"], ["repair_estimate.pdf"], []),
    (["fir_copy"], ["fir.pdf"], []),
    (["other_driver_info"], ["police_report.pdf"], ["other_driver_info"]),
    (["key_affidavit", "proof_of_ownership"], [], ["key_affidavit", "proof_of_ownership"]),
    ([], ["anything.pdf"], []),
])
def test_missing_documents(required, provided, missing):
    assert find_missing_documents(required, provided) == missing
