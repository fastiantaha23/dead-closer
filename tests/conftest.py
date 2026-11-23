import os
import sys
from datetime import datetime, timezone

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deadcloser.config import Settings  # noqa: E402
from deadcloser.engine import DeadCloser  # noqa: E402

# Monday 2026-11-23 10:00 in the default tenant timezone (UTC-6)
T0 = datetime(2026, 11, 23, 16, 0, tzinfo=timezone.utc)


@pytest.fixture()
def dc():
    return DeadCloser(Settings(db_path=":memory:", checkpoint_path=":memory:", llm_backend="rules"))


@pytest.fixture()
def realty(dc):
    tid, key = dc.create_tenant("Skyline Realty", "real_estate")
    return tid


@pytest.fixture()
def motors(dc):
    tid, key = dc.create_tenant("Northside Motors", "auto")
    return tid
