import pytest
from pseudo8051.prototypes import PROTOTYPES


@pytest.fixture(autouse=True)
def clean_prototypes():
    PROTOTYPES.clear()
    yield
    PROTOTYPES.clear()
