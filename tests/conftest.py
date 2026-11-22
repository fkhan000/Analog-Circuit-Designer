import pytest

from acd.sim.ngspice import Ngspice, NgspiceNotFound, find_library


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    try:
        find_library()
    except NgspiceNotFound:
        skip = pytest.mark.skip(reason="libngspice not installed")
        for item in items:
            if "ngspice" in item.keywords:
                item.add_marker(skip)


@pytest.fixture(scope="session")
def ngspice() -> Ngspice:
    return Ngspice.get()
