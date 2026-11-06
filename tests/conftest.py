import pytest

from caddy.kernel import graph as G
from caddy.kernel.catalogue import rows
from caddy.kernel.family import build_family


def pytest_addoption(parser):
    parser.addoption("--update-golden", action="store_true", help="re-approve golden digests after a reviewed change")


@pytest.fixture(scope="session")
def update_golden(request):
    return request.config.getoption("--update-golden")


@pytest.fixture(scope="session")
def ev():
    return G.Evaluator()


@pytest.fixture(scope="session")
def graph():
    return G.load()


@pytest.fixture(scope="session")
def family(graph, ev):
    return build_family(graph, ev=ev)


ROWS = rows()
