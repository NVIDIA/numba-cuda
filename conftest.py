import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--dump-failed-filechecks",
        action="store_true",
        help="Dump reproducers for FileCheck tests that fail.",
    )


@pytest.fixture(scope="class")
def initialize_from_pytest_config(request):
    """
    Fixture to initialize the test case with pytest configuration options.
    """
    request.cls._dump_failed_filechecks = request.config.getoption(
        "dump_failed_filechecks"
    )
