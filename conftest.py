import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--skip-cuda",
        action="store_true",
        default=False,
        help="Skip all tests marked with @pytest.mark.cuda",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--skip-cuda"):
        skip_cuda = pytest.mark.skip(reason="--skip-cuda flag set")
        for item in items:
            if "cuda" in item.keywords:
                item.add_marker(skip_cuda)
