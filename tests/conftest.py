import pytest


def pytest_addoption(parser):
    parser.addoption("--db", action="store_true", default=False)


def pytest_configure(config):
    config.addinivalue_line("markers", "db: requires live DB tunnel")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--db"):
        skip = pytest.mark.skip(reason="pass --db to run")
        for item in items:
            if "db" in item.keywords:
                item.add_marker(skip)


def pytest_generate_tests(metafunc):
    if "fit_label" in metafunc.fixturenames:
        if metafunc.config.getoption("--db"):
            from pdgfits.query import all_fits
            fits_df = all_fits()
            labels = [
                row["label"]
                for _, row in fits_df.iterrows()
                if row["algorithm"] != "IGNORE"
                and row["label"] != "tauhflav"
            ]
            metafunc.parametrize("fit_label", labels)
        else:
            metafunc.parametrize("fit_label", ["__no_db__"])
