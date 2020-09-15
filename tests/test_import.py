import sys

import pytest


def test_import():
    if sys.version_info.major == 2:
        import physbo
    else:
        with pytest.raises(ImportError):
            import physbo
