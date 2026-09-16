"""The package installs, imports, and reports a version.

Thin, but it catches the class of breakage that is otherwise only discovered
by a user: a wrong `packages` entry in pyproject, a missing `__init__`, or a
src-layout mistake that makes the working directory shadow the installed
package.
"""

from __future__ import annotations

import re

import evalbench


def test_the_package_imports():
    assert evalbench is not None


def test_a_version_is_reported():
    assert isinstance(evalbench.__version__, str)
    assert evalbench.__version__


def test_the_version_is_pep440_shaped():
    assert re.fullmatch(r"\d+\.\d+\.\d+(\.(dev|a|b|rc)\d+)?", evalbench.__version__)
