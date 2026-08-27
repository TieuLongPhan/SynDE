# Repository scripts

Run these entrypoints from any working directory:

- `lint.sh`: Python file-size, docstring-style, and Flake8 checks;
- `pytest.sh`: the full test suite, or supplied pytest arguments;
- `build_doc.sh`: a strict Sphinx HTML documentation build;
- `check_package_artifacts.py`: inspect a built wheel/source archive and
  smoke-test the wheel in a temporary environment.

The remaining Python and JSON files support these entrypoints. They are kept
here, rather than at the repository root, to separate package code from
repository maintenance tools.
