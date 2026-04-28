# mypythonlibrary

A minimal example Python library using the **src/** layout, tested with **pytest**.

## Quickstart
```powershell
# from the project root (where this README lives)
pip install -e ".[dev]"
pytest -q
python -c "import mypythonlib as m; print(m.__version__)"
```
