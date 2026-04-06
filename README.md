# Code Layout

Top-level folders:
- `app`: all runnable Python code (DGP, train/eval, runners, shared helpers)
- `data`: input and generated data
- `out`: models and result artifacts
- `기타`: archived/non-core materials

Main entry points:
- `app/run_sweep_tables.py`
- `app/run_nmes_table.py`
- `app/build_tables_from_results.py`

Core method note:
- EIPM is model-free in this code path: it directly learns `log w_theta(x,t)` and uses checkpoint model outputs at evaluation time.
