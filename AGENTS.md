# Development conventions

Use Python/NumPy for initial numerical implementations. Keep reusable algorithms
in src/nonlinear_workbook and chapter demonstrations in examples. Cite printed
book pages in correction notes; document departures from the book. Preserve
explicit seeds, units, parameter conventions, and synthetic reference checks.
Run `python -m unittest discover -s tests -v` with the package installed or
PYTHONPATH=src. Do not commit the source PDF or generated environments/outputs.
Do not infer deterministic chaos or trading usefulness from a single statistic.
