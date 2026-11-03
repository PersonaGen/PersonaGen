# Contributing

Use Python 3.10 or newer, install with `python -m pip install -e .`, and run `python -m unittest discover -s tests -v`. Changes to scientific metrics should include a small numerical reference test and explain any change in the metric definition. Keep generated outputs, API keys, downloaded models, and private datasets out of commits.

For a bug report, include the command, redacted configuration, Python/package versions, and a minimal input that can be shared. Do not attach participant-level data without permission. For a proposed experiment or method extension, distinguish reproduced measurements from new results and include seeds and model/checkpoint identifiers.

The package namespace remains `popalign` for compatibility; both `personagen` and `popalign` invoke the same CLI.
