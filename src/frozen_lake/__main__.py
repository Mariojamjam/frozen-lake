"""Run the command line as `python -m frozen_lake`, for when `frozen-lake` is not on the PATH."""

from frozen_lake.cli import app

app(prog_name="frozen-lake")
