from __future__ import annotations

# Shim module required by `openenv validate` in repo mode.
# The actual implementation lives in `openenv_submission.server.app`.

from openenv_submission.server.app import app, main as _main  # re-export


def main() -> None:
    _main()


if __name__ == "__main__":
    main()
