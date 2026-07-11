"""Allow ``python -m lampc_cbf`` to invoke the dry-run CLI."""

from lampc_cbf.cli import main

raise SystemExit(main())

