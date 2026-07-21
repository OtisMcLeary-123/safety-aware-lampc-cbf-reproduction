# Archived stage tests

Tests for concluded research stages whose modules remain in
`src/lampc_cbf/` as reproducibility evidence. Pytest still discovers and
runs everything here (`testpaths = ["tests"]` recurses), so these must stay
green — they are archived only to keep the top-level `tests/` directory
limited to the active surface. If a module here is ever deleted from
`src/`, delete its test with it in the same commit.
