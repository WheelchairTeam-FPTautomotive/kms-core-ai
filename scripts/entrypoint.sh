#!/bin/bash
set -e

# Runtime permission fix for bind-mounted host directories.
# The container starts as root briefly so it can ensure the non-root
# app user owns the runtime data, log, and output directories, then it
# drops privileges and runs the application process.

chown -R appuser:appuser /app/data /app/logs /app/outputs

exec gosu appuser "$@"
