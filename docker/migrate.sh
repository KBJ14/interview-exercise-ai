#!/bin/sh
# Apply every migrations/*.sql file in file-name order and stop at the first error.
# Connection settings come from libpq environment variables (PGHOST, PGPORT, PGUSER,
# PGPASSWORD, PGDATABASE) set by docker-compose; nothing here prints them.
#
# There is no table recording which migrations ran: each migration must be safe to re-run
# (001 checks its index definition and is a no-op when it already matches).
set -eu

dir="${MIGRATIONS_DIR:-/migrations}"
applied=0

for file in "$dir"/*.sql; do
    [ -e "$file" ] || continue  # no match: the glob stays literal
    echo "migrate: applying $(basename "$file")"
    # ON_ERROR_STOP makes psql exit non-zero on the first failing statement, so set -e stops here.
    psql -X -q -v ON_ERROR_STOP=1 -f "$file"
    applied=$((applied + 1))
done

echo "migrate: done ($applied file(s))"
