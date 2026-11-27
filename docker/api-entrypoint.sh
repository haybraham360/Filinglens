#!/bin/sh
# If SEED_URL is set, load the pre-ingested corpus before serving: data files are always
# re-extracted (ephemeral filesystems), tables only when the database is empty.
set -e
if [ -n "${SEED_URL:-}" ]; then
  echo "seeding from SEED_URL"
  filinglens seed-import --url "$SEED_URL" --if-empty
fi
exec "$@"
