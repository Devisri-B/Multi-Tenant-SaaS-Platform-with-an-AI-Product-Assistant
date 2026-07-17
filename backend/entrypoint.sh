#!/usr/bin/env sh
# Wait for Postgres, apply migrations, then start the API.
set -e

echo "Waiting for the database at ${POSTGRES_HOST:-db}:${POSTGRES_PORT:-5432}…"
until python -c "
import os, socket, sys
host = os.environ.get('POSTGRES_HOST', 'db')
port = int(os.environ.get('POSTGRES_PORT', '5432'))
sock = socket.socket()
sock.settimeout(2)
try:
    sock.connect((host, port))
except OSError:
    sys.exit(1)
finally:
    sock.close()
"; do
  sleep 1
done

echo "Applying migrations…"
alembic upgrade head

exec "$@"
