# Postgres 16 + pgvector with the FilingLens schema applied on first start.
FROM pgvector/pgvector:pg16
COPY src/filinglens/db/schema.sql /docker-entrypoint-initdb.d/001_schema.sql
