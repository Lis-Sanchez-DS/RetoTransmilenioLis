import os

import psycopg

# Sin connect_timeout libpq puede quedarse colgado minutos en una red rota y el job
# parece "vivo" sin enviar nada (2026-10-04: no volver a tener paradas silenciosas).
CONNECT_TIMEOUT_SECONDS = 15


def connection() -> psycopg.Connection:
    return psycopg.connect(os.environ["DATABASE_URL"], connect_timeout=CONNECT_TIMEOUT_SECONDS)
