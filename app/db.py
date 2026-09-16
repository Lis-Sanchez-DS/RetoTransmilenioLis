import os

import psycopg


def connection() -> psycopg.Connection:
    return psycopg.connect(os.environ["DATABASE_URL"])
