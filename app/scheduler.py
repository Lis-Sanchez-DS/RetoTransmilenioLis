import time
import os

from app.collector import collect_new_data

INTERVAL_SECONDS = int(os.getenv("COLLECTOR_INTERVAL_SECONDS", "7200"))


def run() -> None:
    while True:
        result = collect_new_data()
        print(
            f"Collector run: {result['collected']} observations in {result['pages']} pages",
            flush=True,
        )
        time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    run()
