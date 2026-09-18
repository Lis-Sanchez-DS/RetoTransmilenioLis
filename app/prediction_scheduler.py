"""Scheduler que descubre ciclos y envía submissions XGBoost."""

import os
import time

from app.submit_xgboost import submit_current_cycle


INTERVAL_SECONDS = int(os.getenv("PREDICTION_INTERVAL_SECONDS", "300"))


def run() -> None:
    while True:
        try:
            result = submit_current_cycle()
            if result is None:
                print("No hay un ciclo abierto.", flush=True)
            else:
                print(f"Submission enviada: {result['submission_id']}", flush=True)
        except Exception as exc:
            print(f"Error en prediction scheduler: {exc}", flush=True)
        time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    run()
