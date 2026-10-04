"""Ejecuta el bloque bash REAL de submissions.yml / collector.yml contra un `python`
falso que devuelve codigos de salida programados, para verificar la logica del loop
(2026-10-04: un 75 reintentado para siempre dejo pasar 2 ciclos)."""

import os
import re
import stat
import subprocess
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def loop_script(workflow: str, step_prefix: str) -> str:
    lines = (ROOT / ".github" / "workflows" / workflow).read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if step_prefix in line)
    run = next(i for i in range(start, len(lines)) if lines[i].strip() == "run: |")
    block = []
    for line in lines[run + 1 :]:
        if line.strip() and not line.startswith("          "):
            break
        block.append(line)
    return textwrap.dedent("\n".join(block))


def run_loop(tmp_path, workflow, step_prefix, statuses, seconds=5):
    script = loop_script(workflow, step_prefix).replace("345 * 60", str(seconds))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    counter = tmp_path / "calls"
    counter.write_text("0")
    plan = " ".join(str(s) for s in statuses)
    fake_python = bin_dir / "python"
    fake_python.write_text(
        f"""#!/bin/bash
n=$(cat {counter}); n=$((n+1)); echo $n > {counter}
plan=({plan})
idx=$((n-1)); last=$(( ${{#plan[@]}} - 1 ))
[ $idx -gt $last ] && idx=$last
exit ${{plan[$idx]}}
"""
    )
    fake_sleep = bin_dir / "sleep"
    fake_sleep.write_text("#!/bin/bash\nexit 0\n")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    fake_models = scripts / "download_models.sh"
    fake_models.write_text("#!/bin/bash\nexit 0\n")
    for path in (fake_python, fake_sleep, fake_models):
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
    result = subprocess.run(
        ["bash", "-c", script],
        cwd=tmp_path,
        env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    return result, int(counter.read_text())


SUBMIT = ("submissions.yml", "Enviar submission cada 5 minutos")
COLLECT = ("collector.yml", "Ejecutar collector cada 30 minutos")


@pytest.mark.parametrize("target", [SUBMIT, COLLECT])
def test_persistent_server_outage_stops_the_job_after_three_rounds(tmp_path, target):
    result, calls = run_loop(tmp_path, *target, statuses=[75])
    assert result.returncode == 1 and calls == 3
    assert "::error::" in result.stdout


@pytest.mark.parametrize("target", [SUBMIT, COLLECT])
def test_a_recovered_server_resets_the_outage_counter(tmp_path, target):
    result, calls = run_loop(tmp_path, *target, statuses=[75, 75, 0, 75, 75, 0])
    assert result.returncode == 0  # nunca 3 seguidas
    assert calls > 6


@pytest.mark.parametrize("target", [SUBMIT, COLLECT])
def test_three_real_failures_in_a_row_still_stop_the_job(tmp_path, target):
    result, calls = run_loop(tmp_path, *target, statuses=[1])
    assert result.returncode == 1 and calls == 3


def test_a_timeout_hang_counts_as_a_failure(tmp_path):
    result, calls = run_loop(tmp_path, *SUBMIT, statuses=[124])
    assert result.returncode == 1 and calls == 3


def test_a_stale_collector_warning_never_kills_the_submitter(tmp_path):
    result, calls = run_loop(tmp_path, *SUBMIT, statuses=[76])
    assert result.returncode == 0 and calls >= 1


def test_loop_blocks_use_a_per_run_timeout():
    for workflow, prefix in (SUBMIT, COLLECT):
        assert re.search(r"timeout \d+ python -m app\.", loop_script(workflow, prefix))


def test_submitter_runs_two_replicas_on_separate_runners():
    text = (ROOT / ".github" / "workflows" / "submissions.yml").read_text()
    assert "replica: [1, 2]" in text and "fail-fast: false" in text
    assert "REPLICA: ${{ matrix.replica }}" in text
    assert 'if [ "$REPLICA" = "2" ]; then sleep 150; fi' in text  # desfase entre replicas


def test_pulso_calls_use_a_short_connect_timeout():
    source = (ROOT / "app" / "submit_xgboost.py").read_text()
    assert "HTTP_TIMEOUT = (10, 30)" in source
    assert "timeout=30" not in source  # ninguna llamada a Pulso con el timeout largo de antes


WATCH = ("watchdog.yml", "Vigilar ciclos mientras el job siga vivo")


def run_watchdog(tmp_path, needed_sequence, seconds=4):
    """Corre el bloque real del vigilante con `python` falso: cycle_watch.py escribe
    las lineas programadas en $GITHUB_OUTPUT y `--until-delivered` sale 0."""
    script = loop_script(*WATCH).replace("345 * 60", str(seconds))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "log"
    log.write_text("")
    counter = tmp_path / "n"
    counter.write_text("0")
    seq = " ".join(needed_sequence)
    fake_python = bin_dir / "python"
    fake_python.write_text(
        f"""#!/bin/bash
echo "$@" >> {log}
if [ "$1" = "scripts/cycle_watch.py" ]; then
  if [ "$2" = "--until-delivered" ]; then exit 0; fi
  n=$(cat {counter}); n=$((n+1)); echo $n > {counter}
  plan=({seq}); idx=$((n-1)); last=$(( ${{#plan[@]}} - 1 )); [ $idx -gt $last ] && idx=$last
  if [ "${{plan[$idx]}}" = "true" ]; then
    echo "needed=true" >> "$GITHUB_OUTPUT"; echo "cycle_id=cyc_x" >> "$GITHUB_OUTPUT"
    echo "closes_at=$(date -u -d '+10 minutes' +%Y-%m-%dT%H:%M:%SZ)" >> "$GITHUB_OUTPUT"
  else
    echo "needed=false" >> "$GITHUB_OUTPUT"
  fi
fi
exit 0
"""
    )
    for name in ("sleep", "pip", "timeout"):
        path = bin_dir / name
        path.write_text('#!/bin/bash\n[ "$(basename $0)" = "timeout" ] && shift && exec "$@"\nexit 0\n')
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "download_models.sh").write_text("#!/bin/bash\nexit 0\n")
    for path in list(bin_dir.iterdir()) + [scripts / "download_models.sh"]:
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
    result = subprocess.run(
        ["bash", "-c", script],
        cwd=tmp_path,
        env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "RUNNER_TEMP": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    return result, log.read_text()


def test_watchdog_does_nothing_and_installs_nothing_while_the_submitter_delivers(tmp_path):
    result, calls = run_watchdog(tmp_path, ["false"])
    assert result.returncode == 0
    assert "app.submit_xgboost" not in calls and "-r requirements.txt" not in calls


def test_watchdog_steps_in_prepares_once_and_ends_red(tmp_path):
    result, calls = run_watchdog(tmp_path, ["true", "false"])
    assert "-m app.submit_xgboost" in calls  # entrego el ciclo
    assert result.returncode == 1  # termina en rojo al final: el submitter fallo
    assert "::error::" in result.stdout
