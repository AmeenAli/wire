"""Command-line scripts: --help, dry runs and answer parsing."""

import importlib.util
import os
import subprocess
import sys

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PY_SCRIPTS = ["scripts/answer.py", "scripts/export_release.py",
              "scripts/data/build_llavavideo.py", "scripts/paper/rescore_mlvu.py",
              "eval/videomme/run.py", "eval/lmms/check_format.py", "reproduce/cpu_check.py"]
SH_SCRIPTS = ["scripts/train.sh", "reproduce/main_results.sh", "reproduce/ablations.sh",
              "eval/lmms/run_lmms.sh", "eval/videomme/download.sh"]


def run(args, **env):
    return subprocess.run(args, cwd=REPO, capture_output=True, text=True,
                          env={**os.environ, **env}, timeout=120)


def load(rel):
    spec = importlib.util.spec_from_file_location(os.path.basename(rel)[:-3], os.path.join(REPO, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("script", PY_SCRIPTS)
def test_help(script):
    r = run([sys.executable, script, "--help"])
    assert r.returncode == 0, r.stderr
    assert "usage" in r.stdout


def test_train_module_help():
    pytest.importorskip("transformers")
    r = run([sys.executable, "-m", "wire.train.train", "--help"])
    assert r.returncode == 0, r.stderr


@pytest.mark.parametrize("script", SH_SCRIPTS)
def test_shell_syntax(script):
    assert run(["bash", "-n", script]).returncode == 0


def test_train_dry_run():
    r = run(["bash", "scripts/train.sh", "16"], DRY_RUN="1", NGPU="6")
    assert r.returncode == 0, r.stderr
    assert "--k 16" in r.stdout and "--stop-at-step 5400" in r.stdout
    assert "--nproc_per_node 6" in r.stdout
    assert run(["bash", "scripts/train.sh", "10"], DRY_RUN="1").returncode == 2


def test_reproduce_dry_runs():
    r = run(["bash", "reproduce/main_results.sh"], DRY_RUN="1")
    assert r.returncode == 0, r.stderr
    lines = [x for x in r.stdout.splitlines() if x.startswith("+ ")]
    assert len(lines) == 4 + 4 * 8                      # Video-MME and eight lmms-eval tasks
    r = run(["bash", "reproduce/ablations.sh"], DRY_RUN="1")
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert "--model pool16" in out and "--model stock" in out and "--condition no_update" in out


def test_videomme_answer_extraction():
    vm = load("eval/videomme/run.py")
    assert vm.extract_answer_letter("The best answer is: C") == "C"
    assert vm.extract_answer_letter("B. A dog walks in.") == "B"
    assert vm.extract_answer_letter("") == ""
    prompt = vm.build_prompt("What happens?", ["A. x", "B. y"])
    assert prompt.endswith("A. x\nB. y\nThe best answer is:")


def test_mlvu_rescoring_matches_the_patch():
    rescore = load("scripts/paper/rescore_mlvu.py")
    assert rescore.extract("D.") == "D"
    assert rescore.extract("(b) the second") == "B"
    assert rescore.extract("A) yes") == "A"
    patch = open(os.path.join(REPO, "eval/lmms/patches/mlvu_answer_letter.patch")).read()
    for pattern in (r"^\(?\s*([A-Za-z])\s*[\.\):,\-]", r"^\(?\s*([A-Za-z])\s*$", r"\b([A-Ha-h])\b"):
        assert pattern in patch


def test_format_check_counts_letters():
    fmt = load("eval/lmms/check_format.py")
    assert fmt.LETTER.match("B") and fmt.LETTER.match("(c)") and fmt.LETTER_PREFIX.match("D. dog")
    assert not fmt.LETTER.match("The dog runs")


@pytest.mark.parametrize("script", ["scripts/train.sh", "reproduce/main_results.sh", "reproduce/ablations.sh"])
def test_shell_help_prints_the_header_and_runs_nothing(script):
    r = run(["bash", script, "--help"])
    assert r.returncode == 0, r.stderr
    assert "bash " + script in r.stdout and "+ " not in r.stdout


def test_reproduce_scripts_reject_arguments():
    for script in ("reproduce/main_results.sh", "reproduce/ablations.sh"):
        assert run(["bash", script, "wire-k16"], DRY_RUN="1").returncode == 2


@pytest.mark.parametrize("script", PY_SCRIPTS + ["wire/train/train.py"])
def test_help_keeps_example_line_breaks(script):
    import ast
    src = open(os.path.join(REPO, script)).read()
    doc = ast.get_source_segment(src, ast.parse(src).body[0])
    assert "\\\n" not in doc or doc.lstrip().startswith('r"""'), "a backslash at a line end joins lines in --help"
