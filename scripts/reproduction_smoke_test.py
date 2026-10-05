from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(name: str, command: list[str], cwd: Path | None = None) -> bool:
    print(f"\n{'=' * 70}")
    print(f"[TEST] {name}")
    print("$", " ".join(map(str, command)))

    result = subprocess.run(
        command,
        cwd=cwd or ROOT,
        text=True,
    )

    if result.returncode == 0:
        print(f"[PASS] {name}")
        return True

    print(f"[FAIL] {name} (exit={result.returncode})")
    return False


def main() -> int:
    print("JEV-Star reproduction smoke test")
    print("Repository:", ROOT)
    print("Python:", sys.version.replace("\n", " "))
    print("Platform:", sys.platform)

    expected = [
        ROOT / "jev_star.py",
        ROOT / "macro" / "requirements.txt",
        ROOT / "micro" / "requirements.txt",
        ROOT / "macro" / "tests",
        ROOT / "micro" / "tests",
    ]

    print("\nRepository files:")
    missing = False

    for path in expected:
        ok = path.exists()
        print(f"{'OK' if ok else 'MISSING':8} {path.relative_to(ROOT)}")
        missing |= not ok

    macro_maps = list((ROOT / "macro" / "Maps").glob("*.SC2Map"))
    micro_maps = list((ROOT / "micro" / "SMAC_HARD_maps").glob("*.SC2Map"))

    print(f"\nMacro maps: {len(macro_maps)} (expected 6)")
    print(f"Micro maps: {len(micro_maps)} (expected 35)")

    macro_python = ROOT / ".venvs" / "macro" / (
        "Scripts/python.exe" if os.name == "nt" else "bin/python"
    )
    micro_python = ROOT / ".venvs" / "micro" / (
        "Scripts/python.exe" if os.name == "nt" else "bin/python"
    )

    print("\nEnvironments:")
    print("macro:", macro_python, macro_python.is_file())
    print("micro:", micro_python, micro_python.is_file())

    if missing or len(macro_maps) != 6 or len(micro_maps) != 35:
        print("\nRepository structure check failed.")
        return 1

    if not macro_python.is_file() or not micro_python.is_file():
        print("\nCreate environments first:")
        print("  python scripts/setup_environment.py macro")
        print("  python scripts/setup_environment.py micro")
        return 1

    tests = []

    tests.append(run(
        "macro pip check",
        [str(macro_python), "-m", "pip", "check"],
    ))

    tests.append(run(
        "micro pip check",
        [str(micro_python), "-m", "pip", "check"],
    ))

    macro_env = os.environ.copy()
    macro_env["PYTHONPATH"] = str(ROOT / "macro")

    tests.append(run(
        "macro unit tests",
        [
            str(macro_python),
            "-B",
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests",
            "-p",
            "test_jev*.py",
        ],
        cwd=ROOT / "macro",
    ))

    tests.append(run(
        "micro unit tests",
        [
            str(micro_python),
            "-B",
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests",
            "-p",
            "test_jev*.py",
        ],
        cwd=ROOT / "micro",
    ))

    tests.append(run(
        "macro CLI",
        [sys.executable, str(ROOT / "jev_star.py"), "macro", "--help"],
    ))

    tests.append(run(
        "micro CLI",
        [sys.executable, str(ROOT / "jev_star.py"), "micro", "--help"],
    ))

    sc2_path = os.environ.get("SC2PATH")

    print("\nSC2 integration:")
    if sc2_path:
        path = Path(sc2_path)
        print("SC2PATH =", path)
        print("exists  =", path.is_dir())
        print("Versions=", (path / "Versions").is_dir())
        print("Maps    =", (path / "Maps").is_dir())
    else:
        print("SC2PATH is not configured; live-game tests intentionally skipped.")

    print("\nCredentials:")
    print("TYPESAFE_API_KEY:",
          "configured" if os.environ.get("TYPESAFE_API_KEY") else "not configured")

    passed = sum(tests)
    total = len(tests)

    print(f"\nResult: {passed}/{total} automated checks passed.")

    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())