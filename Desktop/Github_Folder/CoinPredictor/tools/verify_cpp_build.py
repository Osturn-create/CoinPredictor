#!/usr/bin/env python3
"""Discover a C++ toolchain and verify CoinPredictor C++ targets."""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, asdict


@dataclass
class CompilerCandidate:
    compiler_type: str
    path: str
    source: str
    priority: int


def _norm(path: str | None) -> str:
    return os.path.abspath(path) if path else ""


def _which_all(names: list[str], path_env: str | None = None) -> list[str]:
    found = []
    search_path = path_env if path_env is not None else os.environ.get("PATH", "")
    for name in names:
        located = shutil.which(name, path=search_path)
        if located:
            found.append(_norm(located))
    return found


def _candidate(candidate_type: str, path: str, source: str, priority: int) -> CompilerCandidate:
    return CompilerCandidate(candidate_type, _norm(path), source, priority)


def discover_vswhere() -> list[str]:
    paths = []
    direct = shutil.which("vswhere.exe")
    if direct:
        paths.append(direct)
    program_files_x86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    paths.append(os.path.join(program_files_x86, "Microsoft Visual Studio", "Installer", "vswhere.exe"))
    return [path for path in paths if os.path.exists(path)]


def discover_visual_studio_with_vswhere() -> list[CompilerCandidate]:
    candidates: list[CompilerCandidate] = []
    for vswhere in discover_vswhere():
        try:
            result = subprocess.run(
                [
                    vswhere,
                    "-latest",
                    "-products",
                    "*",
                    "-requires",
                    "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                    "-property",
                    "installationPath",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError:
            continue
        install_path = result.stdout.strip().splitlines()[0].strip() if result.stdout.strip() else ""
        if not install_path:
            continue
        for cl_path in glob.glob(os.path.join(install_path, "VC", "Tools", "MSVC", "*", "bin", "Hostx64", "x64", "cl.exe")):
            candidates.append(_candidate("msvc", cl_path, "vswhere:{}".format(vswhere), 10))
        msbuild = os.path.join(install_path, "MSBuild", "Current", "Bin", "MSBuild.exe")
        if os.path.exists(msbuild):
            candidates.append(_candidate("msbuild", msbuild, "vswhere:{}".format(vswhere), 40))
    return candidates


def discover_common_windows_compilers() -> list[CompilerCandidate]:
    patterns = [
        (r"C:\msys64\mingw64\bin\g++.exe", "mingw", "MSYS2", 30),
        (r"C:\msys64\ucrt64\bin\g++.exe", "mingw", "MSYS2 UCRT", 30),
        (r"C:\ProgramData\chocolatey\bin\g++.exe", "mingw", "Chocolatey", 35),
        (os.path.expanduser(r"~\scoop\apps\mingw\current\bin\g++.exe"), "mingw", "Scoop", 35),
        (r"C:\Program Files\LLVM\bin\clang++.exe", "clang", "LLVM", 45),
    ]
    return [_candidate(kind, path, source, priority) for path, kind, source, priority in patterns if os.path.exists(path)]


def discover_compilers(system: str | None = None) -> list[CompilerCandidate]:
    system = system or platform.system()
    candidates: list[CompilerCandidate] = []
    if system == "Windows":
        for path in _which_all(["cl.exe"]):
            candidates.append(_candidate("msvc", path, "PATH", 0))
        candidates.extend(discover_visual_studio_with_vswhere())
        for path in _which_all(["g++.exe", "clang++.exe"]):
            kind = "clang" if "clang" in os.path.basename(path).lower() else "mingw"
            candidates.append(_candidate(kind, path, "PATH", 20))
        candidates.extend(discover_common_windows_compilers())
    else:
        for path in _which_all(["g++", "clang++"]):
            kind = "clang" if "clang" in os.path.basename(path) else "gcc"
            candidates.append(_candidate(kind, path, "PATH", 0 if kind == "gcc" else 10))
    unique: dict[str, CompilerCandidate] = {}
    for candidate in candidates:
        unique.setdefault(candidate.path.lower(), candidate)
    return sorted(unique.values(), key=lambda item: (item.priority, item.path.lower()))


def compiler_flags(candidate: CompilerCandidate) -> list[str]:
    if candidate.compiler_type in ("gcc", "mingw", "clang"):
        return ["-std=c++17", "-Wall", "-Wextra", "-Wpedantic", "-Wshadow", "-Wconversion"]
    if candidate.compiler_type == "msvc":
        return ["/std:c++17", "/W4", "/permissive-"]
    return []


def target_commands(repo_root: Path, candidate: CompilerCandidate, out_dir: Path) -> dict[str, list[str]]:
    compiler = candidate.path
    if candidate.compiler_type == "msvc":
        exe = ".exe"
        return {
            "main": [compiler, *compiler_flags(candidate), "/EHsc", "/Fe:" + str(out_dir / ("coin_predictor" + exe)), str(repo_root / "main.cpp"), str(repo_root / "DataScraper.cpp"), str(repo_root / "RiskAnalyzer.cpp")],
            "datascraper": [compiler, *compiler_flags(candidate), "/EHsc", "/DDATASCRAPER_STANDALONE", "/Fe:" + str(out_dir / ("data_scraper" + exe)), str(repo_root / "DataScraper.cpp")],
            "riskanalyzer": [compiler, *compiler_flags(candidate), "/EHsc", "/DRISK_ANALYZER_STANDALONE", "/Fe:" + str(out_dir / ("risk_analyzer" + exe)), str(repo_root / "RiskAnalyzer.cpp")],
        }
    return {
        "main": [compiler, *compiler_flags(candidate), "-o", str(out_dir / "coin_predictor"), str(repo_root / "main.cpp"), str(repo_root / "DataScraper.cpp"), str(repo_root / "RiskAnalyzer.cpp"), "-lz"],
        "datascraper": [compiler, *compiler_flags(candidate), "-DDATASCRAPER_STANDALONE", str(repo_root / "DataScraper.cpp"), "-o", str(out_dir / "data_scraper"), "-lz"],
        "riskanalyzer": [compiler, *compiler_flags(candidate), "-DRISK_ANALYZER_STANDALONE", str(repo_root / "RiskAnalyzer.cpp"), "-o", str(out_dir / "risk_analyzer")],
    }


def classify_build_failure(output: str) -> list[str]:
    lowered = output.lower()
    missing = []
    if "zlib.h" in lowered or "cannot find -lz" in lowered or "unresolved external" in lowered and "inflate" in lowered:
        missing.append("zlib")
    return missing


def installation_guidance(system: str | None = None) -> list[str]:
    system = system or platform.system()
    if system == "Windows":
        return [
            "Install Visual Studio Build Tools and select 'Desktop development with C++', then reopen Developer PowerShell.",
            "Or install MSYS2, run `pacman -S --needed mingw-w64-ucrt-x86_64-gcc zlib-devel`, and add the UCRT64 bin directory to PATH.",
            "Or install LLVM and zlib development headers, then reopen the shell so PATH is refreshed.",
            "After installation, run `powershell -ExecutionPolicy Bypass -File .\\tools\\verify_cpp_build.ps1`.",
        ]
    if system == "Darwin":
        return [
            "Install Xcode Command Line Tools with `xcode-select --install`.",
            "If zlib headers are missing, install them through the active SDK or Homebrew.",
        ]
    return [
        "Install a compiler with your package manager, for example `sudo apt-get install build-essential zlib1g-dev`.",
        "Alternatively install clang and zlib development headers.",
    ]


def run_builds(repo_root: Path, candidate: CompilerCandidate) -> dict:
    result = {
        "build_attempted": True,
        "chosen_compiler": asdict(candidate),
        "targets": {},
        "warnings": [],
        "missing_dependencies": [],
    }
    with tempfile.TemporaryDirectory(prefix="coinpredictor_cpp_build_") as tmp:
        out_dir = Path(tmp)
        for target, command in target_commands(repo_root, candidate, out_dir).items():
            try:
                completed = subprocess.run(command, cwd=str(repo_root), capture_output=True, text=True, check=False)
            except OSError as error:
                result["targets"][target] = {
                    "passed": False,
                    "returncode": None,
                    "command": command,
                    "stdout": "",
                    "stderr": str(error),
                }
                continue
            output = (completed.stdout or "") + "\n" + (completed.stderr or "")
            for dep in classify_build_failure(output):
                if dep not in result["missing_dependencies"]:
                    result["missing_dependencies"].append(dep)
            result["targets"][target] = {
                "passed": completed.returncode == 0,
                "returncode": completed.returncode,
                "command": command,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
    result["main_build_passed"] = bool(result["targets"].get("main", {}).get("passed"))
    result["datascraper_build_passed"] = bool(result["targets"].get("datascraper", {}).get("passed"))
    result["riskanalyzer_build_passed"] = bool(result["targets"].get("riskanalyzer", {}).get("passed"))
    result["build_passed"] = all(item.get("passed") for item in result["targets"].values()) if result["targets"] else False
    return result


def verify(repo_root: str | os.PathLike[str], build: bool = True) -> dict:
    root = Path(repo_root).resolve()
    candidates = discover_compilers()
    status = {
        "compiler_found": bool(candidates),
        "compiler_type": candidates[0].compiler_type if candidates else "",
        "compiler_path": candidates[0].path if candidates else "",
        "discovered_candidates": [asdict(candidate) for candidate in candidates],
        "build_attempted": False,
        "main_build_passed": False,
        "datascraper_build_passed": False,
        "riskanalyzer_build_passed": False,
        "build_passed": False,
        "warnings": [],
        "missing_dependencies": [],
        "installation_guidance": [] if candidates else installation_guidance(),
    }
    if not candidates:
        status["verification_status"] = "unavailable_no_compiler"
        return status
    if not build:
        status["verification_status"] = "compiler_discovered_build_not_requested"
        return status
    build_status = run_builds(root, candidates[0])
    status.update(build_status)
    status["verification_status"] = "passed" if status["build_passed"] else "failed"
    if status["missing_dependencies"]:
        status["verification_status"] = "failed_missing_dependency"
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--no-build", action="store_true", help="Only discover compilers; do not compile.")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args(argv)
    status = verify(args.repo_root, build=not args.no_build)
    text = json.dumps(status, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(text + "\n", encoding="utf-8")
    if status["verification_status"] == "passed":
        return 0
    if status["verification_status"].startswith("unavailable"):
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
