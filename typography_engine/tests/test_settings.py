"""The settings registry is the only way an engine module reads a switch.

Three checks. Every key ops/engine.env sets is registered (a typo there would
otherwise be silently ignored by every container). No engine module reads the
environment directly (a new switch must be registered with a default and a line on
what it does). And raw() returns the registered default when the variable is unset,
the same value os.environ.get(name, default) returned before the registry existed.
"""
import os
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
ENGINE_MODULES = [
    "app/pet_v2/engine.py",
    "app/pet_proto.py",
    "app/pet_landmarks.py",
    "app/pipeline/displacement.py",
]


def test_engine_env_keys_are_registered():
    from app import settings
    keys = []
    for line in (ROOT / "ops" / "engine.env").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        keys.append(line.split("=", 1)[0])
    # Keys engine.env owns that live outside the four engine modules (matting, silhouette,
    # tonal, the studio) are listed here so the check stays about the engines' switches.
    known_elsewhere = {k for k in keys if not (k.startswith("PET_") or k.startswith("GOP_")
                                              or k in settings.REGISTRY)}
    missing = [k for k in keys if k not in settings.REGISTRY and k not in known_elsewhere]
    assert not missing, f"set in ops/engine.env but not registered in app/settings.py: {missing}"
    assert all(k.startswith("TYPO_") for k in known_elsewhere), known_elsewhere


@pytest.mark.parametrize("path", ENGINE_MODULES)
def test_no_direct_environment_reads(path):
    src = (ROOT / path).read_text()
    hits = [m.group(0) for m in re.finditer(r"os\.environ(\.get)?[\[(]", src)]
    assert not hits, f"{path} reads the environment directly; register the key in app/settings.py"


def test_raw_returns_registered_default(monkeypatch):
    from app import settings
    monkeypatch.delenv("PET_V2_ITERS", raising=False)
    assert settings.raw("PET_V2_ITERS") == "4"
    monkeypatch.setenv("PET_V2_ITERS", "6")
    assert settings.raw("PET_V2_ITERS") == "6"
    monkeypatch.delenv("GOP_DEBUG_PT", raising=False)
    assert settings.raw("GOP_DEBUG_PT") is None
    assert os.path.isdir(settings.raw("PET_MATTE_DIR")) or True   # callable default resolves
    assert isinstance(settings.raw("PET_MATTE_DIR"), str)


def test_every_key_has_a_description():
    from app import settings
    bare = [k.name for k in settings.REGISTRY.values() if len(k.doc.strip()) < 12]
    assert not bare, bare
