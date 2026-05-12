from __future__ import annotations

import re
from pathlib import Path


MAKEFILE = Path("Makefile")


def _makefile() -> str:
    return MAKEFILE.read_text(encoding="utf-8")


def _target_block(name: str) -> str:
    text = _makefile()
    marker = f"{name}:"
    start = text.index(marker)
    next_target = len(text)
    for line_start in range(start + len(marker), len(text)):
        if text[line_start - 1] != "\n":
            continue
        line = text[line_start : text.find("\n", line_start)]
        if line and not line.startswith("\t") and line[0].isalnum() and ":" in line:
            next_target = line_start
            break
    return text[start:next_target]


def test_existing_dev_targets_are_preserved():
    text = _makefile()

    for target in ("dev", "dev-real", "dev-ollama"):
        assert f"{target}:" in text

    # dev is the internal smoke path - must remain deterministic/scripted
    dev_block = _target_block("dev")
    assert "MOCK_LLM=true" in dev_block
    assert "DEMO_ROUTER=true" in dev_block
    assert "env LLM_PROVIDER=ollama MOCK_LLM=false" in _target_block("dev-ollama")


def test_demo_target_exists_and_uses_remote_llm():
    block = _target_block("demo")

    assert "LLM_PROVIDER=remote" in block
    assert "LLM_PROMPT_PROFILE=demo" in block
    assert "MOCK_LLM=false" in block


def test_demo_target_requires_env_validation():
    block = _target_block("demo")

    # Must check that .env file exists before starting
    assert "test -f .env" in block
    # Must validate all three required remote-provider fields
    assert "LLM_BASE_URL" in block
    assert "LLM_API_KEY" in block
    assert "LLM_MODEL" in block
    # Validation must happen before install
    assert block.index("test -f .env") < block.index("$(MAKE) install")


def test_demo_target_does_not_use_demo_router():
    block = _target_block("demo")

    assert "DEMO_ROUTER=true" not in block


def test_demo_target_enables_rag_with_seed_path():
    block = _target_block("demo")

    assert "MOCK_RAG=true" not in block
    assert "RAG_SEED_PATH=" in block
    assert ".chroma-demo/" in block


def test_demo_local_target_uses_ollama_and_seeded_rag():
    block = _target_block("demo-local")

    assert "LLM_PROVIDER=ollama" in block
    assert "LLM_PROMPT_PROFILE=demo" in block
    assert "MOCK_LLM=false" in block
    assert "DEMO_ROUTER=false" in block
    assert "MOCK_RAG=false" in block
    assert "RAG_SEED_PATH=" in block
    assert ".chroma-demo/" in block


def test_demo_local_target_does_not_require_env():
    block = _target_block("demo-local")

    # demo-local must not gate on .env - it must work without any API key
    assert "test -f .env" not in block
    assert "LLM_API_KEY" not in block


def test_demo_target_forces_policy_profile_demo():
    """C4.3: demo target must force POLICY_PROFILE=demo so consent gates are skipped."""
    block = _target_block("demo")
    assert "POLICY_PROFILE=demo" in block


def test_demo_local_target_forces_policy_profile_demo():
    """C4.3: demo-local must also force POLICY_PROFILE=demo (same product intent)."""
    block = _target_block("demo-local")
    assert "POLICY_PROFILE=demo" in block


def test_readme_quickstart_leads_with_make_demo():
    readme = Path("README.md").read_text(encoding="utf-8")
    quickstart_start = readme.index("## Quickstart")
    next_section = readme.index("\n## ", quickstart_start + 1)
    quickstart = readme[quickstart_start:next_section]

    assert "make demo" in quickstart
    # make dev must not appear as a bare command in the quickstart
    assert not re.search(r"^\s*make dev\s*$", quickstart, re.MULTILINE)


def test_readme_does_not_recommend_paid_providers_as_primary():
    readme = Path("README.md").read_text(encoding="utf-8")
    quickstart_start = readme.index("## Quickstart")
    next_section = readme.index("\n## ", quickstart_start + 1)
    quickstart = readme[quickstart_start:next_section].lower()

    # Paid-only providers must not appear as recommended primary path
    for term in ("openai.com/v1", "api.anthropic.com", "billing-required"):
        assert term not in quickstart, f"Forbidden paid-provider reference in Quickstart: {term}"


def test_public_makefile_and_readme_do_not_use_recruiter_wording():
    public_text = _makefile() + "\n" + Path("README.md").read_text(encoding="utf-8")

    assert "recruiter" not in public_text.lower()
