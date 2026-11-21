import builtins

from src import generator


def test_support_generation_stays_local_even_with_old_llm_flags(monkeypatch):
    monkeypatch.setenv("USE_LLM_GENERATION", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "unused-test-value")
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert not name.startswith("langchain"), "Support must not import LangChain"
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    answer, mode = generator.generate_grounded_answer(
        "My VPN failed", [{"source": "vpn.md", "text": "Restart the VPN client."}])
    assert mode == "rule_based"
    assert "VPN" in answer
    assert not hasattr(generator, "generate_llm_grounded_answer")


def test_legacy_account_support_template_is_preserved():
    chunks = [{"source": "account.md", "text": "Use the approved password reset instructions."}]
    answer, mode = generator.generate_grounded_answer("My account is locked", chunks)
    assert "password" in answer.lower()
    assert mode == "rule_based"


def test_missing_context_keeps_service_desk_fallback(monkeypatch):
    monkeypatch.setattr(generator, "retrieve_relevant_chunks", lambda *args, **kwargs: [])
    monkeypatch.setattr(generator, "classify_ticket", lambda _: {"category": "General IT Support"})
    result = generator.answer_question("Unknown issue")
    assert result["fallback"]
    assert result["answer_generation_mode"] == "fallback"
    assert "Service Desk" in result["answer"]
