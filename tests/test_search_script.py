import json
import sys
from unittest.mock import Mock

import pytest

from scripts import search


def test_main_prompts_for_query_when_omitted(monkeypatch, capsys):
    search_faiss = Mock(return_value=[{"scene_id": "scene-0061", "score": 0.82}])
    monkeypatch.setattr(search, "search_faiss", search_faiss)
    monkeypatch.setattr("builtins.input", Mock(return_value=" pedestrian crossing "))
    monkeypatch.setattr(sys, "argv", ["search.py", "--top-k", "5"])

    assert search.main() == 0

    search_faiss.assert_called_once_with(
        "pedestrian crossing",
        search.Path(search.__file__).resolve().parents[1] / "artifacts/index/v1",
        top_k=5,
    )
    assert json.loads(capsys.readouterr().out) == [
        {"scene_id": "scene-0061", "score": 0.82}
    ]


def test_main_keeps_command_line_query_without_prompt(monkeypatch, capsys):
    search_faiss = Mock(return_value=[])
    prompt = Mock(side_effect=AssertionError("input() should not be called"))
    monkeypatch.setattr(search, "search_faiss", search_faiss)
    monkeypatch.setattr("builtins.input", prompt)
    monkeypatch.setattr(sys, "argv", ["search.py", "car turning", "--top-k", "3"])

    assert search.main() == 0

    search_faiss.assert_called_once()
    assert search_faiss.call_args.args[0] == "car turning"
    assert search_faiss.call_args.kwargs == {"top_k": 3}
    assert json.loads(capsys.readouterr().out) == []


def test_main_rejects_empty_interactive_query(monkeypatch):
    monkeypatch.setattr("builtins.input", Mock(return_value="   "))
    monkeypatch.setattr(sys, "argv", ["search.py"])

    with pytest.raises(SystemExit, match="2"):
        search.main()
