from gatecache.eval.novel_eval_dataset import build_novel_eval_examples


def test_novel_eval_examples_are_labeled_and_non_empty() -> None:
    novel = build_novel_eval_examples()
    assert len(novel) >= 4
    for ex in novel:
        assert ex.query.strip()
        assert ex.label.value
