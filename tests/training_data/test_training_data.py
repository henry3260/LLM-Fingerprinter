from llm_fingerprinter.training_data import balance_grouped_samples


def test_balance_grouped_samples_uses_smallest_group_size():
    grouped = {
        "gpt": list(range(7)),
        "gemini": list(range(5)),
        "claude": list(range(6)),
    }

    balanced, target_count = balance_grouped_samples(grouped, seed=42)

    assert target_count == 5
    assert {group: len(samples) for group, samples in balanced.items()} == {
        "claude": 5,
        "gemini": 5,
        "gpt": 5,
    }


def test_balance_grouped_samples_is_reproducible_and_does_not_mutate_input():
    grouped = {
        "gpt": list(range(10)),
        "gemini": list(range(100, 105)),
    }
    original = {group: list(samples) for group, samples in grouped.items()}

    first, _ = balance_grouped_samples(grouped, seed=123)
    second, _ = balance_grouped_samples(grouped, seed=123)

    assert first == second
    assert grouped == original


def test_balance_grouped_samples_ignores_empty_groups():
    balanced, target_count = balance_grouped_samples(
        {"gpt": [1, 2], "gemini": []}
    )

    assert balanced == {"gpt": [1, 2]}
    assert target_count == 2
