from llm_fingerprinter.promptgen import (
    PromptItem,
    PromptPackage,
    PromptSuite,
    build_prompt_package,
)


def test_build_prompt_package_from_dicts():
    raw = [
        {"text": "hello", "layer": "stylistic", "category": "style"},
        {
            "text": "world",
            "layer": "behavioral",
            "category": "policy",
            "system": "be concise",
        },
    ]

    package = build_prompt_package(raw)

    assert isinstance(package, PromptPackage)
    assert len(package.prompts) == 2
    assert package.prompts[0] == PromptItem(
        text="hello", layer="stylistic", category="style", system=None
    )
    assert package.prompts[1].system == "be concise"


def test_prompt_item_to_dict_omits_none_system():
    item = PromptItem(text="x", layer="stylistic", category="style")
    out = item.to_dict()

    assert out == {"text": "x", "layer": "stylistic", "category": "style"}


def test_prompt_package_by_layer_and_to_dict_list():
    package = PromptPackage(
        prompts=(
            PromptItem(text="a", layer="discriminative", category="identity"),
            PromptItem(text="b", layer="stylistic", category="creative"),
        )
    )

    stylistic = package.by_layer("stylistic")
    assert len(stylistic) == 1
    assert stylistic[0].text == "b"

    as_dict = package.to_dict_list(layer="discriminative")
    assert as_dict == [
        {"text": "a", "layer": "discriminative", "category": "identity"}
    ]


def test_prompt_suite_default_prompts_shape():
    suite = PromptSuite()

    prompts = suite.get_prompts()
    assert len(prompts) == len(suite)
    assert len(prompts) > 0
    assert all("text" in p and "layer" in p and "category" in p for p in prompts)


def test_prompt_suite_layer_filter():
    suite = PromptSuite()

    prompts = suite.get_prompts(layer=PromptSuite.LAYER_BEHAVIORAL)

    assert len(prompts) > 0
    assert all(p["layer"] == PromptSuite.LAYER_BEHAVIORAL for p in prompts)
