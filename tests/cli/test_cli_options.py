from llm_fingerprinter.cli import cli


def _balance_default(command_name):
    command = cli.commands[command_name]
    option = next(
        parameter for parameter in command.params
        if parameter.name == "balance"
    )
    return option.default


def test_family_balancing_defaults_match_command_purposes():
    assert _balance_default("train") is True
    assert _balance_default("build-templates") is False
