from glimmer import channels
from glimmer.channels import Message


def test_three_channels_split():
    raw = (
        "<|start|>assistant to=self<|message|>I should look this up.<|eom|>"
        "<|start|>assistant to=search_docs<|message|>QUERY<|eom|>"
        "<|start|>assistant to=user<|message|>The window is 2048.<|eot|>"
    )
    turn = channels.parse(raw)
    assert len(turn.messages) == 3
    assert turn.reasoning == "I should look this up."
    assert turn.final == "The window is 2048."
    assert [m.recipient for m in turn.tool_calls] == ["search_docs"]


def test_wants_tool_only_without_a_final_message():
    tool_only = channels.parse(
        "<|start|>assistant to=self<|message|>think<|eom|>"
        "<|start|>assistant to=calculate<|message|>CALL<|eom|>"
    )
    assert tool_only.wants_tool is True

    answered = channels.parse("<|start|>assistant to=user<|message|>done<|eot|>")
    assert answered.wants_tool is False


def test_missing_recipient_defaults_to_user():
    turn = channels.parse("<|start|>assistant<|message|>hello<|eot|>")
    assert turn.messages[0].recipient == "user"
    assert turn.final == "hello"


def test_terminator_is_recorded():
    """Which marker ended a message tells the loop whether the turn is over."""
    turn = channels.parse(
        "<|start|>assistant to=calculate<|message|>CALL<|eom|>"
    )
    assert turn.messages[0].terminator == channels.END_OF_MESSAGE

    turn = channels.parse("<|start|>assistant to=user<|message|>hi<|eot|>")
    assert turn.messages[0].terminator == channels.END_OF_TURN


def test_eom_is_not_a_stop_token():
    """The single most common way to break an agentic run."""
    assert channels.END_OF_MESSAGE not in channels.STOP_TOKENS
    assert set(channels.STOP_TOKENS) == {channels.END_OF_TURN, channels.END_OF_TEXT}


def test_plain_text_parses_as_one_final_message():
    """A non-Glimmer completion should not blow up the parser."""
    turn = channels.parse("just a sentence")
    assert turn.final == "just a sentence"
    assert turn.reasoning == ""


def test_empty_input():
    assert channels.parse("").messages == []


def test_round_trip():
    original = [
        Message("assistant", "self", "thinking", channels.END_OF_MESSAGE),
        Message("assistant", "user", "answer", channels.END_OF_TURN),
    ]
    reparsed = channels.parse(channels.render(original)).messages
    assert [(m.recipient, m.content) for m in reparsed] == [
        ("self", "thinking"),
        ("user", "answer"),
    ]
