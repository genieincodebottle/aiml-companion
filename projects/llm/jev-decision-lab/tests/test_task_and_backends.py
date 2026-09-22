"""The dataset, the question sets and the offline backend. No network anywhere."""

from __future__ import annotations

from src.backends import SimulatedBackend, get_backend
from src.experiment import collect, compare, state_of
from src.questions import DECOMPOSED, PRIMITIVE_DEMO, SINGLE, ChoiceQ, NoulQ, ScoreQ
from src.task import PHRASES, SIGNALS, make_tickets, split


def test_tickets_are_identical_on_every_run():
    assert [t.text for t in make_tickets(50)] == [t.text for t in make_tickets(50)]


def test_a_different_seed_gives_different_tickets():
    assert [t.text for t in make_tickets(50, seed=7)] != [t.text for t in make_tickets(50, seed=8)]


def test_every_signal_shows_up_and_none_dominates():
    tickets = make_tickets(600)
    for name in SIGNALS:
        rate = sum(t.signals[name] for t in tickets) / len(tickets)
        assert 0.10 < rate < 0.50, f"{name} appears on {rate:.0%} of tickets"

    escalated = sum(t.escalate for t in tickets) / len(tickets)
    assert 0.20 < escalated < 0.60


def test_signal_text_matches_the_signal_flag():
    for ticket in make_tickets(200):
        for name, on in ticket.signals.items():
            found = any(phrase in ticket.text for phrase in PHRASES[name])
            assert found == on, f"{ticket.id} {name}"


def test_the_state_shown_to_the_model_hides_the_answer():
    ticket = make_tickets(5)[0]
    state = state_of(ticket)
    assert set(state) == {"id", "ticket"}
    assert state["ticket"] == ticket.text


def test_split_does_not_overlap():
    train, test = split(make_tickets(100))
    assert len(train) == len(test) == 50
    assert not ({t.id for t in train} & {t.id for t in test})


def test_decomposed_asks_one_question_per_signal():
    assert set(DECOMPOSED.questions) == set(SIGNALS)
    assert len(SINGLE.questions) == 1


def test_primitive_demo_covers_all_three_types():
    kinds = {type(q) for q in PRIMITIVE_DEMO.questions.values()}
    assert kinds == {NoulQ, ChoiceQ, ScoreQ}


def test_choice_stays_inside_the_cardinality_limit():
    for q in PRIMITIVE_DEMO.questions.values():
        if isinstance(q, ChoiceQ):
            assert 2 <= len(q.criteria) <= 255
        if isinstance(q, ScoreQ):
            assert 2 <= len(q.criteria) <= 10


def test_simulated_backend_is_deterministic():
    ticket = make_tickets(5)[0]
    a = SimulatedBackend().ask(state_of(ticket), DECOMPOSED.questions)
    b = SimulatedBackend().ask(state_of(ticket), DECOMPOSED.questions)
    assert {k: v.noul for k, v in a.items()} == {k: v.noul for k, v in b.items()}


def test_simulated_backend_returns_probabilities_not_labels():
    rows = collect(SimulatedBackend(), make_tickets(40), DECOMPOSED)
    values = [v for row in rows for v in row.values()]
    assert all(0.0 <= v <= 1.0 for v in values)
    assert any(0.05 < v < 0.95 for v in values), "every answer was pinned to an end"


def test_narrow_questions_track_the_signal_they_ask_about():
    tickets = make_tickets(200)
    rows = collect(SimulatedBackend(), tickets, DECOMPOSED)
    for name in SIGNALS:
        on = [r[name] for r, t in zip(rows, tickets) if t.signals[name]]
        off = [r[name] for r, t in zip(rows, tickets) if not t.signals[name]]
        assert sum(on) / len(on) > sum(off) / len(off) + 0.4, name


def test_both_arms_are_fitted_on_the_training_half_only():
    tickets = make_tickets(200)
    arms = compare(SimulatedBackend(), tickets, target=0.95)
    for arm in arms:
        assert len(arm.probs) == len(arm.labels) == 100
        assert arm.calls == 200


def test_decomposing_widens_the_band_you_can_automate():
    arms = {a.name: a for a in compare(SimulatedBackend(), make_tickets(600), target=0.95)}
    single, decomposed = arms["single"], arms["decomposed"]
    assert single.point and decomposed.point
    assert decomposed.point.coverage > single.point.coverage


def test_get_backend_rejects_an_unknown_name():
    try:
        get_backend("jev-ultra")
    except ValueError as exc:
        assert "jev-ultra" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_score_is_a_weighted_mean_not_a_level_index():
    """A Score returns the probability-weighted mean of the level numbers.

    Getting this wrong is easy, because `level index` is the intuitive reading and
    the value often looks like one. It is a decimal, it lands between levels, and
    it stays inside the range of level numbers.
    """
    from src.questions import ScoreQ

    ticket = make_tickets(40)[0]
    q = {k: v for k, v in PRIMITIVE_DEMO.questions.items() if isinstance(v, ScoreQ)}
    assert q, "the demo must carry a Score question"

    seen_fractional = False
    for t in make_tickets(40):
        for key, answer in SimulatedBackend().ask(state_of(t), q).items():
            levels = len(PRIMITIVE_DEMO.questions[key].criteria)
            assert isinstance(answer.score, float)
            assert 0.0 <= answer.score <= levels - 1
            if abs(answer.score - round(answer.score)) > 1e-6:
                seen_fractional = True
    assert seen_fractional, "every score was a whole number, so it is being treated as an index"
