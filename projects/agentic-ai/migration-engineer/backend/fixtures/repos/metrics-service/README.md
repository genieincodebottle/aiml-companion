# metrics-service (exercise fixture)

Used by exercise #2 in [docs/EXERCISES.md](../../../../docs/EXERCISES.md): the rulebook
has no rule for the deprecated `datetime.utcfromtimestamp()` yet - you write it. Two
call sites, no `timezone` import, so a correct migration takes two passes. Check:
`python test_metrics.py`.
