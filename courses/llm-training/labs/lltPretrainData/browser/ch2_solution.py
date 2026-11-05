def p_candidate(j, bands, rows):
    # Probability that two documents with Jaccard similarity j share at least one band
    return 1 - (1 - j ** rows) ** bands

def pick_banding(k, j_dup, j_far):
    # Among (bands, rows) with bands * rows == k, return the one with the highest
    # p_candidate(j_dup) that keeps p_candidate(j_far) at or below 0.10
    best = None
    for rows in range(1, k + 1):
        if k % rows:
            continue
        bands = k // rows
        if p_candidate(j_far, bands, rows) <= 0.10:
            if best is None or p_candidate(j_dup, bands, rows) > p_candidate(j_dup, *best):
                best = (bands, rows)
    return best

# Checks. Run the cell. It prints All checks passed when both functions are right.
assert p_candidate(0.8, 14, 8) is not None, "p_candidate returns None"
assert abs(p_candidate(0.8, 14, 8) - 0.9235) < 1e-4, "14 bands of 8 rows at J = 0.8 should give 0.9235"
assert abs(p_candidate(0.5, 14, 8) - 0.0533) < 1e-4, "14 bands of 8 rows at J = 0.5 should give 0.0533"
assert abs(p_candidate(0.5, 20, 5) - 0.4701) < 1e-4, "20 bands of 5 rows at J = 0.5 should give 0.4701"
choice = pick_banding(112, 0.8, 0.5)
assert choice is not None, "pick_banding returned None"
assert choice[0] * choice[1] == 112, "bands * rows must use all 112 hashes"
assert choice == (14, 8), "the best banding for 112 hashes is 14 bands of 8 rows"
print("chosen", choice, "| P(0.8)", round(p_candidate(0.8, *choice), 4), "| P(0.5)", round(p_candidate(0.5, *choice), 4))
print("All checks passed")
