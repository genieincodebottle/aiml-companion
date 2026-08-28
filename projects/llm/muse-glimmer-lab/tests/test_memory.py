from glimmer.config import ARCH
from glimmer.memory import kv_cache_bytes, total_footprint_gib
from glimmer.speculative import expected_acceptance, simulate

GIB = 1024**3


def test_layer_split_matches_the_pattern():
    """52 layers on a [local, local, local, global] cycle is 13 global."""
    kv = kv_cache_bytes(8192)
    assert kv.global_layers == 13
    assert kv.local_layers == 39
    assert kv.global_layers + kv.local_layers == ARCH.layers


def test_per_token_per_layer_cost():
    """2 (key and value) x 2 kv heads x 128 head dim x 2 bytes = 1024 bytes."""
    kv = kv_cache_bytes(1, use_sliding_window=False)
    assert kv.global_bytes == ARCH.layers * 1024


def test_local_layers_stop_growing_at_the_window():
    """The cap is the whole point of the sliding window."""
    short = kv_cache_bytes(ARCH.sliding_window)
    long = kv_cache_bytes(ARCH.sliding_window * 64)
    assert short.local_bytes == long.local_bytes
    assert long.global_bytes > short.global_bytes


def test_gqa_saves_the_head_ratio():
    with_gqa = kv_cache_bytes(32_768)
    without = kv_cache_bytes(32_768, use_gqa=False)
    ratio = ARCH.q_heads / ARCH.kv_heads
    assert without.total_bytes == with_gqa.total_bytes * ratio


def test_shipped_cache_fits_a_consumer_card():
    """1.8 GiB at full context is the number that makes 24 GB workable."""
    shipped = kv_cache_bytes(131_072).total_gib
    assert 1.5 < shipped < 2.5

    plain = kv_cache_bytes(131_072, use_gqa=False, use_sliding_window=False).total_gib
    assert plain > 100  # a plain transformer of this shape needs a rack


def test_quantized_cache_halves_it():
    assert (
        kv_cache_bytes(131_072, bytes_per_value=1).total_bytes
        == kv_cache_bytes(131_072, bytes_per_value=2).total_bytes // 2
    )


def test_footprint_itemises_the_optional_files():
    lean = total_footprint_gib(131_072, "kquant-17gb")
    loaded = total_footprint_gib(131_072, "kquant-17gb", with_vision=True, with_drafter=True)
    assert loaded["total"] - lean["total"] == 3.0
    assert lean["total"] < 24 < loaded["total"] + 3  # the 24 GB card gets tight


def test_acceptance_grows_with_block_and_saturates():
    small = expected_acceptance(4, 0.87)
    big = expected_acceptance(64, 0.87)
    assert 1 < small < big
    # A geometric series converges, so doubling the block stops helping.
    assert big - expected_acceptance(32, 0.87) < 0.1


def test_diffusion_drafting_cost_is_flat_in_block_size():
    """The single claim the whole DFlash argument rests on."""
    small = simulate(target_step_ms=13.35, drafter="diffusion", block_size=4)
    large = simulate(target_step_ms=13.35, drafter="diffusion", block_size=64)
    assert small.draft_ms == large.draft_ms
    assert large.speedup > small.speedup


def test_autoregressive_drafting_turns_over():
    """Its cost grows with the block while acceptance flattens, so it peaks."""
    speedups = [
        simulate(target_step_ms=13.35, drafter="autoregressive", block_size=b).speedup
        for b in (4, 16, 64)
    ]
    assert speedups[1] > speedups[0]
    assert speedups[2] < speedups[1]


def test_no_drafter_is_the_baseline():
    result = simulate(target_step_ms=13.35, drafter="none")
    assert result.speedup == 1.0
    assert result.acceptance_length == 1.0
