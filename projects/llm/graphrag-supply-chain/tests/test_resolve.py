"""Entity resolution tests.

These are the most important tests in the project. Resolution decides node
identity, and a resolution bug does not raise - it silently splits a supplier
into two nodes, or silently merges a city into a company. Either way the
traversals keep returning rows, just the wrong ones.

Every test below is a real case from the corpus, not an invented one.
"""

from __future__ import annotations

import pytest

from src.ingest.resolve import EntityResolver, make_key, normalise


class TestNormalise:
    def test_strips_legal_suffixes(self):
        assert normalise("Helios Fluidics BV") == normalise("Helios Fluidics")
        assert normalise("Meridian Circuits Sdn Bhd") == normalise("Meridian Circuits")
        assert normalise("Prahara Polymers Pvt Ltd") == normalise("Prahara Polymers")

    def test_folds_accents(self):
        # A corpus containing both spellings must not grow two Swedish towns.
        assert normalise("Skelleftea") == normalise("Skellefteå")

    def test_does_not_stem_company_names(self):
        """No stemming. A resolver that folds plurals turns "Helios Fluidics"
        into `helio-fluidic` and "Sentinel Optics" into `sentinel-optic`. The
        graph still works, but every key looks broken - an unacceptable cost in
        a system whose value is auditability."""
        assert normalise("Helios Fluidics") == "helios fluidics"
        assert normalise("Sentinel Optics Corp") == "sentinel optics"

    def test_singular_plural_pairs_still_merge_via_fuzzy(self):
        """The case the deleted stemmer existed for, handled by the general
        mechanism instead of a special rule."""
        resolver = EntityResolver()
        first = resolver.register("Material", "NdFeB magnets")
        second = resolver.resolve("Material", "NdFeB magnet")
        assert second.key == first.key

    def test_name_of_only_suffixes_survives(self):
        # Degenerate input must not normalise to the empty string, or every
        # such entity would collide into one node.
        assert normalise("Works") != ""


class TestMakeKey:
    def test_type_is_part_of_the_key(self):
        # A Location and a Supplier with the same name are different things.
        assert make_key("Location", "Meridian") != make_key("Supplier", "Meridian")

    def test_component_and_material_share_a_namespace(self):
        # A laminate is legitimately both. It must not become two nodes.
        assert make_key("Component", "cover glass") == make_key("Material", "cover glass")

    def test_city_country_resolves_to_city(self):
        # The single most damaging location bug in this corpus.
        assert make_key("Location", "Kaohsiung, Taiwan") == make_key("Location", "Kaohsiung")


class TestResolverLadder:
    def setup_method(self):
        self.resolver = EntityResolver()
        self.resolver.register(
            "Supplier", "Meridian Circuits Sdn Bhd",
            aliases=["Meridian Circuits", "Meridian"], authoritative=True,
        )
        self.resolver.register(
            "Component", "DSP-3300 5.5in TFT Display Module",
            aliases=["DSP-3300"], authoritative=True,
        )
        self.resolver.register("Location", "Kaohsiung", authoritative=True)

    def test_exact_match(self):
        entity = self.resolver.resolve("Supplier", "Meridian Circuits Sdn Bhd")
        assert entity.key == make_key("Supplier", "Meridian Circuits Sdn Bhd")
        assert self.resolver.stats["exact"] == 1

    def test_alias_match(self):
        assert self.resolver.resolve("Supplier", "Meridian").key == \
               self.resolver.resolve("Supplier", "Meridian Circuits Sdn Bhd").key

    def test_suffix_variation_matches(self):
        assert self.resolver.resolve("Supplier", "Meridian Circuits").key == \
               self.resolver.resolve("Supplier", "Meridian Circuits Sdn Bhd").key

    def test_authoritative_type_wins_over_model_type(self):
        """The retype rule.

        A model reading "the DSP-3300 display module used in the NW-500" may
        reasonably label DSP-3300 a Product. The PLM says it is a Component, and
        the PLM is authoritative about its own parts. Without this, the corpus
        grows a phantom Product no bill of materials contains.
        """
        entity = self.resolver.resolve("Product", "DSP-3300")
        assert entity.type == "Component"
        assert self.resolver.stats["retyped"] == 1

    def test_city_does_not_merge_with_company_in_that_city(self):
        """The failure that produces confident nonsense.

        "Kaohsiung" and "Kaohsiung Precision Glass" are 45% similar as strings
        and are a city and a company. Merging them makes the system report that
        a glass processor was hit by a typhoon that hit a city, citing real
        evidence for a false identity.
        """
        glass = self.resolver.resolve("Supplier", "Kaohsiung Precision Glass")
        city = self.resolver.resolve("Location", "Kaohsiung")
        assert glass.key != city.key
        assert glass.type == "Supplier"

    def test_part_numbers_are_never_fuzzy_matched(self):
        """PCB-A7 and PCB-B2 are 83% similar and are different components.

        Fuzzy matching over identifiers is a bug, not a feature.
        """
        resolver = EntityResolver()
        resolver.register("Component", "PCB-A7", authoritative=True)
        other = resolver.resolve("Component", "PCB-B2")
        assert other.key != make_key("Component", "PCB-A7")

    def test_qualified_superset_is_not_merged(self):
        resolver = EntityResolver()
        resolver.register("Supplier", "Formosa Substrate Materials")
        # A different company that merely shares a word must stay separate.
        other = resolver.resolve("Supplier", "Formosa Chemical Industries")
        assert other.key != make_key("Supplier", "Formosa Substrate Materials")

    def test_aliases_accumulate_on_the_surviving_node(self):
        """Aliases are how the full-text entity linker finds a node later.

        Discarding them at merge time is what makes a question phrased with the
        short name fail to link.
        """
        self.resolver.resolve("Supplier", "Meridian Circuits")
        entity = self.resolver.resolve("Supplier", "Meridian")
        assert "Meridian" in entity.aliases or "Meridian" == entity.name

    def test_resolution_is_deterministic(self):
        """Two resolvers fed the same input in the same order must agree.

        A resolver that depends on dict ordering or on a model call produces a
        different graph on every ingest, and nothing downstream is reproducible.
        """
        def build() -> list[str]:
            r = EntityResolver()
            r.register("Supplier", "Volta Cell Systems", aliases=["Volta"],
                       authoritative=True)
            names = ["Volta", "Volta Cell", "Baltic Lithium Salts",
                     "Baltic Lithium Salts", "volta cell systems"]
            return [r.resolve("Supplier", n).key for n in names]

        assert build() == build()
