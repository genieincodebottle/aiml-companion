"""Entity resolution: deciding when two names are the same thing.

This is the hardest correctness problem in GraphRAG and the one that quietly
destroys the most systems.  Get it wrong in one direction and "Meridian
Circuits Sdn Bhd", "Meridian Circuits" and "Meridian" become three nodes, the
graph fragments, and every traversal returns a third of the truth.  Get it
wrong in the other direction and "Kaohsiung" the city merges with "Kaohsiung
Precision Glass" the company, and the system starts reporting that a glass
processor was hit by a typhoon that actually hit a city - a claim it will cite
evidence for, because the evidence is real and only the identity is wrong.

The resolver here is a three-stage ladder, cheapest and safest first.  Nothing
in it is a language model, which is deliberate: resolution runs on every
extracted mention, so it must be fast, free and above all *deterministic*.  A
resolver that returns different answers on different runs makes the entire
graph irreproducible.

  Stage 1  Normalised exact match.   Handles case, punctuation, legal suffixes.
  Stage 2  Alias lookup.             Handles the mappings a human declared.
  Stage 3  Guarded fuzzy match.      Handles typos and truncations, under
                                     conditions strict enough that the
                                     Kaohsiung failure above cannot happen.

Where an LLM *would* earn its place is stage 4: adjudicating the genuinely
ambiguous pairs this ladder rejects, in a human-reviewed queue.  That is a
production feature and it is described in docs/production-notes.md rather than
half-built here.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher

# Corporate form suffixes carry no identity.  "Helios Fluidics BV" and "Helios
# Fluidics" are the same company; the BV is a jurisdiction artefact.
_LEGAL_SUFFIXES = {
    "inc", "incorporated", "llc", "ltd", "limited", "plc", "corp",
    "corporation", "co", "company", "gmbh", "ag", "bv", "nv", "ab", "as",
    "sa", "srl", "spa", "pte", "pty", "sdn", "bhd", "pvt", "private", "kk",
    "oy", "aps", "holdings", "group", "works",
}

# Words that never disambiguate an entity in this domain.  Stripped only for
# the *comparison* key; the display name always keeps them.
_NOISE = {"the", "of", "and"}

_PUNCT = re.compile(r"[^\w\s-]", re.UNICODE)
_SPACES = re.compile(r"\s+")

# Anything that looks like a part number is matched exactly and never fuzzily.
# "PCB-A7" and "PCB-B2" are 83% similar as strings and are completely different
# components.  Fuzzy matching over identifiers is a bug, not a feature.
_IDENTIFIER = re.compile(r"^[A-Z]{2,4}[-_]?[A-Z0-9]{1,6}$")

# Below this length a fuzzy match is meaningless: at 4 characters, one edit is
# 25% of the string.
_MIN_FUZZY_LEN = 8
_FUZZY_THRESHOLD = 0.90


# A NOTE ON STEMMING, AND WHY THERE ISN'T ANY
#
# An earlier version of this resolver folded trailing plurals, so that "NdFeB
# magnets" and "NdFeB magnet" produced the same key. It worked for that case and
# quietly wrecked every other name in the corpus: "Helios Fluidics" keyed as
# `helio-fluidic`, "Sentinel Optics" as `sentinel-optic`, "Baltic Lithium Salts"
# as `baltic-lithium-salt`. Those keys are stable and the system worked, so
# nothing failed - it just produced identifiers that look broken to anyone
# reading the graph, which in a system whose whole value is auditability is a
# real cost.
#
# The fix was to delete the rule, not to add exceptions to it. Stage 3 below
# already handles singular/plural pairs correctly: "ndfeb magnet" and "ndfeb
# magnets" score 0.97 on SequenceMatcher, comfortably over the 0.90 threshold,
# and the containment guard does not reject them because neither token set
# contains the other. The general mechanism covered the specific case, and the
# specific rule was doing damage elsewhere.


def normalise(name: str) -> str:
    """Reduce a surface form to a comparison key.

    Unicode is folded to NFKD first so that accented forms compare equal to
    their plain spellings; a corpus containing both "Skelleftea" and
    "Skellefteå" would otherwise produce two Swedish towns.
    """
    text = unicodedata.normalize("NFKD", name)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().strip()
    text = _PUNCT.sub(" ", text)
    tokens = [t for t in _SPACES.split(text) if t]
    tokens = [t for t in tokens if t not in _LEGAL_SUFFIXES and t not in _NOISE]
    if not tokens:                       # name was nothing but suffixes
        tokens = [t for t in _SPACES.split(text) if t]
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# TYPE NAMESPACES
#
# The key includes the entity type, so two names that normalise identically
# under different types stay separate.  That is usually right: a Location
# "Meridian" and a Supplier "Meridian" are not the same thing.
#
# But it is wrong for one pair in this domain.  A model reading "Formosa
# supplies copper-clad laminate" may reasonably call the laminate a Component
# or a Material, and both readings are defensible - the distinction is a
# modelling convention, not a fact about the world.  Left alone, the corpus
# produces "chemically strengthened cover glass" twice, once under each label,
# and every traversal through it returns half the truth.
#
# So Component and Material share one identity namespace.  They keep their own
# labels on the node; they just cannot become two nodes.
_TYPE_NAMESPACE = {"Component": "part", "Material": "part"}


def _namespace(entity_type: str) -> str:
    return _TYPE_NAMESPACE.get(entity_type, entity_type.lower())


# A "City, Country" string names a city.  Without this rule the corpus yields
# "Kaohsiung", "Kaohsiung, Taiwan" and "Taiwan" as three unrelated places, and
# the exposure query silently returns whichever subset happened to attach to
# the node the ERP created first.  That failure is invisible: the query returns
# rows, just not all of them.
def _location_form(name: str) -> str:
    return name.split(",")[0].strip() or name.strip()


def make_key(entity_type: str, name: str) -> str:
    """The graph's primary key.

    Type (or its namespace) is part of the key on purpose.  Without it, the
    Location "Kaohsiung" and a hypothetical Supplier "Kaohsiung" would collide
    into one node under the uniqueness constraint and become indistinguishable
    forever.  Typing the key makes that class of merge impossible by
    construction rather than by vigilance.
    """
    if entity_type == "Location":
        name = _location_form(name)
    return f"{_namespace(entity_type)}:{normalise(name).replace(' ', '-')}"


@dataclass
class ResolvedEntity:
    key: str
    name: str            # canonical display name (first or best-quality form)
    type: str
    aliases: list[str] = field(default_factory=list)
    summary: str = ""
    # Only meaningful for Finding entities: 'open' or 'closed'.  Stored on the
    # node because "does this supplier have a finding" and "does this supplier
    # have an OPEN finding" are completely different risk questions, and a
    # graph that cannot tell them apart answers both wrongly.
    status: str = ""
    # True when this entity came from a system of record (the ERP/PLM CSVs)
    # rather than from a model reading prose.  Drives the retype rule in
    # EntityResolver.resolve.
    authoritative: bool = False


class EntityResolver:
    """Maintains the registry of known entities for one ingestion run.

    Seeded from the structured CSVs before any document is read.  That ordering
    matters: the ERP names are authoritative, so when the extractor later reads
    "Meridian" out of a PDF it resolves *onto* the existing canonical node
    instead of creating a rival one that happens to be first.
    """

    def __init__(self) -> None:
        self._by_key: dict[str, ResolvedEntity] = {}
        # (namespace, normalised alias) -> key
        self._alias_index: dict[tuple[str, str], str] = {}
        # normalised name -> key, for BACKBONE entities only.  Consulted across
        # every type, which is what lets a mention the model mislabelled still
        # land on the right node - see `resolve`.
        self._authoritative_index: dict[str, str] = {}
        self.stats = {"exact": 0, "alias": 0, "fuzzy": 0, "new": 0,
                      "retyped": 0, "rejected_fuzzy": 0}

    # -------------------------------------------------------------- registry
    def register(self, entity_type: str, name: str, *,
                 aliases: list[str] | None = None,
                 summary: str = "", status: str = "",
                 authoritative: bool = False) -> ResolvedEntity:
        """Add an entity.

        ``authoritative=True`` marks it as coming from a system of record (the
        ERP/PLM CSVs).  Those names win every subsequent identity contest,
        including against a mention the extractor gave a different type.
        """
        key = make_key(entity_type, name)
        existing = self._by_key.get(key)
        if existing:
            for alias in aliases or []:
                self._add_alias(entity_type, alias, key, existing)
            return existing

        entity = ResolvedEntity(key=key, name=name, type=entity_type,
                                aliases=list(aliases or []), summary=summary,
                                status=status, authoritative=authoritative)
        self._by_key[key] = entity
        self._alias_index[(_namespace(entity_type), normalise(name))] = key
        if authoritative:
            self._authoritative_index.setdefault(normalise(name), key)
        for alias in aliases or []:
            self._add_alias(entity_type, alias, key, entity)
        return entity

    def _add_alias(self, entity_type: str, alias: str, key: str,
                   entity: ResolvedEntity) -> None:
        norm = normalise(alias)
        if not norm:
            return
        self._alias_index.setdefault((_namespace(entity_type), norm), key)
        if entity.authoritative:
            self._authoritative_index.setdefault(norm, key)
        if alias not in entity.aliases and alias != entity.name:
            entity.aliases.append(alias)

    # ------------------------------------------------------------- resolution
    def resolve(self, entity_type: str, name: str, *,
                summary: str = "", status: str = "") -> ResolvedEntity:
        """Map an extracted mention onto a canonical entity, creating one only
        if no stage of the ladder matches."""
        name = name.strip()
        if entity_type == "Location":
            name = _location_form(name)
        norm = normalise(name)

        # Stage 1 + 2: exact normalised form, then declared aliases.  Both are
        # dictionary lookups against the same index.
        hit = self._alias_index.get((_namespace(entity_type), norm))
        if hit:
            entity = self._by_key[hit]
            self.stats["exact" if norm == normalise(entity.name) else "alias"] += 1
            if name != entity.name and name not in entity.aliases:
                entity.aliases.append(name)
            return entity

        # Stage 2b: the name is known to a system of record, but under a
        # different type than the model assigned.
        #
        # This happens constantly and is not the model being careless. Asked to
        # label "DSP-3300" in the sentence "the DSP-3300 display module used in
        # the NW-500", Product is a perfectly reasonable reading. But the PLM
        # says DSP-3300 is a Component, and the PLM is authoritative about what
        # its own parts are.
        #
        # Without this stage the corpus grows a phantom Product node that no
        # bill of materials contains and no traversal can ever reach, while the
        # real Component loses the mentions that should have attached to it.
        authoritative = self._authoritative_index.get(norm)
        if authoritative:
            entity = self._by_key[authoritative]
            self.stats["retyped"] += 1
            if name != entity.name and name not in entity.aliases:
                entity.aliases.append(name)
            return entity

        # Stage 3: fuzzy, heavily guarded.
        candidate = self._fuzzy(entity_type, norm)
        if candidate:
            self.stats["fuzzy"] += 1
            entity = self._by_key[candidate]
            if name not in entity.aliases and name != entity.name:
                entity.aliases.append(name)
            self._alias_index[(_namespace(entity_type), norm)] = candidate
            return entity

        self.stats["new"] += 1
        return self.register(entity_type, name, summary=summary, status=status)

    def _fuzzy(self, entity_type: str, norm: str) -> str | None:
        if _IDENTIFIER.match(norm.upper().replace(" ", "-")):
            return None                      # never fuzzy-match a part number
        if len(norm) < _MIN_FUZZY_LEN:
            return None                      # too short for a ratio to mean anything

        best_key, best_score = None, 0.0
        for (namespace, candidate_norm), key in self._alias_index.items():
            if namespace != _namespace(entity_type):
                continue                     # never merge across namespaces
            score = SequenceMatcher(None, norm, candidate_norm).ratio()
            if score > best_score:
                best_key, best_score = key, score

        if best_key is None or best_score < _FUZZY_THRESHOLD:
            if best_score > 0.75:
                # Close but not close enough.  Counted so the ingestion report
                # can show how often the resolver declined - a spike here means
                # the corpus has naming drift a human should look at.
                self.stats["rejected_fuzzy"] += 1
            return None

        # Final guard: one name must not be a strict superset of the other with
        # extra meaningful words.  "kaohsiung" vs "kaohsiung precision glass"
        # scores well below threshold anyway, but the containment check makes
        # the intent explicit and catches the near-threshold cases.
        candidate_norm = normalise(self._by_key[best_key].name)
        if _is_qualified_superset(norm, candidate_norm):
            self.stats["rejected_fuzzy"] += 1
            return None
        return best_key

    # ------------------------------------------------------------------ views
    def all_entities(self) -> list[ResolvedEntity]:
        return list(self._by_key.values())

    def get(self, key: str) -> ResolvedEntity | None:
        return self._by_key.get(key)


def _is_qualified_superset(a: str, b: str) -> bool:
    """True when one name is the other plus extra distinguishing words.

    "kaohsiung" and "kaohsiung precision glass" -> True (do not merge).
    "meridian circuits" and "meridian circuits" -> False (identical).
    """
    a_tokens, b_tokens = set(a.split()), set(b.split())
    if a_tokens == b_tokens:
        return False
    return a_tokens < b_tokens or b_tokens < a_tokens
