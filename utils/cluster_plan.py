# === Flop granularity stays the same ===
FLOP_CLUSTER_GRANULARITY = 256

# === Contexts (unchanged) ===
CLUSTER_CONTEXTS = ["OPEN", "VS_3BET", "VS_LIMP", "VS_4BET"]

# === Matchups per context (bumped smartly) ===
SRP_MATCHUPS = [
    ("BTN","CO"), ("BTN","MP"), ("BTN","UTG"),
    ("CO","MP"),  ("CO","UTG"),
    ("SB","BTN"),
    ("BB","BTN"), ("BB","CO"), ("BB","MP"), ("BB","UTG"), ("BB","SB"),
    ("CO","BB"),  # common steal vs BB defense
]

# base stacks
SRP_STACKS_BASE = [20, 40, 75, 100]
# add 150bb only for the most valuable SRP pairs
SRP_STACKS_150_UPSELL = {
    ("BTN","CO"), ("SB","BTN"), ("BB","BTN"),
}

THREEBET_MATCHUPS = [
    ("BTN","CO"), ("BTN","MP"), ("BTN","UTG"),
    ("SB","BTN"),
    ("BB","BTN"), ("BB","CO"),
]
# 3bet stacks: keep 40/100/150 globally, add 20bb only where relevant
THREEBET_STACKS_BASE = [40, 100, 150]
THREEBET_STACKS_PLUS_20 = {
    ("BTN","CO"), ("SB","BTN"),
}

LIMP_MATCHUPS = [
    ("SB","BB"),
]
LIMP_STACKS = [20, 40, 75, 100]

FOURBET_MATCHUPS = [
    ("BTN","CO"), ("SB","BTN"), ("BB","BTN"), ("CO","MP"),
    ("BTN","BB"), ("BB","CO"),  # symmetry bump
]
FOURBET_STACKS = [40, 100]

# === Axes per context (profiles/exploit/multiway/pop) ===
# Keep villain profile at GTO everywhere (exploit net will adapt later).
PER_CONTEXT_AXES = {
    "OPEN": {
        "profiles": ["GTO"],
        "exploits": ["GTO"],
        # add HU everywhere; add 3WAY only for BTN_vs_CO
        "multiways": ["HU"],
        # add POP=RECREATIONAL (EV boost in SRP) in addition to REGULAR
        "pops": ["REGULAR", "RECREATIONAL"],
        # optional per-matchup overrides for multiway
        "multiway_overrides": {
            ("BTN","CO"): ["HU", "3WAY"],
        },
    },
    "VS_3BET": {
        "profiles": ["GTO"],
        # allow EXPLOIT_LIGHT here; 3bet pots benefit most from slight aggression tweaks
        "exploits": ["GTO", "EXPLOIT_LIGHT"],
        "multiways": ["HU"],
        "pops": ["REGULAR"],  # keep narrow here; cost control
    },
    "VS_LIMP": {
        "profiles": ["GTO"],
        "exploits": ["GTO"],
        "multiways": ["HU"],   # postflop usually HU (SB limp vs BB check)
        "pops": ["REGULAR", "RECREATIONAL"],
    },
    "VS_4BET": {
        "profiles": ["GTO"],
        "exploits": ["GTO"],
        "multiways": ["HU"],
        "pops": ["REGULAR"],
    },
}

# === Bundle per-context plan (with stack fine-tuning) ===
CONTEXT_PLAN = {
    "OPEN": {
        "matchups": SRP_MATCHUPS,
        "stacks_common": SRP_STACKS_BASE,
        "stacks_by_matchup": {
            m: sorted(set(SRP_STACKS_BASE + [150])) for m in SRP_STACKS_150_UPSELL
        },
    },
    "VS_3BET": {
        "matchups": THREEBET_MATCHUPS,
        "stacks_common": THREEBET_STACKS_BASE,
        "stacks_by_matchup": {
            m: sorted(set(THREEBET_STACKS_BASE + [20])) for m in THREEBET_STACKS_PLUS_20
        },
    },
    "VS_LIMP": {
        "matchups": LIMP_MATCHUPS,
        "stacks_common": LIMP_STACKS,
        "stacks_by_matchup": {},
    },
    "VS_4BET": {
        "matchups": FOURBET_MATCHUPS,
        "stacks_common": FOURBET_STACKS,
        "stacks_by_matchup": {},
    },
}

def _stacks_for(context: str, matchup: tuple[str,str]) -> list[int]:
    plan = CONTEXT_PLAN[context]
    stacks = plan["stacks_by_matchup"].get(matchup)
    return stacks if stacks else plan["stacks_common"]

def _axes_for(context: str, matchup: tuple[str,str]) -> tuple[list[str], list[str], list[str], list[str]]:
    axes = PER_CONTEXT_AXES[context]
    profiles = axes["profiles"]
    exploits = axes["exploits"]
    # multiway overrides (e.g., OPEN BTN_vs_CO → ["HU","3WAY"])
    multiways = axes.get("multiway_overrides", {}).get(matchup, axes["multiways"])
    pops = axes["pops"]
    return profiles, exploits, multiways, pops

def iter_cluster_axes():
    """
    Yields:
      (context, ip_position, oop_position, stack_bb,
       villain_profile, exploit_setting, multiway_context, population_type)
    """
    for context in CLUSTER_CONTEXTS:
        matchups = CONTEXT_PLAN[context]["matchups"]
        for ip, oop in matchups:
            for stack in _stacks_for(context, (ip, oop)):
                profiles, exploits, multiways, pops = _axes_for(context, (ip, oop))
                for prof in profiles:
                    for expl in exploits:
                        for mw in multiways:
                            for pop in pops:
                                yield (context, ip, oop, stack, prof, expl, mw, pop)