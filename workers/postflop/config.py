# --- Fixed constants for postflop key building ---
VILLAIN_PROFILE   = "GTO"                # Balanced, solver-based baseline
EXPLOIT_SETTING   = "GTO"                # Default solver strategy
MULTIWAY_CONTEXT  = "HU"                 # Heads-up context
POPULATION_TYPE   = "RECREATIONAL"       # Default pop type
ACTION_CONTEXT    = "OPEN"               # Default action context

# Optional: you could centralize them in a dict if needed
DEFAULT_CLUSTER_META = {
    "villain_profile": VILLAIN_PROFILE,
    "exploit_setting": EXPLOIT_SETTING,
    "multiway_context": MULTIWAY_CONTEXT,
    "population_type": POPULATION_TYPE,
    "action_context": ACTION_CONTEXT,
}