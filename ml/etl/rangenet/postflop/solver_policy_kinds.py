from enum import Enum, auto

class ScenarioParseKind(Enum):
    IP_ROOT = auto()            # IP acts at root (PFR_IP, Aggressor_IP) → CHECK/BET_xx
    OOP_ROOT = auto()           # OOP acts at root (donk allowed) + after OOP CHECK follow IP BET → OOP responds
    OOP_VS_IP_ROOT_BET = auto() # Root actor is IP; we still want OOP responses vs root BET (SRP Caller_OOP style)
    LIMPed_SB_IP = auto()       # Limp single SB_IP (no raises expected; treat like IP_ROOT but tolerant)