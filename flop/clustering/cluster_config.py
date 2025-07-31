from enum import Enum

class FlopClusterGranularity(Enum):
    LOW = 16
    MEDIUM = 32
    DEFAULT = 64
    HIGH = 128
    VERY_HIGH = 256