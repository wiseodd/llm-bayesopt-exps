from enum import Enum


class LLMFeatureType(Enum):
    LAST_TOKEN = 1
    FIRST_TOKEN = 2
    AVERAGE = 3
