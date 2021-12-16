from enum import Enum


class FlagLabels(Enum):
    co_viet_tan = 0
    co_ba_soc = 1
    co_viet_nam = 2


class NSFWLabels(Enum):
    EXPOSED_BUTTOCKS = 0
    EXPOSED_BREAST_F = 1
    EXPOSED_GENITALIA_F = 2
    EXPOSED_GENITALIA_M = 3


class HorrorLabels(Enum):
    horror = 1
    safe = 0

if __name__ == "__main__":
    x = FlagLabels(0).name
    print(x)

    