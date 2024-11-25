from samplers.betting_by_time.betting_strategies import *

from samplers.betting_by_time.geo_checking import GeoCheckingCapital
from samplers.betting_by_time.sequence_checking import SequenceCheckingCapital


__all__ = ['betting_factory']


def betting_factory(mtd='vanilla_seq'):
    bet_str, cap_str = mtd.split('_')
    match cap_str:
        case 'geo':
            cls = GeoCheckingCapital
        case 'seq':
            cls = SequenceCheckingCapital
        case _:
            raise NotImplementedError(f"Unsupported capital process: {mtd}.")
    match bet_str:
        case 'vanilla':
            return cls, vanilla_betting_factory(cls)
        case 'ada':
            return cls, adaptive_betting_factory(cls, cap_str)
        case _:
            raise  NotImplementedError(f"Unsupported bet strategy: {mtd}.")
