from typing import List, Set
import random
from Agents.PunishSelectAgent import PunishSelectAgent


def select_punishers(population: List[PunishSelectAgent], play_agent_idxs: Set[int]):
    possible_punishers = [(punisher_idx, punisher) for punisher_idx, punisher in enumerate(population)
                          if punisher_idx not in play_agent_idxs]
    # The punisher for the two agents can be the same or different.
    punisher_1_idx, punisher_1 = random.choice(possible_punishers)
    punisher_2_idx, punisher_2 = random.choice(possible_punishers)
    return punisher_1_idx, punisher_1, punisher_2_idx, punisher_2