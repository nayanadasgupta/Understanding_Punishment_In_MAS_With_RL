from typing import List
from Agents.PunishSelectAgent import PunishSelectAgent
import networkx as nx


# No limit to the number of times an agent can be selected. Agent cannot select itself.
def select_partners(agent_population: List[PunishSelectAgent], agent_reputations: List[int]):
    pairings = []
    for agent_index, agent in enumerate(agent_population):
        possible_partners = [partner_idx for partner_idx in range(len(agent_population)) if partner_idx not in
                             {agent_index}]
        possible_partner_idx = int(agent.select_model.select_action(agent_reputations))
        selected_agent = possible_partners[possible_partner_idx]
        pairings.append((selected_agent, possible_partner_idx))
    return pairings


# No limit to the number of times an agent can be selected. Agent cannot select itself.
def select_partners_caveman(G: nx.Graph, agent_population: List[PunishSelectAgent], agent_reputations: List[int]):
    pairings = []
    selection_states = []
    for agent_index, agent in enumerate(agent_population):
        neighbours = list(G.neighbors(agent_index))
        # Agents cannot see the reputations of those who are not its neighbours. They also cannot choose non-neighbours.
        neighbour_reputations = agent_reputations.copy()
        neighbour_reputations = [neighbour_reputations[i] for i in neighbours]
        possible_partners = [partner_idx for partner_idx in neighbours if partner_idx not in {agent_index}]
        possible_partner_idx = int(agent.select_model.select_action(neighbour_reputations))
        selected_agent = possible_partners[possible_partner_idx]
        pairings.append((selected_agent, possible_partner_idx))
        selection_states.append(neighbour_reputations)
    return pairings, selection_states