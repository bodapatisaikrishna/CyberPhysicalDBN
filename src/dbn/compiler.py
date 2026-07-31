"""Attack graph -> 2TBN compiler.

Implements the AG-to-DBN translation of Cerotti et al. Sec. III, targeting the
canonical form defined in Sec. II-B: "A DBN (2TBN) is said to be in canonical
form when intra-slice arcs in slice t - dt are not represented."

Target representation: a flat pgmpy DiscreteBayesianNetwork whose nodes are
(name, time_slice) tuples with time_slice in {0, 1}.

Why not pgmpy's own DynamicBayesianNetwork: its add_edge normalizes any
intra-slice edge to slice 0 and then mirrors it to slice 1, so an ulterior-layer
intra-slice arc such as CredAccess(1) -> MITM(1) necessarily also creates
CredAccess(0) -> MITM(0). That is an anterior-layer intra-slice arc, which
canonical form forbids. A flat network places exactly the arcs asked for.
"""

from __future__ import annotations

import networkx as nx
from pgmpy.models import DiscreteBayesianNetwork

DBNNode = tuple[str, int]

ANTERIOR = 0
ULTERIOR = 1


def compile_to_2tbn(ag: nx.DiGraph) -> DiscreteBayesianNetwork:
    """Compile the attack graph into a 2TBN structure. No CPDs are attached.

    Edge placement follows one rule, from Cerotti et al. Sec. III-A and Fig. 2:
    an edge crosses a slice exactly when both endpoints persist. Concretely,
    for an attack-graph edge (u, v):

      - both u and v self-loop  -> (u, 0) -> (v, 1), an inter-slice arc
      - otherwise               -> (u, 1) -> (v, 1), intra-slice at the
                                   ulterior layer

    Self-loops fall out of the first case as (n, 0) -> (n, 1), the temporal arc
    carrying persistence. Untimed nodes (the analytics, and the CredAccess/
    UnstablePS gates) fall into the second case, so they exist only at the
    ulterior layer and depend on their parents within the same slice. This is
    what makes MITM depend on CredAccess at time t rather than t-1.

    Raises if the result carries an anterior-layer intra-slice arc, which would
    mean the graph is not in canonical form.
    """
    dbn = DiscreteBayesianNetwork()

    for source, target, data in ag.edges(data=True):
        if data["inter_slice"]:
            dbn.add_edge((source, ANTERIOR), (target, ULTERIOR))
        else:
            dbn.add_edge((source, ULTERIOR), (target, ULTERIOR))

    anterior_intra_slice = [
        (u, v) for u, v in dbn.edges() if u[1] == ANTERIOR and v[1] == ANTERIOR
    ]
    if anterior_intra_slice:
        raise ValueError(
            "compiled 2TBN is not in canonical form; anterior-layer intra-slice "
            f"arcs present: {anterior_intra_slice}"
        )

    return dbn
