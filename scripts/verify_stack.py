"""Dependency + smoke-test verification for the CyberPhysicalDBN stack.

Run before any research code is written. Each check is independent: one
failure does not prevent the others from running. Exits non-zero if any
check fails. Prints PASS/FAIL per check plus a final summary.

Per CLAUDE.md rule 1, this script never fabricates a result — every line
it prints comes from an import, a computation, or an exception that just
happened.
"""

from __future__ import annotations

import platform
import sys
import traceback
from dataclasses import dataclass, field


@dataclass
class Results:
    passed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)


RESULTS = Results()


def run_check(name: str, fn) -> None:
    print(f"\n--- {name} ---")
    try:
        detail = fn()
        print(f"PASS {name} — {detail}")
        RESULTS.passed.append(name)
    except Exception as exc:  # noqa: BLE001 — smoke test, want any failure surfaced
        print(f"FAIL {name} — {type(exc).__name__}: {exc}")
        traceback.print_exc()
        RESULTS.failed.append(name)


def check_versions() -> str:
    import matplotlib
    import networkx
    import pandapower
    import pandas
    import pgmpy
    import simpy
    import sklearn
    import stable_baselines3
    import torch
    import torch_geometric
    import yaml

    mods = {
        "pandapower": pandapower.__version__,
        "pgmpy": pgmpy.__version__,
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "simpy": simpy.__version__,
        "networkx": networkx.__version__,
        "scikit-learn": sklearn.__version__,
        "stable_baselines3": stable_baselines3.__version__,
        "pyyaml": yaml.__version__,
        "pandas": pandas.__version__,
        "matplotlib": matplotlib.__version__,
        "pytest": __import__("pytest").__version__,
    }
    for name, version in mods.items():
        print(f"  {name}=={version}")
    return f"{len(mods)} packages imported"


def check_pandapower() -> str:
    import pandapower as pp
    import pandapower.networks as pn

    net = pn.case14()
    pp.runpp(net)
    assert net["converged"], "power flow did not converge"
    vm_bus0 = net.res_bus.loc[0, "vm_pu"]
    return f"case14 converged, bus 0 vm_pu={vm_bus0:.6f}"


def check_pgmpy() -> str:
    try:
        from pgmpy.models import DiscreteBayesianNetwork as BNModel

        model_cls_name = "DiscreteBayesianNetwork"
    except ImportError:
        from pgmpy.models import BayesianNetwork as BNModel

        model_cls_name = "BayesianNetwork"

    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.inference import VariableElimination

    model = BNModel([("A", "B"), ("B", "C")])
    cpd_a = TabularCPD("A", 2, [[0.6], [0.4]])
    cpd_b = TabularCPD(
        "B", 2, [[0.7, 0.2], [0.3, 0.8]], evidence=["A"], evidence_card=[2]
    )
    cpd_c = TabularCPD(
        "C", 2, [[0.9, 0.3], [0.1, 0.7]], evidence=["B"], evidence_card=[2]
    )
    model.add_cpds(cpd_a, cpd_b, cpd_c)
    assert model.check_model()

    infer = VariableElimination(model)
    result = infer.query(variables=["C"], show_progress=False)
    p_c1 = float(result.values[1])
    return f"used {model_cls_name}, P(C=1)={p_c1:.6f}"


def check_torch_geometric() -> str:
    import torch
    from torch_geometric.data import HeteroData
    from torch_geometric.nn import HGTConv

    data = HeteroData()
    data["ied"].x = torch.randn(5, 8)
    data["bus"].x = torch.randn(3, 8)
    data["ied", "monitors", "bus"].edge_index = torch.tensor(
        [[0, 1, 2, 3, 4], [0, 1, 2, 0, 1]], dtype=torch.long
    )

    metadata = (["ied", "bus"], [("ied", "monitors", "bus")])
    conv = HGTConv(in_channels=8, out_channels=16, metadata=metadata, heads=2)
    out = conv(data.x_dict, data.edge_index_dict)

    assert out["bus"].shape == (3, 16), out["bus"].shape
    return f"HGTConv forward pass ok, bus out shape={tuple(out['bus'].shape)}"


def check_simpy() -> str:
    import simpy

    env = simpy.Environment()
    steps_seen = []

    def process(env):
        for _ in range(10):
            steps_seen.append(env.now)
            yield env.timeout(1)

    env.process(process(env))
    env.run()
    assert env.now == 10, env.now
    return f"10-step process ran, env.now={env.now}, steps={steps_seen}"


def main() -> int:
    print(f"Python: {sys.executable}")
    print(f"Version: {sys.version}")
    print(f"Platform: {platform.platform()}")

    run_check("versions", check_versions)
    run_check("pandapower", check_pandapower)
    run_check("pgmpy", check_pgmpy)
    run_check("torch_geometric", check_torch_geometric)
    run_check("simpy", check_simpy)

    print("\n=== SUMMARY ===")
    print(f"PASS: {RESULTS.passed}")
    print(f"FAIL: {RESULTS.failed}")

    return 1 if RESULTS.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
