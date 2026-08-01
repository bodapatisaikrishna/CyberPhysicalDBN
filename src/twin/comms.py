"""Abstract control-centre <-> DER message exchange (CLAUDE.md layer [0]).

Deliberately NOT a protocol stack. CLAUDE.md lists under rejected directions:
"Real IEC 61850/MMS protocol stack. Abstract communication as SimPy events.
No claim in this project is about protocol-level evasion." So there is no
framing, no ASDUs, no TLS, and no addressing beyond string endpoint names.
What is modelled is exactly what the attack graph needs: messages that carry a
category, and hooks that let a compromised path alter them in transit.

The manipulations correspond one-to-one with attack-graph steps:

  MITM            passive interception -- stamps provenance, alters nothing,
                  and is the precondition that makes the next two reachable
  SpoofRepMsg     rewrites a MEASUREMENT so the control centre's view is false
  UnauthCommand   injects a COMMAND the control centre never issued
  ModCtrlLogic /  DER-side logic modification: an arriving SETPOINT is executed
  WrongLogicExec  as something other than what was sent
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Mapping

import simpy

from src.twin.grid import ActionOrigin


class MessageType(Enum):
    MEASUREMENT = "measurement"  # DER -> control centre
    COMMAND = "command"  # control centre -> DER
    CONFIG = "config"  # control centre -> DER, logic/parameter update


class PayloadCategory(Enum):
    VOLTAGE_REPORT = "voltage_report"
    SETPOINT = "setpoint"
    LOGIC_UPDATE = "logic_update"


@dataclass(frozen=True)
class Message:
    msg_type: MessageType
    sender: str
    receiver: str
    payload_category: PayloadCategory
    payload: Mapping[str, float | str]
    timestamp: float  # twin time units
    origin: ActionOrigin = ActionOrigin.OPERATOR
    tampered_by: tuple[str, ...] = ()


# A hook returns the (possibly rewritten) message, or None to drop it.
Manipulation = Callable[[Message], "Message | None"]


@dataclass
class DeliveryRecord:
    """One message's journey, for the trace. Session 4 needs this."""

    sent: Message
    delivered: Message | None  # None if a manipulation dropped it


class CommsBus:
    """SimPy message bus with ordered, named in-transit manipulations."""

    def __init__(self, env: simpy.Environment, latency_time_units: float = 0.0):
        self.env = env
        # Transport delay is NOT MODELLED -- see configs/twin.yaml. A nonzero
        # default would be an invented number; the knob exists only so a
        # deliberate sensitivity study can vary it.
        self.latency_time_units = float(latency_time_units)
        self._stores: dict[str, simpy.Store] = {}
        self._manipulations: dict[str, Manipulation] = {}
        self._log: list[DeliveryRecord] = []

    # --- manipulation registry -------------------------------------------

    def register_manipulation(self, name: str, hook: Manipulation) -> None:
        """Install a hook. Hooks apply in registration order."""
        self._manipulations[name] = hook

    def unregister_manipulation(self, name: str) -> None:
        self._manipulations.pop(name, None)

    @property
    def active_manipulations(self) -> tuple[str, ...]:
        return tuple(self._manipulations)

    # --- transport --------------------------------------------------------

    def subscribe(self, endpoint: str) -> simpy.Store:
        if endpoint not in self._stores:
            self._stores[endpoint] = simpy.Store(self.env)
        return self._stores[endpoint]

    def send(self, message: Message) -> None:
        self.env.process(self._deliver(message))

    def _apply_manipulations(self, message: Message) -> Message | None:
        current: Message | None = message
        for name, hook in self._manipulations.items():
            if current is None:
                return None
            result = hook(current)
            if result is None:
                return None
            if result is not current:
                result = replace(result, tampered_by=result.tampered_by + (name,))
            current = result
        return current

    def _deliver(self, message: Message):
        if self.latency_time_units > 0.0:
            yield self.env.timeout(self.latency_time_units)
        else:
            yield self.env.timeout(0)

        delivered = self._apply_manipulations(message)
        self._log.append(DeliveryRecord(sent=message, delivered=delivered))
        if delivered is not None:
            self.subscribe(delivered.receiver).put(delivered)

    @property
    def log(self) -> tuple[DeliveryRecord, ...]:
        return tuple(self._log)


# --- the four manipulations the attack graph requires ---------------------


def mitm_intercept() -> Manipulation:
    """MITM: passive. Marks the message as intercepted, alters nothing.

    Registering this changes no payload -- it records that the attacker holds
    the position that makes SpoofRepMsg and UnauthCommand possible, which is
    exactly MITM's role in Fig. 2 (a precondition, not an effect).
    """

    def hook(message: Message) -> Message:
        return replace(message, payload=dict(message.payload))

    return hook


def spoof_reporting_message(reported_vm_pu: float) -> Manipulation:
    """SpoofRepMsg: rewrite a voltage report so the control centre sees false data.

    Only MEASUREMENT/VOLTAGE_REPORT traffic is touched; commands pass through.
    The spoofed value is driven low so the control centre believes the feeder
    is undervolted and responds by RAISING DER injection -- which, on the real
    grid, drives it toward overvoltage. This is the false-data-injection
    mechanism behind CorrReact: see the note on CorrReact's direction below.
    """

    def hook(message: Message) -> Message:
        if (
            message.msg_type is not MessageType.MEASUREMENT
            or message.payload_category is not PayloadCategory.VOLTAGE_REPORT
        ):
            return message
        payload = dict(message.payload)
        payload["vm_pu_min"] = float(reported_vm_pu)
        payload["spoofed"] = "true"
        return replace(message, payload=payload)

    return hook


def unauthorized_command(der_ids: tuple[str, ...], p_mw: float) -> Manipulation:
    """UnauthCommand: force any setpoint reaching a DER to the attacker's value.

    Modelled as an in-transit rewrite rather than a separately injected
    message so that the ground truth stays unambiguous: every COMMAND the DER
    executes is in the bus log with its `tampered_by` provenance.
    """

    def hook(message: Message) -> Message:
        if (
            message.msg_type is not MessageType.COMMAND
            or message.payload_category is not PayloadCategory.SETPOINT
            or message.receiver not in der_ids
        ):
            return message
        payload = dict(message.payload)
        payload["p_mw"] = float(p_mw)
        return replace(message, payload=payload, origin=ActionOrigin.ATTACKER)

    return hook


def modified_control_logic(der_ids: tuple[str, ...], p_mw: float) -> Manipulation:
    """ModCtrlLogic/WrongLogicExec: the DER executes something other than what was sent.

    Distinct from UnauthCommand in mechanism though similar in effect: here the
    command on the wire is legitimate and the *device* misinterprets it, which
    is the Stuxnet-style branch of Fig. 2 (left). Kept separate so the trace
    attributes the physical consequence to the right attack step.
    """

    def hook(message: Message) -> Message:
        if (
            message.msg_type is not MessageType.COMMAND
            or message.payload_category is not PayloadCategory.SETPOINT
            or message.receiver not in der_ids
        ):
            return message
        payload = dict(message.payload)
        payload["p_mw"] = float(p_mw)
        payload["logic_modified"] = "true"
        return replace(message, payload=payload)

    return hook
