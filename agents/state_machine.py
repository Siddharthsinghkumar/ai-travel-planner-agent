from enum import Enum


class PlannerState(Enum):
    IDLE = "idle"
    INTENT_PARSING = "intent_parsing"
    PLANNING = "planning"
    PENDING_APPROVAL = "pending_approval"
    EXECUTING = "executing"
    COMPLETE = "complete"
    REJECTED = "rejected"
    ERROR = "error"


VALID_TRANSITIONS = {
    PlannerState.IDLE: {PlannerState.INTENT_PARSING, PlannerState.ERROR},
    PlannerState.INTENT_PARSING: {PlannerState.PLANNING, PlannerState.ERROR, PlannerState.IDLE},
    PlannerState.PLANNING: {PlannerState.PENDING_APPROVAL, PlannerState.EXECUTING, PlannerState.ERROR},
    PlannerState.PENDING_APPROVAL: {PlannerState.EXECUTING, PlannerState.REJECTED, PlannerState.ERROR},
    PlannerState.EXECUTING: {PlannerState.COMPLETE, PlannerState.ERROR},
    PlannerState.ERROR: {PlannerState.IDLE, PlannerState.INTENT_PARSING},
    PlannerState.COMPLETE: set(),
    PlannerState.REJECTED: set(),
}


class IllegalTransition(Exception):
    pass


def transition(current: PlannerState, target: PlannerState) -> PlannerState:
    if target not in VALID_TRANSITIONS[current]:
        raise IllegalTransition(
            f"Cannot transition {current.value} -> {target.value}"
        )
    return target
