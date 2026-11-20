"""Orchestration for the junction check endpoint."""

from __future__ import annotations

from .imaging import DecodedImage
from .junctions import JunctionInputError, TerminalRecord, interpret
from .models import CrossingVerdictModel, JunctionResponse, JunctionSpec


def check_junctions(image: DecodedImage, spec: JunctionSpec) -> JunctionResponse:
    """Run the whole-image junction interpretation for one sketch."""
    crossings = [(p.x, p.y) for p in spec.crossings]
    records = [
        TerminalRecord(a=(t.a.x, t.a.y), b=(t.b.x, t.b.y), connected=t.connected)
        for t in spec.terminals
    ]
    # Validation depends on the actual image size, so it happens here rather
    # than in the Pydantic model: a 100x100 sketch cannot contain (200, 50).
    interpretation = interpret(image.array, crossings, records)
    return JunctionResponse(
        status=interpretation.status,
        message=interpretation.message,
        consistent_assignments=interpretation.consistent_assignments,
        crossings=[
            CrossingVerdictModel(x=v.x, y=v.y, state=v.state)
            for v in interpretation.verdicts
        ],
    )


__all__ = ["JunctionInputError", "check_junctions"]
