"""
CUDA graph timer
"""
from __future__ import annotations
import typing
__all__: list[str] = ['EventHandle', 'EventRecordNodeHandle', 'GraphHandle']
class EventHandle:
    def __eq__(self, arg0: EventHandle) -> bool:
        """
        Whether both handles refer to the same event.
        """
    def __hash__(self) -> int:
        """
        Hash over the event referred to.
        """
    def __init__(self) -> None:
        """
        Create a fresh event.
        """
    def elapsed_time(self, end_event: EventHandle) -> float:
        """
        Milliseconds from this event to one recorded later.
        """
    def query(self) -> bool:
        """
        Whether the event has completed.
        """
    def record(self, stream: typing.SupportsInt | typing.SupportsIndex, capture_external: bool) -> None:
        """
        Record on a stream; capture_external sets the external flag for capture.
        """
    def synchronize(self) -> None:
        """
        Block until the event completes.
        """
class EventRecordNodeHandle:
    def get_event(self) -> EventHandle:
        """
        Event this record node records.
        """
    def set_event(self, event: EventHandle) -> None:
        """
        Swap the event this record node records into.
        """
class GraphHandle:
    def __eq__(self, arg0: GraphHandle) -> bool:
        """
        Whether both handles refer to the same graph.
        """
    def __hash__(self) -> int:
        """
        Hash over the graph referred to.
        """
    def __init__(self, graph: typing.Any) -> None:
        """
        Wrap a captured platform graph; raw handles are extracted per platform.
        """
    def get_event_record_nodes(self) -> list[EventRecordNodeHandle]:
        """
        Event-record nodes of the captured graph.
        """
