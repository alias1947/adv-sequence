"""Definition of EventList class.

This module defines the EventList class, used by the timeline to order and execute events.
EventList is implemented as a min heap ordered by simulation time.
"""

from typing import TYPE_CHECKING, List
import time
from heapq import heappush, heappop, merge

if TYPE_CHECKING:
    from .event import Event

class EventList:
    """Class of event list.

    This class is implemented as a min-heap with event batching for better performance.
    The event with the lowest time and priority is placed at the top of heap.

    Attributes:
        data (List[Event]): heap storing events.
        batch_size (int): number of events to batch before merging with main queue
        current_batch (List[Event]): temporary storage for batched events
        batch_time_threshold (float): time threshold for batching similar events
    """

    def __init__(self, batch_size: int = 1000, batch_time_threshold: float = 1e-6):
        self.data = []
        self.batch_size = batch_size
        self.current_batch = []
        self.batch_time_threshold = batch_time_threshold
        self._last_batch_time = time.time()
        self._batch_count = 0
        self._total_events = 0
        self._batch_merge_time = 0

    def __len__(self):
        return len(self.data) + len(self.current_batch)

    def __iter__(self):
        for data in self.data:
            yield data
        for data in self.current_batch:
            yield data

    def push(self, event: "Event") -> None:
        """Add an event to the queue with batching optimization."""
        self._total_events += 1
        self.current_batch.append(event)
        
        # Check if we should flush the batch
        current_time = time.time()
        if (len(self.current_batch) >= self.batch_size or 
            current_time - self._last_batch_time > self.batch_time_threshold):
            self._flush_batch()

    def _flush_batch(self) -> None:
        """Merge the current batch with the main queue."""
        if not self.current_batch:
            return
            
        start_time = time.time()
        self._batch_count += 1
        
        # Sort the batch by time and priority
        self.current_batch.sort(key=lambda x: (x.time, x.priority))
        
        # Merge with main queue
        self.data = list(merge(self.data, self.current_batch, key=lambda x: (x.time, x.priority)))
        self.current_batch = []
        self._last_batch_time = time.time()
        self._batch_merge_time += time.time() - start_time

    def pop(self) -> "Event":
        """Remove and return the next event from the queue."""
        # Ensure any pending events are merged
        if self.current_batch:
            self._flush_batch()
            
        if not self.data:
            raise IndexError("pop from empty event list")
            
        return heappop(self.data)

    def top(self) -> "Event":
        """Get the next event without removing it."""
        # Ensure any pending events are merged
        if self.current_batch:
            self._flush_batch()
            
        if not self.data:
            raise IndexError("top from empty event list")
            
        return self.data[0]

    def isempty(self) -> bool:
        """Check if the event list is empty."""
        return len(self.data) == 0 and len(self.current_batch) == 0

    def remove(self, event: "Event") -> None:
        """Method to remove events from heap.

        The event is set as the invalid state to save the time of removing event from heap.
        """
        event.set_invalid()

    def update_event_time(self, event: "Event", time: int):
        """Method to update the timestamp of event and maintain the min-heap structure."""
        if time == event.time:
            return

        def _pop_updated_event(heap: "List", index: int):
            parent_i = (index - 1) // 2
            while index > 0 and event < self.data[parent_i]:
                heap[index], heap[parent_i] = heap[parent_i], heap[index]
                index = parent_i
                parent_i = (parent_i - 1) // 2

        # Flush any pending events first
        if self.current_batch:
            self._flush_batch()

        for i, e in enumerate(self.data):
            if id(e) == id(event):
                if event.time > time:
                    event.time = time
                    _pop_updated_event(self.data, i)
                elif event.time < time:
                    event.time = -1
                    _pop_updated_event(self.data, i)
                    self.pop()
                    event.time = time
                    self.push(event)
                break

    def get_stats(self) -> dict:
        """Get performance statistics for the event list."""
        return {
            "total_events": self._total_events,
            "batch_count": self._batch_count,
            "batch_merge_time": self._batch_merge_time,
            "avg_batch_size": self._total_events / max(1, self._batch_count),
            "current_queue_size": len(self.data),
            "current_batch_size": len(self.current_batch)
        }
