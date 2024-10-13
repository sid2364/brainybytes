from entities import TaskSet
from abc import ABC, abstractmethod

class Scheduler(ABC):
    # Base interface for all schedulers: RateMonotonic, EarliestDeadlineFirst, RoundRobin
    def schedule(self, task_set: TaskSet, time_max: int):
        pass

    @abstractmethod
    def is_schedulable(self, task_set: TaskSet):
        pass

