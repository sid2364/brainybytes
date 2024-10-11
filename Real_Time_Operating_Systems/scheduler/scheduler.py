from dataclasses import dataclass

@dataclass
class TaskSet:
    tasks: list
    """
    Rate Monotonic Scheduling
    """

    def rate_monotonic_schedule(self, time_max):
        """
        1. check for deadline misses
        2. check for new job releases
        3. if any, schedule the job with the highest priority
        """
        t = 0
        time_step = 1 # optimize this to bigger delta t based on smallest divisible period
        current_jobs = []
        active_tasks = []
        previous_cycle_job = None # since it's uni-processor, we can only execute one job at a time!
        previous_cycle_task = None # same as above

        while t < time_max:
            # check for deadline misses
            print(f"Time {t}:")
            for task in self.tasks:
                for job in task.jobs:
                    if job.deadline_missed(t):
                        print(f"Deadline missed for job {job} at time {t}")
                        return
            # check if the previous cycle job is finished
            if previous_cycle_job is not None and previous_cycle_task is not None:
                if previous_cycle_job.is_finished():
                    print(f"Finished {previous_cycle_job} at time {t}")
                    active_tasks.remove(previous_cycle_job.task)
                    current_jobs.remove(previous_cycle_job)
                    previous_cycle_task.finish_job(previous_cycle_job)
                    previous_cycle_job = None
            # check for new job releases
            for task in self.tasks:
                job = task.release_job(t)
                current_jobs.append(job)
                if job is not None:
                    active_tasks.append(task)
                    #print(f"{job} released at time {t}")

            # execute the highest priority job that is released and not finished for one time_unit
            highest_priority_task = rate_monotonic_highest_priority(active_tasks)
            if highest_priority_task is None:
                print(f"No tasks to schedule at time {t}, idle time!\n")
                t += time_step
                continue

            #print(f"Active tasks: {[f"T{task.task_id}" for task in active_tasks]}")
            #print(f"Highest priority task: T{highest_priority_task.task_id}")
            current_cycle_job = highest_priority_task.get_first_job()
            if current_cycle_job is None:
                print(f"No jobs to schedule at time {t}, idle time!\n")
                t += time_step
                continue

            current_cycle_job.schedule(time_step)

            previous_cycle_task = highest_priority_task

            if current_cycle_job == previous_cycle_job:
                print(f"Same {current_cycle_job} running at time {t}")
            else:
                print(f"Running {current_cycle_job} at time {t}")
                previous_cycle_job = current_cycle_job

            t += time_step
            print()


@dataclass
class Task:
    task_id = 0 # static member that keeps track of count of instances
    offset: int
    computation_time: int
    deadline: int
    period: int

    def __init__(self, offset, computation_time, deadline, period):
        Task.task_id += 1
        self.task_id = Task.task_id
        self.offset = offset
        self.computation_time = computation_time
        self.deadline = deadline
        self.period = period
        self.jobs = []
        self.job_count = 0

    def release_job(self, release_time):
        if release_time < self.offset:
            return None
        if (release_time - self.offset) % self.period == 0:
            job = Job(self, release_time)
            self.job_count += 1
            self.jobs.append(job)
            return job
    def get_first_job(self):
        if len(self.jobs) == 0:
            return None
        return self.jobs[0]
    def finish_job(self, job):
        self.jobs.remove(job)

    def __str__(self):
        return f"T{self.task_id} with period {self.period} and deadline {self.deadline}"

@dataclass
class Job:
    job_id: int
    task: Task
    computation_time_remaining: int
    release_time: int
    def __init__(self, task, release_time):
        self.job_id = task.job_count + 1
        self.task = task
        self.computation_time_remaining = task.computation_time
        self.release_time = release_time

    def deadline_missed(self, current_time):
        return current_time > self.release_time + self.task.deadline

    def schedule(self, for_time):
        # just subtract the computation time from the job, "executing" it for that time
        self.computation_time_remaining -= for_time

    def get_absolute_period(self):
        return self.task.period + self.release_time

    def is_finished(self):
        return self.computation_time_remaining <= 0 # equals should be enough but why not

    def __str__(self):
        #return f"Job {self.job_id} of Task {self.task.task_id} released at {self.release_time}"
        return f"T{self.task.task_id} - J{self.job_id}"


"""
Rate Monotonic Scheduling
"""
def rate_monotonic_sorted(tasks):
    # sort the tasks by period and return the task with the smallest period
    if len(tasks) == 0:
        return None
    return sorted(tasks, key=lambda x: x.period)

def rate_monotonic_highest_priority(tasks):
    sorted_tasks = rate_monotonic_sorted(tasks)
    if sorted_tasks is None:
        return None
    return sorted_tasks[0]

"""
Tests
"""
def test_release_job():
    task0 = Task(0, 2, 5, 10)
    task1 = Task(1, 0, 3, 5)

    assert task0.release_job(0) is not None
    assert task0.release_job(10) is not None

    assert task1.release_job(0) is None
    assert task1.release_job(5) is None
    assert task1.release_job(6) is not None

def test_rate_monotonic_schedule():
    task1 = Task(0, 1, 5, 5)
    task2 = Task(0, 2, 8, 8)
    task3 = Task(0, 1, 10, 10)
    task4 = Task(0, 5, 20, 20)

    task_set = TaskSet([task1, task2, task3, task4])
    task_set.rate_monotonic_schedule(24)

if __name__ == "__main__":
    #test_release_job()
    test_rate_monotonic_schedule() #use argparse for passing in a task_set
    print("All tests pass")