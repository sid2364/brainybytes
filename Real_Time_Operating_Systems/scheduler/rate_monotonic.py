from entities import TaskSet
from helpers import is_utilisation_lt_69, is_utilisation_gt_1, get_delta_t

"""
Exit code
0 The task set is schedulable and you had to simulate the execution.
1 The task set is schedulable and you took a shortcut.
2 The task set is not schedulable and you had to simulate the execution.
3 The task set is not schedulable and you took a shortcut
"""

"""
Rate Monotonic Priority Assignment
"""

def rate_monotonic_sorted(tasks: list):
    # sort the tasks by period and return the task with the smallest period
    if len(tasks) == 0:
        return None
    return sorted(tasks, key=lambda x: x.period)

def rate_monotonic_top_priority(tasks: list):
    sorted_tasks = rate_monotonic_sorted(tasks)
    if sorted_tasks is None:
        return None
    return sorted_tasks[0]

def is_rate_monotonic_schedule_able(task_set: TaskSet):
    if is_utilisation_lt_69(task_set):
        return 1
    if is_utilisation_gt_1(task_set):
        return 3
    # Otherwise, we need to simulate the execution
    # TODO: Implement the simulation

    return 0

"""
Rate Monotonic Scheduling
"""
def rate_monotonic_schedule(task_set: TaskSet, time_max: int):
    """
    Rate Monotonic Scheduling
    1. Check for deadline misses
    2. Check if the previous cycle job is finished, if so, remove it from the active tasks
    3. Check for new job releases
    4. Execute the highest priority job that is released and not finished for one time_unit

    Return values:
        1 if all jobs finish on time till time_max
        0 if a deadline is missed

    """
    t = 0
    time_step = get_delta_t(task_set)
    current_jobs = []
    active_tasks = []

    # Since we're dealing with uni-processors, we can only execute one job at a time!
    previous_cycle_job = None
    previous_cycle_task = None

    while t < time_max:
        # Check for deadline misses
        print(f"Time {t}-{t+time_step}:")
        for task in task_set.tasks:
            for job in task.jobs:
                if job.deadline_missed(t):
                    print(f"Deadline missed for job {job} at time {t}")
                    return 0
        # Check if the previous cycle job is finished
        if previous_cycle_job is not None and previous_cycle_task is not None:
            if previous_cycle_job.is_finished():
                print(f"Finished {previous_cycle_job} at time {t}")
                active_tasks.remove(previous_cycle_job.task)
                current_jobs.remove(previous_cycle_job)
                previous_cycle_task.finish_job(previous_cycle_job)
                previous_cycle_job = None
        # Check for new job releases
        for task in task_set.tasks:
            job = task.release_job(t)
            current_jobs.append(job)
            if job is not None:
                active_tasks.append(task)
                #print(f"{job} released at time {t}")

        # Execute the highest priority job that is released and not finished for one time_unit
        highest_priority_task = rate_monotonic_top_priority(active_tasks)
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
    return 1 # All jobs finished on time till time_max
