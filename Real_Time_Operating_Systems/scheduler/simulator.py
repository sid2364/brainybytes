from entities import TaskSet, Task, Job
from rate_monotonic import rate_monotonic_schedule

import csv
import argparse

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
    rate_monotonic_schedule(task_set, 24)

def parse_task_file(file_path):
    tasks = []
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) == 4:
                offset = int(row[0])
                computation_time = int(row[1])
                deadline = int(row[2])
                period = int(row[3])
                task = Task(offset, computation_time, deadline, period)
                tasks.append(task)
    task_set = TaskSet(tasks)
    return task_set


def main():
    parser = argparse.ArgumentParser(description='Select a scheduling algorithm and specify a task set file.')

    # Add the scheduling algorithm argument (mandatory)
    parser.add_argument('algorithm', choices=['rm', 'dm', 'audsley', 'edf', 'rr'],
                        help='Scheduling algorithm to use: dm, audsley, edf, or rr.')

    # Add the task set file argument (mandatory)
    parser.add_argument('task_set_file', type=str,
                        help='Path to the task set file.')

    # Parse the arguments
    args = parser.parse_args()

    # Access the parameters
    algorithm = args.algorithm.lower()
    task_set_file = args.task_set_file

    task_set = parse_task_file(task_set_file)

    if algorithm == 'rm':
        rate_monotonic_schedule(task_set, 24)
    elif algorithm == 'dm' or algorithm == 'audsley':
        print("DM/Audsley scheduling")
    elif algorithm == 'edf':
        print("EDF scheduling")
    elif algorithm == 'rr':
        print("Round Robin scheduling")

if __name__ == "__main__":
    #test_release_job()
    #test_rate_monotonic_schedule() #use argparse for passing in a task_set

    main()