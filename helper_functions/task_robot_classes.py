import numpy as np


class Task:
    def __init__(self, task_id, location, duration, requirements):
        self.task_id = task_id
        self.location = np.array(location, dtype=np.float64)
        self.overall_duration = duration
        self.remaining_duration = duration
        self.requirements = np.array(requirements, dtype=bool)
        self.status = "PENDING" if task_id != 0 else "DONE"  # could be PENDING, IN_PROGRESS, DONE
        self.ready = True
        self.assigned = False
        self.incomplete = True if task_id != 0 else False

    def start(self):
        self.status = "IN_PROGRESS"
        self.assigned = True

    def complete(self):
        self.status = "DONE"
        self.assigned = False
        self.incomplete = False

    def decrement_duration(self):
        self.remaining_duration -= 1.0
        if self.remaining_duration <= 0:
            self.complete()

    def predecessors_completed(self, sim):
        if sim.precedence_constraints is None:
            return True

        predecessors = [j for (j, k) in sim.precedence_constraints if k == self.task_id]
        preceding_tasks = [t for t in sim.tasks if t.task_id in predecessors]
        for preceding_task in preceding_tasks:
            if preceding_task.status != "DONE":
                return False
        return True

    def feature_vector(self, location_normalization=100, duration_normalization=100):
        return np.concatenate(
            [
                np.atleast_1d(self.location / location_normalization),
                np.atleast_1d(self.overall_duration / duration_normalization),
                np.atleast_1d(self.requirements),
                np.array([self.ready, self.assigned, self.incomplete], dtype=float),
            ],
            dtype=float,
        )


class Robot:
    def __init__(self, robot_id, location, speed=1.0, capabilities=None):
        self.robot_id = robot_id
        if capabilities is None:
            capabilities = [0, 0]
        self.location = np.array(location, dtype=np.float64)
        self.speed = speed
        self.capabilities = np.array(capabilities, dtype=bool)
        self.current_task = None
        self.available = True
        self.remaining_workload = 0

    def update_position_on_task(self):
        """Move the robot one step toward its current_task if assigned."""
        if self.current_task:
            self.position_towards_task(self.current_task)

    def position_towards_task(self, task):
        movement_vector = task.location - self.location
        dist = np.linalg.norm(movement_vector)
        if dist > self.speed:
            normalized_mv = movement_vector / dist + 1e-9
            self.location += normalized_mv * self.speed
        else:
            self.location = np.copy(task.location)

    def check_task_status(self, idle_task_id):
        if (
            self.current_task
            and self.current_task.status == "DONE"
            and not self.current_task.task_id == idle_task_id
        ):
            self.current_task = None

        self.available = self.current_task is None or self.current_task.task_id == idle_task_id
        self.remaining_workload = (
            0 if self.current_task is None else self.current_task.remaining_duration
        )

    def feature_vector(self, location_normalization_factor=100, workload_normalization_factor=100):
        return np.concatenate(
            [
                np.atleast_1d(self.location / location_normalization_factor),
                np.atleast_1d(self.remaining_workload / workload_normalization_factor),
                np.atleast_1d(self.capabilities),
                np.array([self.available], dtype=float),
            ],
            dtype=float,
        )
