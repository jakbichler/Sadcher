import numpy as np


class Task:
    def __init__(self, task_id, location, duration, requirements):
        self.task_id = task_id
        self.location = np.array(location, dtype=np.float64)
        self.duration = duration
        self.requirements = np.array(requirements, dtype=bool)
        self.status = 'PENDING'  if task_id != 0 else 'DONE' # could be PENDING, IN_PROGRESS, DONE
        self.ready = True
        self.assigned = False
        self.incomplete = True if task_id != 0 else False

    def start(self):
        self.status = 'IN_PROGRESS'
        self.assigned = True

    def complete(self):
        self.status = 'DONE'
        self.assigned = False
        self.incomplete = False

    def decrement_duration(self):
        self.duration -= 1
        if self.duration <= 0:
            self.complete()

    def predecessors_completed(self, sim):
        if sim.precedence_constraints is None:
            return True
        
        predecessors = [j for (j, k) in sim.precedence_constraints if k == self.task_id]
        preceding_tasks = [t for t in sim.tasks if t.task_id in predecessors] 
        for preceding_task in preceding_tasks:
            if preceding_task.status != 'DONE':
                return False
        return True
    
    def feature_vector(self):
        return np.concatenate([self.location/100, self.requirements, np.array([self.ready, self.assigned, self.incomplete])], dtype=float)


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
    
    def update_position(self):
        """Move the robot one step toward its current_task if assigned."""
        if self.current_task:
            # Simple step in the direction of goal based on speed
            movement_vector = self.current_task.location - self.location
            dist = np.linalg.norm(movement_vector)
            if dist > self.speed:  
                normalized_mv = movement_vector / dist + 1e-9
                self.location += normalized_mv * self.speed
            else: # Arrived at task
                self.location = np.copy(self.current_task.location)

    def check_task_status(self):
        if self.current_task and self.current_task.status == 'DONE':
            self.current_task = None
        
        self.available = self.current_task is None

    def feature_vector(self):
        return np.concatenate([self.location/100, self.capabilities, np.array([self.available])], dtype=float)