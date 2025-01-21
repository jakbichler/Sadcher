import argparse
import numpy as np
import sys
import yaml
sys.path.append('..')

from data_generation.problem_generator import ProblemData, generate_random_data, generate_simple_data
from display_simulation import visualize
from schedulers.greedy_instantaneous_scheduler import greedy_instantaneous_assignment
from helper_functions.schedules import Full_Horizon_Schedule
from visualizations.solution_visualization import plot_gantt_chart

class Task:
    def __init__(self, task_id, location, duration, requirements):
        self.task_id = task_id
        self.location = np.array(location, dtype=np.float64)
        self.duration = duration
        self.requirements = np.array(requirements, dtype=bool)
        self.status = 'PENDING'  if task_id != 0 else 'DONE' # could be PENDING, IN_PROGRESS, DONE
        self.ready = False


    def start(self):
        self.status = 'IN_PROGRESS'

    def complete(self):
        self.status = 'DONE'

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

class Simulation:
    def __init__(self, problem_instance, precedence_constraints):
        self.timestep = 0
        self.robots: list[Robot] = self.create_robots(problem_instance)
        self.tasks: list[Task] = self.create_tasks(problem_instance)
        self.precedence_constraints = precedence_constraints
        self.sim_done = False
        self.makespan = -1 
        self.robot_schedules = {robot.robot_id: [] for robot in self.robots}
        self.n_tasks = len(self.tasks)
        self.last_task_id = self.n_tasks - 1

    def create_robots(self, problem_instance):
        # For example, Q is a list of robot capabilities
        robot_capabilities = problem_instance['Q']
        start_location = problem_instance['task_locations'][0]
        return [Robot(robot_id = idx, location=start_location, capabilities=cap) 
                for idx,cap in enumerate(robot_capabilities)]

    def create_tasks(self, problem_instance):
        locations = problem_instance['task_locations']
        durations = problem_instance['T_e']
        requirements = problem_instance['R']
        return [
            Task(idx, loc, dur, req) 
            for idx, (loc, dur, req) in enumerate(zip(locations, durations, requirements))
        ]
        
    def update_robot(self, robot):
        robot.update_position()
        robot.check_task_status()

    def update_task(self, task):
        previous_status = task.status

        # Check if task is ready to start based on precedence constraints
        if task.task_id == 0 or task.predecessors_completed(self):
            task.ready = True
        else:
            task.ready = False

        if task.status == 'DONE':
            return

        # Special handling of last task
        if task.task_id == len(self.tasks) - 1:
            if self.all_robots_at_exit_location(threshold=0.01):
                task.status = 'DONE'
                self.finish_simulation()
            else:
                task.status = 'PENDING'
            return

        # Normal tasks
        if self.all_skills_assigned(task) and self.all_robots_at_task(task, threshold=0.01):
            if task.status == 'PENDING':
                task.start()
            task.decrement_duration()
        else:
            task.status = 'PENDING'

        self.log_into_full_horizon_schedule(task, previous_status)

    def finish_simulation(self):
        self.sim_done = True
        self.makespan = self.timestep
    
    def all_skills_assigned(self, task):
        """
        Returns True if:
        1) The logical OR of all assigned robots' capabilities covers all task requirements.
        2) Every assigned robot is within 1 unit of the task location.
        Otherwise, returns False.
        """
        assigned_robots = [r for r in self.robots if r.current_task == task]

        # Combine capabilities across all assigned robots
        combined_capabilities = np.zeros_like(task.requirements, dtype=bool)
        for robot in assigned_robots:
            robot_cap = np.array(robot.capabilities, dtype=bool)
            combined_capabilities = np.logical_or(combined_capabilities, robot_cap)

        required_skills = np.array(task.requirements, dtype=bool)
        # Check if the combined team covers all required skills
        return np.all(combined_capabilities[required_skills])

    def all_robots_at_task(self, task, threshold=0.01):
        """True if all robots are within 'threshold' distance of 'task' location."""
        assigned_robots = [r for r in self.robots if r.current_task == task]
        if not assigned_robots:
            return False
        
        for r in assigned_robots:
            if np.linalg.norm(r.location - task.location) > threshold:
                return False
        return True  

    def all_robots_at_exit_location(self, threshold=0.01):
        """True if all robots are within 'threshold' distance of the exit location."""
        exit_location = self.tasks[-1].location
        for r in self.robots:
            if np.linalg.norm(r.location - exit_location) > threshold:
                return False
        return True    


    def log_into_full_horizon_schedule(self, task, previous_status):
        # Check for transition from PENDING -> IN_PROGRESS: log start time
        if previous_status == 'PENDING' and task.status == 'IN_PROGRESS':
            for r in [rb for rb in self.robots if rb.current_task == task]:
                self.robot_schedules[r.robot_id].append((task.task_id, self.timestep, None))

        # Check for transition from IN_PROGRESS -> DONE: log end time
        if previous_status == 'IN_PROGRESS' and task.status == 'DONE':
            for r in [rb for rb in self.robots if rb.current_task == task]:
                tid, start, _ = self.robot_schedules[r.robot_id][-1]
                if tid == task.task_id:
                    self.robot_schedules[r.robot_id][-1] = (tid, start, self.timestep)

    def step(self):
        """Advance the simulation by one timestep, moving robots and updating tasks."""
        for robot in self.robots:   
            self.update_robot(robot)

        for task in self.tasks:
            self.update_task(task)

        idle_robots = [r for r in sim.robots if not r.current_task or r.current_task.status == 'DONE']

        if idle_robots:
            instantaneous_assignment = greedy_instantaneous_assignment(sim)
            assign_tasks_to_robots(instantaneous_assignment, sim.robots)

        self.timestep += 1

def assign_tasks_to_robots(instantaneous_schedule, robots):
    """
    Example scheduling logic:
      - Check for any idle robots
      - Assign them tasks if any are PENDING
    This could be replaced by a call to your NN or heuristic.
    """
    for robot in robots:
        task_id = instantaneous_schedule.robot_assignments.get(robot.robot_id)
        if task_id is not None:
            robot.current_task = sim.tasks[task_id]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualize", action='store_true', help="Visualize the simulation")
    args = parser.parse_args()

    with open("simulation_config.yaml", "r") as file:
        config = yaml.safe_load(file)

    n_tasks = config["n_tasks"]
    n_robots = config["n_robots"]
    n_skills = config["n_skills"]
    np.random.seed(config["random_seed"])
    precedence_constraints = config["precedence_constraints"]
    problem_instance: ProblemData = generate_random_data(n_tasks, n_robots, n_skills, precedence_constraints)

    sim = Simulation(problem_instance, precedence_constraints)
    
    if args.visualize: 
        visualize(sim)

    else:
        while not sim.sim_done:
            sim.step()

    rolled_out_schedule = Full_Horizon_Schedule(sim.makespan, sim.robot_schedules, n_tasks)
    print(rolled_out_schedule)
    plot_gantt_chart("Greedily Rolled-Out Schedule", rolled_out_schedule)
