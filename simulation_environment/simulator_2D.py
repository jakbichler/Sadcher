import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import sys
sys.path.append('..')

from data_generation.problem_generator import ProblemData, generate_random_data, generate_simple_data

class Task:
    def __init__(self, location, duration, requirements):
        self.location = np.array(location, dtype=np.float64)
        self.duration = duration
        self.requirements = requirements
        self.status = 'PENDING'  # could be PENDING, IN_PROGRESS, DONE

class Robot:
    def __init__(self, location, speed=1.0, capabilities=None):
        if capabilities is None:
            capabilities = [0, 0]
        self.location = np.array(location, dtype=np.float64)
        self.speed = speed
        self.capabilities = capabilities
        self.current_task = None
    
    def move_towards(self):
        """Move the robot one step toward its current_task if assigned."""
        if self.current_task:
            # Simple step in the direction of goal based on speed
            movement_vector = self.current_task.location - self.location
            dist = np.linalg.norm(movement_vector)
            if dist > 1e-9:  # avoid division by zero
                normalized_mv = movement_vector / dist
                self.location += normalized_mv * self.speed

class Simulation:
    def __init__(self, problem_instance):
        self.timestep = 0
        self.robots = self.create_robots(problem_instance)
        self.tasks = self.create_tasks(problem_instance)

    def create_robots(self, problem_instance):
        # For example, Q is a list of robot capabilities
        robot_capabilities = problem_instance['Q']
        start_location = problem_instance['task_locations'][0]
        robots = []
        for cap in robot_capabilities:
            robots.append(Robot(location=start_location, capabilities=cap))
        return robots

    def create_tasks(self, problem_instance):
        # T_e = durations, R = requirements
        locations = problem_instance['task_locations']
        durations = problem_instance['T_e']
        requirements = problem_instance['R']
        return [
            Task(loc, dur, req) 
            for loc, dur, req in zip(locations, durations, requirements)
        ]

    def step(self):
        """Advance the simulation by one timestep, moving robots and updating tasks."""
        self.timestep += 1

        # Move robots
        for robot in self.robots:
            robot.move_towards()

        # Update tasks (if in progress, decrement duration, mark done if complete)
        for task in self.tasks:
            if task.status == 'IN_PROGRESS':
                task.duration -= 1
                if task.duration <= 0:
                    task.status = 'DONE'

        # Check if any robot is idle and there are pending tasks
        idle_robots = [r for r in sim.robots if not r.current_task or r.current_task.status == 'DONE']

        if idle_robots:
            schedule_tasks(sim)


def schedule_tasks(sim):
    """
    Example scheduling logic:
      - Check for any idle robots
      - Assign them tasks if any are PENDING
    This could be replaced by a call to your NN or heuristic.
    """
    if sim.timestep > 3:
        sim.robots[0].current_task = sim.tasks[1]

    if sim.timestep > 5:
        sim.robots[1].current_task = sim.tasks[2]


def visualize(sim):
    """Interactive Matplotlib figure with a Next Timestep button."""
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(bottom=0.2)

    def update_plot():
        ax.clear()

        # Plot tasks
        task_x = [t.location[0] for t in sim.tasks if t.status != 'DONE']
        task_y = [t.location[1] for t in sim.tasks if t.status != 'DONE']
        ax.scatter(task_x, task_y, marker='o', c='blue', label='Tasks')

        # Plot robots
        robot_x = [r.location[0] for r in sim.robots]
        robot_y = [r.location[1] for r in sim.robots]
        ax.scatter(robot_x, robot_y, marker='s', c='red', label='Robots')

        ax.set_title(f"Timestep: {sim.timestep}")
        ax.legend()
        plt.draw()

    def next_step_callback(event):
        # Advance one timestep and update
        sim.step()
        update_plot()

    def advance_10_steps_callback(event):
        # Advance 10 timesteps
        for _ in range(10):
            sim.step()
        update_plot()

    # Create buttons
    ax_button_next = plt.axes([0.5, 0.05, 0.2, 0.07])  # Position for 'Next Timestep' button
    ax_button_10 = plt.axes([0.7, 0.05, 0.2, 0.07])    # Position for 'Advance 10 Timesteps' button

    btn_next = Button(ax_button_next, 'Next Timestep')
    btn_next.on_clicked(next_step_callback)

    btn_advance_10 = Button(ax_button_10, '10 Timesteps')
    btn_advance_10.on_clicked(advance_10_steps_callback)

    # Initial draw
    update_plot()
    plt.show()


def simulation_main_loop(sim):
    """
    This wraps the visualization. 
    You can later add code here for logging or additional checks, 
    but the main interactive loop is driven by the Matplotlib UI.
    """
    visualize(sim)


if __name__ == '__main__':
    # Example usage
    n_tasks = 8
    n_robots = 4
    n_skills = 3
    # np.random.seed(35)

    precedence_constraints = np.array([[1,2]])
    problem_instance: ProblemData = generate_random_data(n_tasks, n_robots, n_skills, precedence_constraints)

    sim = Simulation(problem_instance)
    simulation_main_loop(sim)

