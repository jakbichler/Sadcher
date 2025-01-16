import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.table import Table
from matplotlib.patches import Wedge
from matplotlib.widgets import Button
import sys
sys.path.append('..')

from data_generation.problem_generator import ProblemData, generate_random_data, generate_simple_data
from greedy_instantaneous_scheduler import greedy_instantaneous_assignment
from helper_functions.schedules import Full_Horizon_Schedule
from visualizations.solution_visualization import plot_gantt_chart, plot_robot_trajectories
class Task:
    def __init__(self, task_id, location, duration, requirements):
        self.task_id = task_id
        self.location = np.array(location, dtype=np.float64)
        self.duration = duration
        self.requirements = np.array(requirements, dtype=bool)
        self.status = 'PENDING'  # could be PENDING, IN_PROGRESS, DONE


class Robot:
    def __init__(self, robot_id, location, speed=1.0, capabilities=None):
        self.robot_id = robot_id
        if capabilities is None:
            capabilities = [0, 0]
        self.location = np.array(location, dtype=np.float64)
        self.speed = speed
        self.capabilities = np.array(capabilities, dtype=bool)
        self.current_task = None
    
    
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
                self.location = self.current_task.location

class Simulation:
    def __init__(self, problem_instance, precedence_constraints):
        self.timestep = 0
        self.robots: list[Robot] = self.create_robots(problem_instance)
        self.tasks: list[Task] = self.create_tasks(problem_instance)
        self.precedence_constraints = precedence_constraints
        self.sim_done = False
        self.makespan = -1 
        self.robot_schedules = {robot.robot_id: [] for robot in self.robots}

    def create_robots(self, problem_instance):
        # For example, Q is a list of robot capabilities
        robot_capabilities = problem_instance['Q']
        start_location = problem_instance['task_locations'][0]
        return [Robot(robot_id = idx, location=start_location, capabilities=cap) for idx,cap in enumerate(robot_capabilities)]


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
        if robot.current_task is not None and robot.current_task.status == 'DONE':
            robot.current_task = None    

    def update_task(self, task):
        previous_status = task.status

        if task.status == 'DONE':
            return

        # If this is the last task:
        if task.task_id == len(self.tasks) - 1:
            # Check if all robots have arrived
            if self.all_robots_at_task(self.robots, task, threshold=0.01):
                task.status = 'DONE'
                self.sim_done = True
                self.makespan = self.timestep
            else:
                task.status = 'PENDING'
            return

        # Else: normal tasks
        if self.all_skills_present(task):
            task.status = 'IN_PROGRESS'
            task.duration -= 1
            if task.duration <= 0:
                task.status = 'DONE'
        else:
            task.status = 'PENDING'

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

    def all_robots_at_task(self, robots, task, threshold=0.01):
        """True if all robots are within 'threshold' distance of 'task' location."""
        for r in robots:
            if np.linalg.norm(r.location - task.location) > threshold:
                return False
        return True  


    def all_skills_present(self, task):
        """
        Returns True if:
        1) The logical OR of all assigned robots' capabilities covers all task requirements.
        2) Every assigned robot is within 1 unit of the task location.
        Otherwise, returns False.
        """
        assigned_robots = [r for r in self.robots if r.current_task == task]

        # Combine capabilities via logical OR across all assigned robots
        combined_capabilities = np.zeros_like(task.requirements, dtype=bool)
        for robot in assigned_robots:
            robot_cap = np.array(robot.capabilities, dtype=bool)
            combined_capabilities = np.logical_or(combined_capabilities, robot_cap)

        required_skills = np.array(task.requirements, dtype=bool)
        # Check if the combined team covers all required skills
        if not np.all(combined_capabilities[required_skills]):
            return False

        # Check if all assigned robots are close enough to the task
        return self.all_robots_at_task(assigned_robots, task)
    

    def step(self):
        """Advance the simulation by one timestep, moving robots and updating tasks."""

        for robot in self.robots:   
            self.update_robot(robot)
        # Update tasks (if in progress, decrement duration, mark done if complete)
        for task in self.tasks:
            self.update_task(task)


        # Check if any robot is idle and there are pending tasks
        idle_robots = [r for r in sim.robots if not r.current_task or r.current_task.status == 'DONE']

        if idle_robots:
            schedule_tasks(sim)

        

        self.timestep += 1


def schedule_tasks(sim):
    """
    Example scheduling logic:
      - Check for any idle robots
      - Assign them tasks if any are PENDING
    This could be replaced by a call to your NN or heuristic.
    """
    instantaneous_schedule = greedy_instantaneous_assignment(sim)
    for robot in sim.robots:
        task_id = instantaneous_schedule.robot_assignments.get(robot.robot_id)
        if task_id is not None:
            robot.current_task = sim.tasks[task_id]

    

def visualize(sim):
    """Interactive Matplotlib figure with tasks as pie charts and step buttons."""
    n_skills = len(sim.tasks[0].requirements)  # Assume all tasks have the same number of skills
    colors = plt.cm.Set1(np.linspace(0, 1, n_skills))  # Generate a color palette
    
    def add_robot_skills_table(fig, robots):
        """Add a table below the plot displaying robots and their skills."""
        # Create table data
        table_data = [["Robot", "Skills", "Current Task"]]
        for i, robot in enumerate(robots):
            skills = ", ".join([str(skill) for skill in robot.capabilities])
            current_task = robot.current_task.task_id if robot.current_task else "None"
            table_data.append([f"Robot {i}", skills, f"Task {current_task}"])
        # Create the table
        ax_table = plt.axes([0.1, 0.05, 0.8, 0.2])  # Position for the table
        ax_table.axis("off")  # Hide the axes
        table = Table(ax_table, bbox=[0, 0, 1, 1])
        cell_colors = [["#d4e6f1" if row % 2 == 0 else "#f2f3f4" for col in range(2)] for row in range(len(table_data))]

        # Add cells to the table
        for row, (robot, skills, task) in enumerate(table_data):
            table.add_cell(row, 0, width=0.4, height=0.15, text=robot, loc="center", facecolor=cell_colors[row][0])
            table.add_cell(row, 1, width=0.6, height=0.15, text=skills, loc="center", facecolor=cell_colors[row][1])
            table.add_cell(row, 2, width=0.6, height=0.15, text=task, loc="center", facecolor=cell_colors[row][1])
        # Add table to the axes
        ax_table.add_table(table)


    def draw_pie(ax, x, y, sizes, radius):
        """Draw a pie chart at the specified (x, y) position."""
        start_angle = 0
        for size, color in zip(sizes, colors):
            end_angle = start_angle + size * 360
            if size > 0:
                wedge = Wedge((x, y), radius, start_angle, end_angle, facecolor=color, edgecolor="black", lw=0.5)
                ax.add_patch(wedge)
            start_angle = end_angle

    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(bottom=0.3)  # Make space for buttons


    def update_plot():
        ax.clear()

        # Plot tasks with pie-chart representation
        for task in sim.tasks:
            if task.status != 'DONE':
                total_skills = np.sum(task.requirements)
                skill_sizes = task.requirements / total_skills if total_skills > 0 else np.zeros_like(task.requirements)
                draw_pie(ax, task.location[0], task.location[1], skill_sizes, task.duration / 50)  # Scale pie size by duration
                ax.text(task.location[0], task.location[1], f"Task {task.task_id}", fontsize=10, ha='center')


        # Plot robots and add their numbers
        for robot_idx, robot in enumerate(sim.robots):
            ax.scatter(robot.location[0], robot.location[1], marker='s', c='black', label='Robots' if robot_idx == 0 else None)
            ax.text(robot.location[0] + 1, robot.location[1] + 1, f"{robot_idx}", fontsize=10, color="black", ha='center')



        # Plot start and end points
        ax.scatter(sim.tasks[0].location[0], sim.tasks[0].location[1], color='green', s=150, marker='x', label="Start (Task 0)")
        plt.text(sim.tasks[0].location[0] + 6, sim.tasks[0].location[1] - 1, "Start", fontsize=15, ha='center')
        ax.scatter(sim.tasks[-1].location[0], sim.tasks[-1].location[1], color='red', s=150, marker='x', label="End (Task -1)")
        plt.text(sim.tasks[-1].location[0] + 6, sim.tasks[-1].location[1] - 1, "End", fontsize=15, ha='center')

        # Add robot skills table
        add_robot_skills_table(fig, sim.robots)

        # Add legend for skills
        legend_patches = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], markersize=10, label=f"Skill {i}")
            for i in range(n_skills)
        ]
        ax.legend(handles=legend_patches, title="Task Skills", loc="upper right")
        
        if sim.sim_done:
            ax.text(
                50, 50, 
                f"Simulation Finished in {sim.makespan} timesteps!", 
                fontsize=15, color='blue', ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8)
            )

        ax.set_title(f"Timestep: {sim.timestep}")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        plt.draw()


    def next_step_callback(event=None):
        sim.step()
        update_plot()


    def advance_10_steps_callback(event=None):
        for _ in range(10):
            sim.step()
        update_plot()

    def key_press(event):
        """Handle key presses to trigger button actions."""
        if event.key == 'n':  # Press 'n' for Next Timestep
            next_step_callback()
        elif event.key == 'm':  # Press 't' for Advance 10 Timesteps
            advance_10_steps_callback()

    # Create buttons
    ax_button_next = plt.axes([0.5, 0.92, 0.2, 0.07])  # Position for 'Next Timestep' button
    ax_button_10 = plt.axes([0.7, 0.92, 0.2, 0.07])    # Position for 'Advance 10 Timesteps' button

    btn_next = Button(ax_button_next, 'Next Timestep')
    btn_next.on_clicked(next_step_callback)

    btn_advance_10 = Button(ax_button_10, '10 Timesteps')
    btn_advance_10.on_clicked(advance_10_steps_callback)
    fig.canvas.mpl_connect('key_press_event', key_press)

    # Initial draw
    update_plot()
    plt.show()



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--visualize", action='store_true', help="Visualize the simulation")
    args = parser.parse_args()

    # Example usage
    n_tasks = 7 
    n_robots = 3
    n_skills = 3
    np.random.seed(20)

    precedence_constraints = [[1,2], [2,3]]
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
