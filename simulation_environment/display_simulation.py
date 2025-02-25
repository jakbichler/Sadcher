import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.table import Table
from matplotlib.patches import Wedge
from matplotlib.widgets import Button
import imageio


def visualize(sim):
    """Interactive Matplotlib figure with tasks as pie charts and step buttons."""
    n_skills = len(sim.tasks[0].requirements)  # Assume all tasks have the same number of skills
    colors = plt.cm.Set1(np.linspace(0, 1, n_skills))  # Generate a color palette
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(bottom=0.3)  # Make space for buttons

    # Create buttons
    ax_button_next = plt.axes([0.3, 0.92, 0.2, 0.07])   # 'Next Timestep' button
    ax_button_10   = plt.axes([0.5, 0.92, 0.2, 0.07])   # 'Advance 10 Timesteps' button


    btn_next = Button(ax_button_next, 'Next Timestep')
    btn_next.on_clicked(lambda event: next_step_callback(sim, ax, fig, colors, n_skills))

    btn_advance_10 = Button(ax_button_10, '10 Timesteps')
    btn_advance_10.on_clicked(lambda event: advance_10_steps_callback(sim, ax, fig, colors, n_skills))


    fig.canvas.mpl_connect(
        'key_press_event',
        lambda event: key_press(event, sim, ax, fig, colors, n_skills)
    )
    update_plot(sim, ax, fig, colors, n_skills)
    plt.show()


def add_robot_skills_table(fig, robots):
    """Add a table below the plot displaying robots and their skills."""
    table_data = [["Robot", "Skills", "Current Task"]]
    for i, robot in enumerate(robots):
        skills = ", ".join([str(skill) for skill in robot.capabilities])
        current_task = robot.current_task.task_id if robot.current_task else "None"
        table_data.append([f"Robot {i}", skills, f"Task {current_task}"])

    ax_table = plt.axes([0.1, 0.05, 0.8, 0.2])  # Position for the table
    ax_table.axis("off")  # Hide the axes
    table = Table(ax_table, bbox=[0, 0, 1, 1])
    cell_colors = [["#d4e6f1" if row % 2 == 0 else "#f2f3f4" for col in range(3)] for row in range(len(table_data))]

    for row, (robot, skills, task) in enumerate(table_data):
        table.add_cell(row, 0, width=0.4, height=0.15, text=robot, loc="center", facecolor=cell_colors[row][0])
        table.add_cell(row, 1, width=0.6, height=0.15, text=skills, loc="center", facecolor=cell_colors[row][1])
        table.add_cell(row, 2, width=0.6, height=0.15, text=task, loc="center", facecolor=cell_colors[row][1])

    ax_table.add_table(table)


def add_precedence_constraints_text(fig, precedence_constraints):
    """Display precedence constraints as a single line of text below the robot table."""
    ax_text = plt.axes([0.1, 0.0, 0.8, 0.05])  # Position below the robot table
    ax_text.axis("off")  # Hide the axes

    precedence_text = f"Precedence Constraints: {precedence_constraints}"
    ax_text.text(0.5, 0.5, precedence_text, ha='center', va='center', fontsize=10, wrap=True)


def draw_pie(ax, x, y, sizes, radius, colors):
    """Draw a pie chart at the specified (x, y) position."""
    start_angle = 0
    for size, color in zip(sizes, colors):
        end_angle = start_angle + size * 360
        if size > 0:
            wedge = Wedge((x, y), radius, start_angle, end_angle, facecolor=color, edgecolor="black", lw=0.5)
            ax.add_patch(wedge)
        start_angle = end_angle


def update_plot(sim, ax, fig, colors, n_skills):
    ax.clear()

    for task in sim.tasks:
        if task.status != 'DONE':
            total_skills = np.sum(task.requirements)
            skill_sizes = task.requirements / total_skills if total_skills > 0 else np.zeros_like(task.requirements)
            draw_pie(ax, task.location[0], task.location[1], skill_sizes, task.remaining_duration / 50, colors)
            ax.text(task.location[0], task.location[1], f"Task {task.task_id}", fontsize=10, ha='center')

    for robot_idx, robot in enumerate(sim.robots):
        ax.scatter(robot.location[0], robot.location[1], marker='s', c='black', label='Robots' if robot_idx == 0 else None)
        ax.text(robot.location[0] + 1, robot.location[1] + 1, f"{robot_idx}", fontsize=10, color="black", ha='center')

    ax.scatter(sim.tasks[0].location[0], sim.tasks[0].location[1], color='green', s=150, marker='x', label="Start (Task 0)")
    ax.text(sim.tasks[0].location[0] + 6, sim.tasks[0].location[1] - 1, "Start", fontsize=15, ha='center')
    ax.scatter(sim.tasks[-1].location[0], sim.tasks[-1].location[1], color='red', s=150, marker='x', label="End (Task -1)")
    ax.text(sim.tasks[-1].location[0] + 6, sim.tasks[-1].location[1] - 1, "End", fontsize=15, ha='center')

    add_robot_skills_table(fig, sim.robots)
    add_precedence_constraints_text(fig, sim.precedence_constraints)

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


def next_step_callback(sim, ax, fig, colors, n_skills):
    sim.step()
    update_plot(sim, ax, fig, colors, n_skills)


def advance_10_steps_callback(sim, ax, fig, colors, n_skills):
    for _ in range(10):
        sim.step()
    update_plot(sim, ax, fig, colors, n_skills)


def key_press(event, sim, ax, fig, colors, n_skills):
    if event.key == 'n':  # Press 'n' for Next Timestep
        next_step_callback(sim, ax, fig, colors, n_skills)
    elif event.key == 'm':  # Press 'm' for Advance 10 Timesteps
        advance_10_steps_callback(sim, ax, fig, colors, n_skills)


def make_video_from_frames(frame_dir, output="simulation.mp4", fps=5):
    images = []
    for fname in sorted(os.listdir(frame_dir)):
        if fname.endswith(".png"):
            images.append(imageio.imread(os.path.join(frame_dir, fname)))
    imageio.mimsave(output, images, fps=fps)
    # Delete the frames directory
    shutil.rmtree(frame_dir)


def run_video_mode(sim):
    colors = plt.cm.Set1(np.linspace(0, 1, len(sim.tasks[0].requirements)))
    n_skills = len(sim.tasks[0].requirements)
    frame_dir = f"frames_{sim.scheduler_name}"
    os.makedirs(frame_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))
    while not sim.sim_done:
        for _ in range(5):
            sim.step()
        update_plot(sim, ax, fig, colors, n_skills)
        plt.show(block=False)
        fig.canvas.draw()
        fig.savefig(os.path.join(frame_dir, f"frame_{sim.timestep:04d}.png"))

    # Make final video
    make_video_from_frames(frame_dir, output=f"{sim.scheduler_name}.mp4", fps=2)
    print("Video saved as simulation.mp4")