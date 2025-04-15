import os
import shutil

import imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import (
    Rectangle,  # add at the top with your other import
    Wedge,
)
from matplotlib.table import Table
from matplotlib.widgets import Button

from schedulers.sadcher import SadcherScheduler


def visualize(sim, scheduler):
    """Interactive Matplotlib figure with tasks as pie charts and step buttons."""

    n_skills = len(sim.tasks[0].requirements)  # Assume all tasks have the same number of skills
    colors = plt.cm.Set1(np.linspace(0, 1, n_skills))  # Generate a color palette
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(bottom=0.3)  # Make space for buttons

    # Create buttons
    ax_button_next = plt.axes([0.3, 0.92, 0.2, 0.07])  # 'Next Timestep' button
    ax_button_10 = plt.axes([0.5, 0.92, 0.2, 0.07])  # 'Advance 10 Timesteps' button

    btn_next = Button(ax_button_next, "Next Timestep")
    btn_next.on_clicked(lambda event: next_step_callback(sim, ax, fig, colors, n_skills, scheduler))

    btn_advance_10 = Button(ax_button_10, "10 Timesteps")
    btn_advance_10.on_clicked(
        lambda event: advance_10_steps_callback(sim, ax, fig, colors, n_skills, scheduler)
    )

    fig.canvas.mpl_connect(
        "key_press_event", lambda event: key_press(event, sim, ax, fig, colors, n_skills, scheduler)
    )
    update_plot(sim, ax, fig, colors, n_skills)
    plt.show()


def add_robot_skills_table(fig, robots, colors, n_skills):
    """Add a table below the plot displaying robots and their skills as colored squares."""
    # Create header: one column for Robot, one for each skill, then one for Current Task.
    header = ["Robot"] + [f"Skill {i}" for i in range(n_skills)] + ["Current Task"]
    table_data = [header]
    for i, robot in enumerate(robots):
        row = [f"Robot {i}"]
        for j in range(n_skills):
            row.append(robot.capabilities[j])
        current_task = robot.current_task.task_id if robot.current_task else "None"
        row.append(f"Task {current_task}")
        table_data.append(row)

    ax_table = plt.axes([0.1, -0.1, 0.8, 0.2])
    ax_table.axis("off")
    table = Table(ax_table, bbox=[0, 0, 1, 1])

    # Set column widths:
    robot_col_width = 0.2
    current_task_width = 0.2
    skill_col_width = 0.15 / n_skills  # remaining width evenly divided

    n_rows = len(table_data)
    n_cols = n_skills + 2  # Robot and Current Task plus one per skill

    for row in range(n_rows):
        for col in range(n_cols):
            # Header row uses a fixed color.
            if row == 0:
                facecolor = "#d4e6f1"
            else:
                # For the Robot and Current Task columns, use alternating row colors.
                if col == 0 or col == n_cols - 1:
                    facecolor = "white"
                else:
                    # For skill columns, if the robot has the skill, use the corresponding color.
                    skill_index = col - 1
                    facecolor = colors[skill_index] if table_data[row][col] else "white"

            # Determine the cell width.
            if col == 0:
                cell_width = robot_col_width
            elif col == n_cols - 1:
                cell_width = current_task_width
            else:
                cell_width = skill_col_width

            # For skill cells (non-header, non-robot/task), we leave text empty.
            cell_text = (
                str(table_data[row][col]) if (row == 0 or col == 0 or col == n_cols - 1) else ""
            )
            table.add_cell(
                row,
                col,
                width=cell_width,
                height=0.15,
                text=cell_text,
                loc="center",
                facecolor=facecolor,
            )
    ax_table.add_table(table)


def add_precedence_constraints_text(fig, precedence_constraints):
    """Display precedence constraints as a single line of text below the robot table."""
    ax_text = plt.axes([0.1, 0.0, 0.8, 0.05])  # Position below the robot table
    ax_text.axis("off")  # Hide the axes

    precedence_text = f"Precedence Constraints: {precedence_constraints}"
    ax_text.text(0.5, 0.5, precedence_text, ha="center", va="center", fontsize=10, wrap=True)


def add_current_task_text(fig, robots):
    # Create or reuse a dedicated axes for current task text
    if not hasattr(fig, "_current_task_ax"):
        fig._current_task_ax = fig.add_axes([0.5, 0.0, 0.8, 0.05])
        fig._current_task_ax.axis("off")
    else:
        fig._current_task_ax.cla()  # Clear previous text
        fig._current_task_ax.axis("off")

    current_tasks = [
        f"Robot {robot.robot_id}: Task {robot.current_task.task_id}"
        for robot in robots
        if robot.current_task
    ]
    current_tasks_text = "\n".join(current_tasks)
    fig._current_task_ax.text(
        0.5, 0.5, current_tasks_text, ha="center", va="center", fontsize=10, wrap=True
    )


def draw_pie(ax, x, y, sizes, radius, colors):
    """Draw a pie chart at the specified (x, y) position."""
    start_angle = 0
    for size, color in zip(sizes, colors):
        end_angle = start_angle + size * 360
        if size > 0:
            wedge = Wedge(
                (x, y), radius, start_angle, end_angle, facecolor=color, edgecolor="black", lw=0.5
            )
            ax.add_patch(wedge)
        start_angle = end_angle


def draw_robot_skills_squares(ax, robot, colors, square_size=2.5):
    """
    Draws a fixed number of vertically stacked squares for each robot.
    Each square corresponds to a possible skill. If the skill is present (truthy),
    the square is filled with the corresponding color; if not, it is transparent.
    """

    bottom = robot.location[1] + 2
    left = robot.location[0] - square_size / 2
    for skill_index, skill in enumerate(robot.capabilities):
        facecolor = colors[int(skill_index)] if skill else "none"
        rect = Rectangle(
            (left, bottom + skill_index * (square_size)),
            square_size,
            square_size,
            facecolor=facecolor,
            edgecolor="black",
            lw=1.0,
        )
        ax.add_patch(rect)
    ax.scatter(robot.location[0], robot.location[1], color="black", s=50)
    ax.text(
        robot.location[0] - 2.0,
        robot.location[1],
        f"{robot.robot_id}",
        fontsize=10,
        color="black",
        ha="center",
        va="center",
    )


def update_plot(sim, ax, fig, colors, n_skills, video_mode=False):
    ax.clear()

    for task in sim.tasks:
        if task.status != "DONE":
            total_skills = np.sum(task.requirements)
            skill_sizes = (
                task.requirements / total_skills
                if total_skills > 0
                else np.zeros_like(task.requirements)
            )
            draw_pie(
                ax,
                task.location[0],
                task.location[1],
                skill_sizes,
                task.remaining_duration / 30,
                colors,
            )
            ax.text(
                task.location[0], task.location[1], f"Task {task.task_id}", fontsize=10, ha="center"
            )

    for robot in sim.robots:
        draw_robot_skills_squares(ax, robot, colors)

    ax.scatter(
        sim.tasks[0].location[0],
        sim.tasks[0].location[1],
        color="green",
        s=150,
        marker="x",
        label="Start (Task 0)",
    )
    ax.text(
        sim.tasks[0].location[0] + 6,
        sim.tasks[0].location[1] - 1,
        "Start",
        fontsize=15,
        ha="center",
    )
    ax.scatter(
        sim.tasks[-1].location[0],
        sim.tasks[-1].location[1],
        color="red",
        s=150,
        marker="x",
        label="End (Task -1)",
    )
    ax.text(
        sim.tasks[-1].location[0] + 6,
        sim.tasks[-1].location[1] - 1,
        "End",
        fontsize=15,
        ha="center",
    )

    if video_mode:
        add_current_task_text(fig, sim.robots)
    else:
        add_robot_skills_table(fig, sim.robots, colors, n_skills)
    add_precedence_constraints_text(fig, sim.precedence_constraints)

    legend_patches = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=colors[i],
            markersize=10,
            label=f"Skill {i}",
        )
        for i in range(n_skills)
    ]
    ax.legend(handles=legend_patches, title="Task Skills", loc="upper right")

    if sim.sim_done:
        ax.text(
            50,
            50,
            f"Simulation Finished in {sim.makespan} timesteps!",
            fontsize=15,
            color="blue",
            ha="center",
            va="center",
            bbox=dict(facecolor="white", alpha=0.8),
        )
    ax.set_title(f"Timestep: {sim.timestep}")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    plt.draw()


def next_step_callback(sim, ax, fig, colors, n_skills, scheduler):
    sim.step()
    if isinstance(scheduler, SadcherScheduler):
        predicted_reward, instantaneous_schedule = scheduler.calculate_robot_assignment(sim)
        sim.find_highest_non_idle_reward(predicted_reward)
    else:
        instantaneous_schedule = scheduler.calculate_robot_assignment(sim)
    sim.assign_tasks_to_robots(instantaneous_schedule)
    update_plot(sim, ax, fig, colors, n_skills)


def advance_10_steps_callback(sim, ax, fig, colors, n_skills, scheduler):
    for _ in range(10):
        sim.step()
    if isinstance(scheduler, SadcherScheduler):
        predicted_reward, instantaneous_schedule = scheduler.calculate_robot_assignment(sim)
        sim.find_highest_non_idle_reward(predicted_reward)
    else:
        instantaneous_schedule = scheduler.calculate_robot_assignment(sim)
    sim.assign_tasks_to_robots(instantaneous_schedule)
    update_plot(sim, ax, fig, colors, n_skills)


def key_press(event, sim, ax, fig, colors, n_skills, scheduler):
    if event.key == "n":  # Press 'n' for Next Timestep
        next_step_callback(sim, ax, fig, colors, n_skills, scheduler)
    elif event.key == "m":  # Press 'm' for Advance 10 Timesteps
        advance_10_steps_callback(sim, ax, fig, colors, n_skills, scheduler)


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
        sim.step()
        update_plot(sim, ax, fig, colors, n_skills, video_mode=True)
        plt.show(block=False)
        fig.canvas.draw()
        fig.savefig(os.path.join(frame_dir, f"frame_{sim.timestep:04d}.png"))

    # Make final video
    make_video_from_frames(frame_dir, output=f"{sim.scheduler_name}.mp4", fps=2)
    print("Video saved as simulation.mp4")
