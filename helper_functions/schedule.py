import json
class Schedule:
    """
    Represents a solution to the scheduling problem.

    The schedule does not include the start and end tasks, which are assumed to be present in the problem instance.
    These have execution time , but the travel time from start to task_1 and task_n to end are included.

    The robot schedules are given as a dictionary, where the keys are robot indices and the values 
    are lists of the format [(task, start_time, end_time), ...] for each task assigned to the robot.
    """
    def __init__(self, makespan, robot_schedules, n_tasks):
        self.makespan: float = makespan
        self.robot_schedules: dict = robot_schedules
        self.n_tasks: int = n_tasks
        self.n_robots: int = len(robot_schedules)

        # Sort tasks for each robot by starting time
        self.robot_schedules = {
            robot: sorted(tasks, key=lambda x: x[1])
            for robot, tasks in robot_schedules.items()
        }
    def __str__(self):
        """
        String representation for better readability.
        """
        result = f"Schedule:\n"
        result += f"  Number of tasks: {self.n_tasks}\n"
        result += f"  Number of robots: {self.n_robots}\n"
        result += f"  Makespan/Arrival at end location: {self.makespan:.2f}\n"
        result += "  Robot Tasks:\n"
        for robot, tasks in self.robot_schedules.items():
            result += f"    Robot {robot}:\n"
            for task, start, end in tasks:
                result += f"      Task {task}: {start:.2f} -> {end:.2f}\n"
        return result
    

    def to_dict(self):
        """
        Converts the Schedule object into a dictionary for JSON serialization.
        """
        return {
            "makespan": self.makespan,
            "n_tasks": self.n_tasks,
            "n_robots": self.n_robots,
            "robot_schedules": {
                robot: [
                    {"task": task, "start_time": start, "end_time": end}
                    for task, start, end in tasks
                ]
                for robot, tasks in self.robot_schedules.items()
            },

        }

    @classmethod
    def from_dict(cls, data):
        """
        Creates a Schedule object from a dictionary (useful for JSON deserialization).
        """
        robot_schedules = {
            int(robot): [(t["task"], t["start_time"], t["end_time"]) for t in tasks]
            for robot, tasks in data["robot_schedules"].items()
        }
        return cls(
            makespan=data["makespan"],
            robot_schedules=robot_schedules,
            n_tasks=data["n_tasks"],
        )