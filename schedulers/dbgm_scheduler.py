import torch
from .bigraph_matching import solve_bipartite_matching
from method_explorations.LVWS.transformer_models import TransformerScheduler
from helper_functions.schedules import Instantaneous_Schedule



class DBGMScheduler:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trained_model = TransformerScheduler(robot_input_dimensions=3, task_input_dimension=5, embed_dim=64, ff_dim=128, num_layers=2).to(self.device)
        self.trained_model.load_state_dict(torch.load("/home/jakob/thesis/method_explorations/LVWS/checkpoints/checkpoint_epoch_10.pt")
                                    )
    def assign_tasks_to_robots(self, sim):
        n_robots = len(sim.robots)
        robot_assignments = {}
        # Special case for the last task
        idle_robots = [r for r in sim.robots if r.current_task is None or r.current_task.status == 'DONE']
        pending_tasks = [t for t in sim.tasks if t.status == 'PENDING']
        # Check if all tasks are done -> send all robots to the exit task
        if len(pending_tasks) == 1:
            for robot in idle_robots:
                robot_assignments[robot.robot_id] = pending_tasks[0].task_id
                robot.current_task = pending_tasks[0]
            return Instantaneous_Schedule(robot_assignments)

        task_features = [task.feature_vector() for task in sim.tasks[1:-1]]
        task_features = torch.tensor(task_features, dtype=torch.float32).unsqueeze(0).to(self.device)

        robot_features = [robot.feature_vector() for robot in sim.robots]
        robot_features = torch.tensor(robot_features, dtype= torch.float32).unsqueeze(0).to(self.device)


        predicted_reward = self.trained_model(robot_features, task_features).squeeze()

        # Add  negative rewards for for the start and end task --> not to be selected, will be handled by the scheduler
        reward_start_end = torch.ones(n_robots, 1).to(self.device) * (-1000)
        predicted_reward = torch.cat((reward_start_end, predicted_reward, reward_start_end), dim=1)


        for robot_idx, robot in enumerate(sim.robots):
            for task_idx, task in enumerate(sim.tasks[1:-1]):
                print(f"Robot {robot_idx} -> Task {task_idx}: {predicted_reward[robot_idx][task_idx]:.2f}")

        bipartite_matching_solution = solve_bipartite_matching(predicted_reward, sim)

        robot_assignments = {robot: task for (robot, task), val in bipartite_matching_solution.items() if val == 1}

        return Instantaneous_Schedule(robot_assignments)