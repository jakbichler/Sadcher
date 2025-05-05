import pulp
import torch


class CachedBipartiteMatcher:
    def __init__(self, sim):
        """
        Build and cache the static part of the MIP:
         – variables  A[i][j], X[j]
         – linking & capability constraints for every task
        (availability & readiness go into solve())
        """
        self.sim = sim
        self.n_robots = len(sim.robots)
        self.n_tasks = len(sim.tasks)
        self.n_skills = len(sim.robots[0].capabilities)
        self.M = self.n_robots

        # 1) create vars
        self.A = pulp.LpVariable.dicts(
            "A",
            (range(self.n_robots), range(self.n_tasks)),
            lowBound=0,
            upBound=1,
            cat=pulp.LpBinary,
        )
        self.X = pulp.LpVariable.dicts(
            "X", range(self.n_tasks), lowBound=0, upBound=1, cat=pulp.LpBinary
        )

        # 2) build template problem (no objective, no availability/readiness)
        self.template = pulp.LpProblem("BipartiteMatching", pulp.LpMaximize)

        # --- static linking & requirements for every task ---
        for j, task in enumerate(sim.tasks):
            # big-M linking
            self.template += (
                pulp.lpSum(self.A[i][j] for i in range(self.n_robots)) <= self.M * self.X[j]
            )
            self.template += pulp.lpSum(self.A[i][j] for i in range(self.n_robots)) >= self.X[j]
            # capability requirements
            for cap in range(self.n_skills):
                req = task.requirements[cap]
                if req:
                    self.template += (
                        pulp.lpSum(
                            sim.robots[i].capabilities[cap] * self.A[i][j]
                            for i in range(self.n_robots)
                        )
                        >= req * self.X[j]
                    )

    def solve(self, R: torch.Tensor, n_threads: int = 6, gap: float = 0.02):
        """
        Copy template, add dynamic constraints & objective, solve.
        R: torch.Tensor [n_robots, n_tasks]
        """
        problem = self.template.copy()

        # --- dynamic: availability constraints ---
        for i, robot in enumerate(self.sim.robots):
            if robot.available:
                problem += pulp.lpSum(self.A[i][j] for j in range(self.n_tasks)) <= 1
            else:
                for j in range(self.n_tasks):
                    problem += self.A[i][j] == 0

        # --- dynamic: readiness/incomplete tasks off ---
        for j, task in enumerate(self.sim.tasks):
            if not (task.ready and task.incomplete):
                problem += self.X[j] == 0

        problem.objective = pulp.lpSum(
            R[i, j].item() * self.A[i][j] for i in range(self.n_robots) for j in range(self.n_tasks)
        )

        problem.solve(
            pulp.PULP_CBC_CMD(
                msg=False,
                threads=n_threads,
                options=[f"ratioGap {gap}", f"allowableGap {gap}"],
            )
        )

        return {
            (i, j): int(pulp.value(self.A[i][j]))
            for i in range(self.n_robots)
            for j in range(self.n_tasks)
        }
