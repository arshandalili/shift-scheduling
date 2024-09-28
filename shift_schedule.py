import argparse
import numpy as np
from ortools.sat.python import cp_model

class ShiftSolution(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions."""

    def __init__(
        self,
        variables: list[cp_model.IntVar],
        num_days: int,
        num_doctors: int,
        day_to_shift_type: list[int],
        shift_type_names: list[str],
    ):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__variables = variables
        self.__solution_count = 0
        self.__num_days = num_days
        self.__num_doctors = num_doctors
        self.__day_to_shift_type = day_to_shift_type
        self.__shift_type_names = shift_type_names

    def on_solution_callback(self) -> None:
        self.__solution_count += 1
        print(f"Solution {self.__solution_count}:")
        for d in range(self.__num_days):
            print(
                f"Day {d + 1} ({self.__shift_type_names[self.__day_to_shift_type[d]]} shift):"
            )
            assigned_doctors = []
            for doc in range(self.__num_doctors):
                if self.value(self.__variables[(d, doc)]):
                    assigned_doctors.append(f"Doc {doc}")
            print("  Assigned doctors: ", ", ".join(assigned_doctors))

    @property
    def solution_count(self) -> int:
        return self.__solution_count


def main(args):
    
    np.random.seed(args.seed)

    doctor_pairs = [(i, i + 1) for i in range(0, args.num_doctors, 2)]

    regular, pre_holiday, holiday = 0, 1, 2
    shift_type_names = ["Regular", "Pre-holiday", "Holiday"]

    day_to_shift_type = []
    for day in range(args.num_days):
        if day % 7 in [0, 6]:
            day_to_shift_type.append(holiday)
        elif day % 7 == 5:
            day_to_shift_type.append(pre_holiday)
        else:
            day_to_shift_type.append(regular)

    num_shift_days = {
        regular: day_to_shift_type.count(regular),
        pre_holiday: day_to_shift_type.count(pre_holiday),
        holiday: day_to_shift_type.count(holiday),
    }

    fair_shifts_per_type = {
        shift: (num_shift_days[shift] * args.doctors_per_shift) // args.num_doctors
        for shift in num_shift_days
    }
    
    print("Fair shifts per type per doctor:", fair_shifts_per_type)

    shift_requests = []
    for _ in range(args.num_doctors):
        requests = np.random.choice([0, 1], args.num_days)
        days_off_indices = np.random.choice(np.arange(args.num_days), args.days_off, replace=False).astype(int)
        for idx in days_off_indices:
            requests[idx] = -1
        shift_requests.append(requests)

    model = cp_model.CpModel()

    shifts = {}
    for d in range(args.num_days):
        for doc in range(args.num_doctors):
            shifts[(d, doc)] = model.NewBoolVar(f"shift_d{d}_doc{doc}")

    for d in range(args.num_days):
        model.Add(
            sum(shifts[(d, doc)] for doc in range(args.num_doctors)) == args.doctors_per_shift
        )

    slack_fair_vars = {}
    for doc in range(args.num_doctors):
        for shift_type in [regular, pre_holiday, holiday]:
            assigned_shifts = sum(
                shifts[(d, doc)]
                for d in range(args.num_days)
                if day_to_shift_type[d] == shift_type
            )
            slack_fair = model.NewIntVar(
                0, num_shift_days[shift_type], f"slack_fair_doc{doc}_type{shift_type}"
            )
            model.Add(assigned_shifts + slack_fair >= fair_shifts_per_type[shift_type])
            slack_fair_vars[(doc, shift_type)] = slack_fair

    for doc in range(args.num_doctors):
        for d in range(args.num_days - 1):
            model.Add(shifts[(d, doc)] + shifts[(d + 1, doc)] <= 1)

    paired_shift_penalty_vars = []
    for doc1, doc2 in doctor_pairs:
        for d in range(args.num_days):
            same_shift = model.NewBoolVar(f"paired_shift_doc{doc1}_doc{doc2}_day{d}")
            model.Add(shifts[(d, doc1)] == shifts[(d, doc2)]).OnlyEnforceIf(same_shift)
            penalty = model.NewBoolVar(
                f"penalty_paired_shift_doc{doc1}_doc{doc2}_day{d}"
            )
            model.Add(penalty == 1 - same_shift)
            paired_shift_penalty_vars.append(penalty)

    preference_penalty_vars = []
    for d in range(args.num_days):
        for doc in range(args.num_doctors):
            if shift_requests[doc][d] == 1:
                penalty = model.NewBoolVar(f"penalty_doc{doc}_day{d}_work")
                model.Add(penalty == 1 - shifts[(d, doc)])
                preference_penalty_vars.append(penalty)
            elif shift_requests[doc][d] == -1:
                penalty = model.NewBoolVar(f"penalty_doc{doc}_day{d}_avoid")
                model.Add(penalty == shifts[(d, doc)])
                preference_penalty_vars.append(penalty)

    model.Minimize(
        sum(slack_fair_vars.values())
        + sum(paired_shift_penalty_vars)
        + sum(preference_penalty_vars)
    )

    solver = cp_model.CpSolver()
    solver.parameters.random_seed = args.seed
    solver.parameters.num_search_workers = 16
    solver.parameters.max_time_in_seconds = 120
    solver.parameters.log_search_progress = True
    solution_printer = ShiftSolution(
        variables=shifts,
        num_days=args.num_days,
        num_doctors=args.num_doctors,
        day_to_shift_type=day_to_shift_type,
        shift_type_names=shift_type_names,
    )
    status = solver.Solve(model, solution_printer)
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print("Solution found:")
        for d in range(args.num_days):
            print(f"Day {d + 1} ({shift_type_names[day_to_shift_type[d]]} shift):")
            assigned_doctors = []
            for doc in range(args.num_doctors):
                if solver.Value(shifts[(d, doc)]):
                    assigned_doctors.append(f"Doc {doc}")
            print("  Assigned doctors: ", ", ".join(assigned_doctors))
        print("\nConstraint Violation Details:")
        print(
            "  Fair distribution slack:",
            {key: solver.Value(val) for key, val in slack_fair_vars.items()},
        )
        for doc in range(args.num_doctors):
            args.days_off = [d for d in range(args.num_days) if solver.Value(shifts[(d, doc)]) == 0]
            print(f"  Doc {doc} reqested to have days off on days: {[d for d in np.arange(args.num_days) if shift_requests[doc][d] == -1]} and status of shifts on days off: {[solver.Value(shifts[(d, doc)]) for d in np.arange(args.num_days) if shift_requests[doc][d] == -1]}")
        with open(args.output, "w") as f:
            for d in range(args.num_days):
                f.write(f"Day {d + 1} ({shift_type_names[day_to_shift_type[d]]} shift):\n")
                assigned_doctors = []
                for doc in range(args.num_doctors):
                    if solver.Value(shifts[(d, doc)]):
                        assigned_doctors.append(f"Doc {doc}")
                f.write("  Assigned doctors: " + ", ".join(assigned_doctors) + "\n")
    else:
        print("No solution found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_doctors", type=int, default=10)
    parser.add_argument("--num_days", type=int, default=30)
    parser.add_argument("--doctors_per_shift", type=int, default=3)
    parser.add_argument("--days_off", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="solution.txt")
    parser.add_argument("--max_time_in_seconds", type=int, default=120)
    parser.add_argument("--num_search_workers", type=int, default=16)
    args = parser.parse_args()
    
    main(args)
