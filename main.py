import time

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from source.llm_agent import LLMAgent
from source.falsification_problem import (
    PedestrianFalsificationProblem,
    LeadVehicleFalsificationProblem,
    StaticObstacleFalsificationProblem,
)

def run_falsification(label: str, problem_cls):
    print(f"\n==============================")
    print(f"  Scenario: {label}")
    print(f"==============================")

    llm_agent = LLMAgent(model="mistral")
    problem = problem_cls(llm_agent)

    algorithm = NSGA2(pop_size=6)   
    termination = get_termination("n_gen", 8) 

    start = time.time()
    result = minimize(
        problem,
        algorithm,
        termination,
        seed=1,
        verbose=True,
    )
    end = time.time()
    mins = (end - start) / 60.0

    print(f"\n[Scenario {label}] Search complete in {mins:.2f} minutes")

    if result.X is not None:
        print(f"Candidate inputs with strongest violation (most negative objective):")
        for i, (x, f) in enumerate(zip(result.X, result.F)):
            print(f"  #{i+1} x = {x}, objective g = {f[0]:.6f}")
    else:
        print("No solutions found (NSGA-II returned empty result).")


if __name__ == "__main__":
    print("--- Starting ADAS LLM-based falsification experiments ---")

    run_falsification("Pedestrian Crossing", PedestrianFalsificationProblem)
    run_falsification("Lead Vehicle Braking", LeadVehicleFalsificationProblem)
    run_falsification("Static Obstacle", StaticObstacleFalsificationProblem)

    print("\nAll scenarios completed.")
