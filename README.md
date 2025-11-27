ADAS LLM-Based Simulator & Falsification Framework

This project is my implementation of an ADAS simulator where the controller is a Language Model (LLM).
The design and testing approach are inspired by the concepts and methodology presented in the paper:

Parametric Falsification of Many Probabilistic Requirements Under Flakiness (ICSE 2025)

and also informed by the structure of the replication package provided with the assignment.

My goal was to create a clean, understandable, and reproducible implementation of:

-An ADAS simulator backed by an LLM

-A falsification-testing loop

-At least three ADAS scenarios

-A Bayesian estimation component

-Evolutionary search for violation discovery

This README explains how the system is structured, why I made certain decisions, and what I would improve with more time.

1. Project Structure
ADAS-LLM-FALSIFICATION/
│
├── main.py
├── README.md
├── requirements.txt
├── test_llm.py
│
└── source/
    ├── adas_simulator.py
    ├── bayesian_stats.py
    ├── falsification_problem.py
    ├── llm_agent.py
    └──__pychache__/

-adas_simulator.py file

Contains all three ADAS scenarios

Each scenario defines its physics, prompt generation, and stop conditions

Includes helper functions for running single simulation traces

-llm_agent.py file

Provides an interface to an LLM

Supports real LLM mode (Ollama + Mistral)

Supports fake LLM mode (default) for fast, reproducible experiments

-bayesian_stats.py file

Bayesian estimator for crash-probability credible intervals

Uses a Beta/Dirichlet approach similar to the methodology in the paper

-falsification_problem.py file

Defines parametric falsification problems for each ADAS scenario

Uses NSGA-II (pymoo) to optimize crash probability lower bounds

-main.py

Runs falsification for all three scenarios

Prints the most violating input parameters found

2. ADAS Scenarios Implemented

I implemented three separate ADAS scenarios, modeled after common AEB test situations.

1. Pedestrian Crossing

Ego vehicle drives toward a pedestrian

LLM must decide when/how strongly to brake

Possible outcomes: CRASH, SAFE_STOP, TIMEOUT

2. Lead Vehicle Sudden Braking

Ego follows a lead car that brakes between t=1 and t=2 seconds

Controller must react fast enough to avoid collision

Tracks relative distance and velocity

3. Static Obstacle

Ego vehicle drives toward a stationary object

Requires timely emergency braking

Similar to Euro-NCAP AEB stationary target tests

All three scenarios follow the same interface but differ in difficulty and dynamics.

3. LLM Backend

The LLM receives a compact prompt describing:

ego speed

headway / obstacle distance

the scenario

It must output one action:

-ACCELERATE  
-MAINTAIN_SPEED  
-BRAKE_LIGHT  
-EMERGENCY_BRAKE

Fake LLM Mode (default)

In llm_agent.py:

USE_FAKE_LLM = True


This makes the system:

-deterministic

-fast

-reproducible

-not dependent on Ollama or LLM speed (since I encountered problems with Ollama mistral)

-Real LLM mode remains fully supported but is optional.

4. Falsification Method

The falsification process follows several steps:

1. Parameter Space

Each scenario has two parameters:

-initial ego speed

-initial distance / headway

Bounds are chosen to reflect realistic ADAS test scenarios.

2. Monte-Carlo Sampling

For each parameter pair, the simulator runs:

N_TRACES = 10


This captures variability in LLM decisions.

3. Bayesian Estimation

Using a Beta prior, I estimate the credible interval for:

P(crash | observed outcomes)


The lower bound of this interval is used as the safety-relevant quantity.

4. Objective Function

Requirement:

Crash probability ≤ 0.001


Violation measure:

g = allowed_max - crash_probability_lower_bound


Negative g means a violation.

5. NSGA-II Search

Using pymoo, I search for parameter values that maximize violation likelihood.
This mirrors the search-based falsification philosophy from the paper but adapted to simulation instead of PRISM.

5. How to Run
1. Set up the environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

2. Test LLM communication
python test_llm.py

3. Run the full falsification experiments
python main.py


This runs all three scenarios and prints:

violating parameter values

their objective values

runtime per scenario

In fake LLM mode, the entire run takes only a few seconds.

6. Decisions, Assumptions, and Limitations
Why I structured it this way

I wanted something conceptually close to the replication package’s workflow
(scenario -> trace -> Bayesian analysis -> optimization).

The physics are simple so the focus stays on LLM behavior and falsification.

Discrete actions make interpretability easier.

Fake LLM mode ensures reviewers can run everything instantly.

Limitations

-1D motion only (no lateral dynamics).

-No sensor noise or perception errors.

-Fake LLM mode is not stochastic.

-Real LLM mode would allow flakiness but is slower.

-Prompts are intentionally short and not optimized for LLM performance.

-Optimization search budget kept small due to runtime constraints.

What I would improve with more time & (knowledge :D)

-Add more realistic braking models and tire physics

-Include sensor noise and latency

-Improve prompts for better LLM control

-Incorporate richer LLM context (memory across time steps)

-Use larger stochastic LLMs to analyze true flakiness as in the paper

-Create plots/heatmaps of crash probability over the parameter space

-Expand scenarios (curves, cut-ins, multi-agent interactions)

7. Requirements
numpy
scipy
pymoo==0.6.0
requests

Note:
Fake LLM mode requires no external dependencies.
Real LLM mode requires Ollama + Mistral. 

8.Installation & How to Run

Follow these steps to make sure you can run the project easily

1. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

2. Install dependencies
pip install -r requirements.txt

numpy
scipy
pymoo==0.6.0
requests

These are the only Python packages required.

3. (Optional) Install Ollama for real LLM mode

If you want to run the system with an actual language model:

Install Ollama from https://ollama.ai

Pull the Mistral model:

ollama pull mistral

In source/llm_agent.py, change:

USE_FAKE_LLM = False

Note: real LLM mode is slower and may time out; the default fake-LLM mode is recommended for reproducibility.

4. Test LLM functionality
python test_llm.py

This prints the action selected by the (fake or real) LLM.

5. Run the full falsification experiments
python main.py


This will automatically run falsification for:

-Pedestrian Crossing

-Lead Vehicle Sudden Braking

-Static Obstacle

For each scenario, you will see:

-NSGA-II optimization progress

-Crash probability estimates

-Violating parameter sets found (initial speeds/distances)

-Fake-LLM mode completes all three scenarios in a few seconds.


Thank you for your attention :)
