# TORCS AI Racing Controller

This repository contains the implementation of an AI-based car racing controller for the **TORCS (The Open Racing Car Simulator)** framework, developed in Python as part of the **AI 2002** course project. The project aims to design a controller that races competitively on various tracks, using telemetry data from sensors to navigate, avoid obstacles, and optimize speed.

## Project Overview

The objective is to create a Python-based controller for a car racing bot in TORCS that competes on unknown tracks, either alone or against other bots. The controller uses telemetry data from sensors (e.g., track limits, car state, lap time) to make real-time driving decisions. Key requirements include:
- **No rule-based controllers**: The controller must use AI techniques (e.g., machine learning, reinforcement learning, or neural networks).
- **Performance Metrics**: Optimize speed, track following, and obstacle avoidance.
- **Simplified Environment**: Noisy sensors, car damage, and fuel consumption are disabled for simplicity.

The project has two deliverables:
1. **Deliverable 1**: Implementation of telemetry (sensor data processing).
2. **Deliverable 2**: Final controller implementation, submitted as a self-contained archive with all execution files.

This repository includes the code for both deliverables.


## Prerequisites

To run this project, you need:
- **TORCS 1.3.4**: Install the TORCS simulator with the competition server patch.
- **Python 3.x**: The client is implemented in Python.
- **Dependencies**: Listed in `requirements.txt`.
- **Operating System**: Linux or Windows (Mac OS X is not supported).

## Installation

1. **Install TORCS**:
   - Download TORCS 1.3.4 from [SourceForge](https://sourceforge.net/projects/torcs/).
   - Apply the competition server patch (`scr-linux-patch.tgz` or `scr-win-patch.zip`) as described in the [Competition Manual](docs/Competition_Manual.pdf).
   - For Linux:
     ```bash
     tar xfvj torcs-1.3.4.tar.bz2
     cd torcs-1.3.4
     tar xfvz scr-linux-patch.tgz
     cd scr-patch
     sh do-patch.sh
     cd ..
     ./configure && make && make install && make datainstall
     ```
   - For Windows: Install TORCS using the installer and unzip `scr-win-patch.zip` into the TORCS directory, overwriting files as prompted.

2. **Set Up the Python Environment**:
   - Clone this repository:
     ```bash
     git clone https://github.com/<your-username>/TORCS-AI-Racing-Controller.git
     cd TORCS-AI-Racing-Controller
     ```
   - Install dependencies:
     ```bash
     pip install -r requirements.txt
     ```

3. **Configure TORCS**:
   - Launch TORCS and select `Race > Quick Race > Configure Race`.
   - Add the `scr_server` bot to the race participants.
   - Choose a track, set race length (laps or distance), and select display mode (`normal` or `results only`).
   - Alternatively, edit the `quickrace.xml` or `practice.xml` configuration files as described in the [Competition Manual](docs/Competition_Manual.pdf).

## Running the Controller

1. **Start the TORCS Server**:
   - Run TORCS with the competition server:
     ```bash
     torcs -nofuel -nodamage -nolaptime
     ```
   - Select `Race > Quick Race > New Race` or `Practice > New Race`. The server will wait for the client on port 3001 (default).

2. **Run the Python Client**:
   - Execute the client script:
     ```bash
     python client/controller.py host:localhost port:3001 id:SCR
     ```
   - Optional parameters:
     - `maxEpisodes`: Number of learning episodes (default: 1).
     - `maxSteps`: Max control steps per episode (default: 0, unlimited).
     - `track`: Track name (default: unknown).
     - `stage`: Competition stage (0: Warm-up, 1: Qualifying, 2: Race, 3: Unknown).

3. **Telemetry (Deliverable 1)**:
   - The `telemetry.py` script processes sensor data (e.g., track sensors, car state) as per the project requirements.
   - Run independently to test telemetry:
     ```bash
     python client/telemetry.py
     ```

4. **Controller (Deliverable 2)**:
   - The `controller.py` script implements the AI controller, using telemetry data to control the car (steering, acceleration, braking, gear).
   - Note: Documentation for Deliverable 2 is not yet available but will be added to the `report/` directory.

## Evaluation

The controller will be evaluated in a tournament with multiple phases:
- **Tracks**: Selected from a predefined set, consistent within each phase.
- **Metric**: Time to complete the race (shorter is better).
- **Advancement**: Winners of each phase advance to the next.

Collisions or off-track driving are not penalized directly, but they are unlikely to result in competitive times due to the TORCS physics engine.

## Deliverables

1. **Deliverable 1 (Telemetry)**:
   - Implemented in `client/telemetry.py`.
   - Processes sensor data (e.g., track sensors, car state, lap time).
   - Submitted by March 27, 2025.

2. **Deliverable 2 (Final Controller)**:
   - Implemented in `client/controller.py`.
   - Self-contained archive with all execution files (included in this repository).
   - Accompanied by a two-page report (to be added in `report/report.pdf`).

## TODO

- Add detailed documentation for Deliverable 2.
- Finalize and upload the two-page report explaining the controller design and training method.
- Optimize the controller for better performance on complex tracks.

## Resources

- **TORCS Installation Tutorial**: [YouTube Video](https://youtu.be/EqR9v6xhXIU)
- **Competition Manual**: [Simulated Car Racing Championship Manual](docs/Competition_Manual.pdf)
- **TORCS Website**: [torcs.org](http://www.torcs.org)
- **CIG Project Page**: [cig.sourceforge.net](http://cig.sourceforge.net/)

## Contact

For questions or issues, contact the project team or refer to the course instructor for AI 2002. Bug reports can be emailed to [scr@gecocompetitions.com](mailto:scr@gecocompetitions.com).

---

**Good Luck Racing!**
```


