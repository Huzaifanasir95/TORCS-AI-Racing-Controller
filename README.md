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

## Key Components

### 1. Neural Network-Based Control

The core of the racing AI is a neural network model (`RacingModel`) that predicts optimal steering, acceleration, and braking values based on the current state of the racing car and track. 

- **Model Architecture**: Multi-layer neural network with input size of 30 features and output size of 3 (steering, acceleration, braking)
- **Optimization Techniques**:
  - JIT compilation with PyTorch for faster inference
  - Batch norm layer fusion
  - Dropout disabling during inference
  - Model warmup phase
  - Prediction caching system with TTL

### 2. Racing Line Optimization

The `RacingLineOptimizer` class calculates the optimal path through the track, implementing racing techniques that professional drivers use:

- **Track Analysis**: Processes track sensor readings to determine track width, curvature, and segment type
- **Racing Line Calculation**: Implements the racing line principle of "out-in-out" for cornering
  - Wide entry into turn (outside)
  - Hit the apex (inside at the middle of the turn)
  - Wide exit (outside)
- **Speed Profile**: Adjusts speed targets based on track curvature and segment type
- **Curvature Estimation**: Uses sensor data to estimate track curvature and classify segments

### 3. Track Segment Analysis

The driver dynamically identifies different track segments and optimizes driving behavior for each:

- **Segment Types**: 
  - Straight sections
  - Left turns
  - Right turns
  - Complex turns (chicanes, S-curves)
  - Transitional segments
- **Analysis Techniques**:
  - Statistical variance analysis of track sensors
  - Gradient calculations for detecting track width changes
  - Gaussian smoothing for noise reduction
- **Adaptations**: Each segment type triggers specific optimizations for:
  - Target racing line
  - Cornering speed
  - Steering sensitivity
  - Braking points

### 4. Smart Collision and Reverse System

A sophisticated collision detection and recovery system enables the car to recover from crashes efficiently:

- **Stuck Detection**: Identifies when the car is stuck using multiple criteria:
  - Low speed with bad track position
  - Abnormal angles combined with obstacles
  - Proximity to walls
- **Smart Reverse**: Implements a recovery strategy when stuck:
  - Detects which side of the track the collision occurred on (left/right)
  - Calculates optimal steering direction during reverse based on collision side
  - Reverses for 2-3 seconds to gain clearance
  - Uses space analysis to determine best escape direction
- **Recovery Enhancement**:
  - Increased steering sensitivity after reversing
  - Progressive steering adjustment during reverse maneuver
  - Multi-phase recovery process

### 5. Adaptive Race Strategy

The driver adapts its behavior based on race conditions:

- **Mode Selection**:
  - Aggressive: Maximum performance, higher risks
  - Balanced: Moderate risks and performance
  - Safe: Prioritizes avoiding damage
- **Contextual Adaptation**: Selects mode based on:
  - Current race position
  - Damage level
  - Race stage (qualifying vs. race)
- **Dynamic Speed Adjustments**: Modifies speed targets based on:
  - Proximity to track edges
  - Visibility/distance to upcoming obstacles
  - Current damage levels

### 6. Gear Management

Implements advanced gear shifting logic for optimal acceleration and speed:

- **Context-Aware Shifting**: Different RPM thresholds based on:
  - Track segment type
  - Current speed
  - Acceleration/braking state
- **Speed-Based Gear Selection**: Lookup tables for optimal gear at different speeds
- **Adaptive Shifting**: Delays upshifts during hard acceleration

## Technical Implementation Details

### Sensor Processing

The driver collects and processes 30 different sensor inputs including:

- **Track Sensors**: 19 rangefinder readings showing distance to track edges
- **Car State**: Speed (X,Y,Z components), angle, track position, wheel spin
- **Race Information**: Damage, race position, lap times, fuel level

### Optimization Techniques

Performance optimizations to ensure real-time control:

- **Feature Caching**: Prevents redundant computations for similar states
- **Prediction Batching**: Groups similar states for batch prediction
- **Prefetching**: Predicts likely future states in advance
- **Model Optimization**: JIT compilation, operation fusion
- **Selective Processing**: Heavy computations only run when necessary

### Error Handling and Robustness

Several safeguards ensure continued operation:

- **Dependency Checking**: Verifies required libraries
- **Graceful Degradation**: Falls back to simpler algorithms when advanced libraries unavailable
- **Sensor Validation**: Handles missing or corrupted sensor data
- **State Persistence**: Maintains history of recent states for trend analysis

## Algorithmic Highlights

### Curvature-Based Racing Line

The curvature-based racing line algorithm:

1. Calculates track curvature using gradient analysis of sensor readings
2. Determines optimal track position based on curvature and car state
3. Applies dynamic correction factors based on speed
4. Uses non-linear adjustments to prevent overcorrection

### Smart Reverse Logic

When stuck, the reverse logic:

1. Analyzes available space using left/right sensor comparisons
2. Determines optimal reverse steering direction based on collision side
3. Adjusts steering angle based on the car's current angle
4. Implements a time-based phase approach: 
   - Initial sharp steering to escape
   - Progressive straightening as reversing continues
   - Increased steering sensitivity after resuming normal driving

### Dynamic Speed Management

Speed control is handled by:

1. Establishing baseline target speed using segment type
2. Applying adjustments for local track features
3. Calculating speed reduction factors for track edges and obstacles
4. Implementing proportional control for acceleration/braking
5. Adding contextual awareness for upcoming track features

## Model Training

The neural network model was trained on:

- Collected driving data from expert demonstrations
- Augmented with synthetic variations
- Normalized using standard scaling
- Trained with mean squared error loss function
- Validated against reference tracks

## Performance Metrics

The driver's performance is measured by:

- Lap time optimization
- Crash reduction
- Racing line adherence
- Recovery efficiency from incidents
- Adaptation to different tracks

## Requirements

- Python 3.x
- PyTorch
- NumPy
- SciPy (optional, enables advanced features)
- Pandas (optional, enables enhanced data analysis)
- Joblib (for model loading)

## Files and Structure

- **driver.py**: Main driver class with core logic
- **carState.py**: Car state representation
- **carControl.py**: Control interface
- **msgParser.py**: Communication parsing
- **best_racing_model.pth**: Trained neural network weights
- **scaler.pkl**: Feature scaler for normalizing inputs
- **racing_line_optimizer.py**: Optimizes racing lines

## Future Improvements

Potential areas for enhancement:

- Reinforcement learning for continuous improvement
- Opponent modeling for strategic overtaking
- Weather condition adaptation
- Tire wear and fuel consumption modeling
- Enhanced track memory for multi-lap optimization

## Technical Challenges Solved

This implementation addresses several challenging aspects of autonomous racing:

1. **Balancing ML and Rules**: Combines neural network predictions with racing domain knowledge
2. **Recovery Behavior**: Smart reverse implementation with collision side detection
3. **Performance Optimization**: Real-time inference with prefetching and caching
4. **Racing Line Calculation**: Dynamic racing line adaptation for unknown tracks
5. **Segment Classification**: Automatic identification of track segments without prior knowledge

## Conclusion

This TORCS racing AI driver represents a sophisticated approach to autonomous racing, combining machine learning with traditional algorithms to achieve optimal performance. The system demonstrates how domain knowledge can enhance machine learning systems, particularly in dynamic control scenarios where pure data-driven approaches may struggle.


