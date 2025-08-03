🎯 TORCS AI Racing Controller
AI-Powered Autonomous Racing for TORCS


📖 Table of Contents

🎯 Overview
✨ Features
🏗️ Architecture
🚀 Quick Start
📁 Project Structure
🔧 Configuration
🛠️ Development
📊 Analytics
🚀 Deployment
🔍 API Documentation
🤝 Contributing
📄 License
📞 Support
🙏 Acknowledgments


🎯 Overview
The TORCS AI Racing Controller is an intelligent Python-based controller for the TORCS (The Open Racing Car Simulator) framework, developed as part of the AI 2002 course project. The controller leverages AI techniques to race competitively on unknown tracks, processing real-time telemetry data to optimize speed, track following, and obstacle avoidance.
🎯 Problem We Solve

Manual racing strategies are time-consuming and ineffective for dynamic tracks.
Complex track navigation without prior knowledge.
Balancing speed and stability in real-time racing scenarios.
Lack of robust recovery from collisions or off-track incidents.

💡 Our Solution
An AI-driven racing controller that provides:

Neural network-based control for steering, acceleration, and braking.
Dynamic racing line optimization for professional-grade cornering.
Smart collision recovery with adaptive reverse logic.
Real-time telemetry processing for track and car state analysis.
Adaptive race strategies for varying track conditions.


✨ Features
🤖 AI-Powered Controller

Neural network model for real-time driving decisions.
Processes 30+ sensor inputs (track sensors, car state, race info).
Optimized with PyTorch JIT compilation and prediction caching.
Trained on expert driving data with synthetic augmentations.

🛤️ Racing Line Optimization

Implements "out-in-out" cornering for optimal track navigation.
Dynamic curvature estimation using sensor data.
Speed profile adjustments based on track segment types.
Statistical analysis for segment classification (straights, turns, chicanes).

🚗 Smart Collision Recovery

Detects stuck scenarios using speed, angle, and proximity metrics.
Smart reverse logic with side-specific steering adjustments.
Multi-phase recovery process for efficient track re-entry.

🏎️ Adaptive Race Strategy

Supports aggressive, balanced, and safe driving modes.
Contextual adaptations based on race position and track conditions.
Dynamic speed and gear management for optimal performance.

📡 Telemetry Processing

Real-time processing of track sensors, car state, and race metrics.
Robust error handling for missing or corrupted sensor data.
Feature caching and selective processing for performance.


🏗️ Architecture
graph TD
    A[Python Client] --> B[TORCS Server]
    A --> C[Neural Network Model]
    A --> D[Racing Line Optimizer]
    A --> E[Telemetry Processor]
    C --> F[PyTorch Backend]
    D --> G[Track Segment Analyzer]
    E --> H[Sensor Data Handler]
    
    I[User] --> A
    B --> J[TORCS Simulator]

    subgraph "AI Components"
        C
        F
        D
        G
        E
        H
    end

🧩 Technology Stack



Category
Technology
Purpose



Core Language
Python 3.x
Main programming language


AI/ML
PyTorch
Neural network implementation


Simulator
TORCS 1.3.4
Racing simulation environment


Data Processing
NumPy, SciPy
Sensor data analysis


Utilities
Pandas, Joblib
Data analysis and model loading


Version Control
Git
Source code management


CI/CD
GitHub Actions
Automated testing and deployment



🚀 Quick Start
📋 Prerequisites

TORCS 1.3.4: Install with competition server patch.
Python 3.x: Ensure compatibility with dependencies.
Git: For version control.
Operating System: Linux or Windows (Mac OS X not supported).

⚡ Installation

Clone the Repository
git clone https://github.com/<your-username>/TORCS-AI-Racing-Controller.git
cd TORCS-AI-Racing-Controller


Install TORCS

Download TORCS 1.3.4 from SourceForge.
Apply the competition server patch:
Linux:tar xfvj torcs-1.3.4.tar.bz2
cd torcs-1.3.4
tar xfvz scr-linux-patch.tgz
cd scr-patch
sh do-patch.sh
cd ..
./configure && make && make install && make datainstall


Windows: Install TORCS and unzip scr-win-patch.zip into the TORCS directory.




Install Dependencies
pip install -r requirements.txt


Configure TORCS

Launch TORCS and select Race > Quick Race > Configure Race.
Add scr_server bot to participants.
Edit quickrace.xml or practice.xml as needed (see Competition Manual).


Run the Application

Start TORCS server:torcs -nofuel -nodamage -nolaptime


Run the Python client:python client/controller.py host:localhost port:3001 id:SCR






📁 Project Structure
TORCS-AI-Racing-Controller/
│
├── 📁 .github/ # GitHub Actions workflows
│   └── workflows/
│       ├── ci-cd-pipeline.yml # CI/CD pipeline
│       ├── test.yml # Automated tests
│       └── lint.yml # Code linting
│
├── 📁 client/ # Python client implementation
│   ├── controller.py # Main AI controller
│   ├── telemetry.py # Telemetry processing
│   ├── driver.py # Core driver logic
│   ├── carState.py # Car state representation
│   ├── carControl.py # Control interface
│   ├── msgParser.py # Communication parsing
│   ├── racing_line_optimizer.py # Racing line optimization
│   ├── best_racing_model.pth # Trained model weights
│   └── scaler.pkl # Feature scaler
│
├── 📁 docs/ # Documentation
│   ├── Competition_Manual.pdf # Competition manual
│   ├── telemetry.md # Telemetry documentation
│   └── report/ # Report folder (WIP)
│       └── report.pdf # Final report (to be added)
│
├── 📁 report/ # Project report
│   └── report.pdf # Two-page controller report (WIP)
│
├── .gitignore # Git ignore rules
├── requirements.txt # Python dependencies
├── README.md # This file
└── LICENSE # MIT License


🔧 Configuration
🌍 Environment Variables



Variable
Description
Required
Example



TORCS_HOST
TORCS server host
✅
localhost


TORCS_PORT
TORCS server port
✅
3001


TORCS_ID
Client ID for TORCS
✅
SCR


MAX_EPISODES
Number of learning episodes
❌
1


MAX_STEPS
Max steps per episode
❌
0 (unlimited)


TRACK_NAME
Track name
❌
unknown


STAGE
Competition stage (0-3)
❌
3 (unknown)


🔧 TORCS Setup

Install TORCS with the competition server patch.
Configure race settings via quickrace.xml or practice.xml.
Ensure scr_server bot is enabled in race configuration.


🛠️ Development
📝 Available Scripts
# Development
python client/controller.py # Run the controller
python client/telemetry.py # Test telemetry processing
pip install -r requirements.txt # Install dependencies

# Testing
pytest tests/ # Run unit tests
python client/test_sensors.py # Test sensor processing

🏗️ Development Workflow

Feature Development
git checkout -b feature/your-feature-name
# Make changes
pytest tests/
git commit -m "feat: add your feature"
git push origin feature/your-feature-name


Testing
pytest tests/ # Run all tests
python client/test_sensors.py # Test sensor data


Code Quality

Use Flake8 for linting.
Ensure Python type hints are used.
Follow PEP 8 style guidelines.



🔍 Debugging

Check Sensor Data
python client/telemetry.py


Monitor TORCS Server

Verify server is running on localhost:3001.
Check race logs in TORCS UI.


Model Debugging

Validate model weights (best_racing_model.pth).
Test feature scaler (scaler.pkl).




📊 Analytics
📈 Performance Metrics

Lap Time: Time to complete a race lap.
Crash Rate: Frequency of collisions or off-track incidents.
Racing Line Adherence: Deviation from optimal racing line.
Recovery Efficiency: Time to recover from stuck scenarios.
Speed Optimization: Average speed per track segment.

📋 Data Collection

Sensor data logs for analysis.
Race performance metrics (lap times, crashes).
Model prediction accuracy.
Error tracking for sensor or control failures.


🚀 Deployment
☁️ Local Deployment

Start TORCS Server
torcs -nofuel -nodamage -nolaptime


Run Python Client
python client/controller.py host:localhost port:3001 id:SCR



🔄 CI/CD Pipeline
GitHub Actions workflows are included:

CI/CD Pipeline (ci-cd-pipeline.yml):
Runs tests and linting.
Validates model and telemetry functionality.


Test Workflow (test.yml):
Executes unit tests for telemetry and controller.


Lint Workflow (lint.yml):
Enforces code style with Flake8.




🔍 API Documentation
📡 Endpoints
The controller communicates with the TORCS server via a socket-based API. Key interactions include:



Action
Description
Parameters



connect
Connect to TORCS server
host, port, id


send_control
Send control commands
steering, acceleration, braking, gear


receive_sensors
Receive telemetry data
None


📝 Example Interaction
Send Control Command
# Send steering, acceleration, and braking
control = {"steering": 0.5, "accel": 0.8, "brake": 0.0, "gear": 3}
client.send_control(control)

Receive Sensor Data
# Receive telemetry data
sensors = client.receive_sensors()
# Example output
{
  "track_sensors": [10.0, 8.5, 9.2, ...], # 19 rangefinder readings
  "speed": {"x": 50.0, "y": 0.0, "z": 0.0},
  "angle": 0.1,
  "track_pos": 0.0
}


🤝 Contributing
🌟 How to Contribute

Fork the Repository

Create a Feature Branch
git checkout -b feature/amazing-feature


Make Changes

Run Tests
pytest tests/


Commit Changes
git commit -m 'feat: add amazing feature'


Push and Open Pull Request
git push origin feature/amazing-feature



📋 Contribution Guidelines

Follow PEP 8 style guidelines.
Add tests for new features.
Update documentation in docs/.
Use conventional commit messages.
Ensure tests pass before submitting PR.

🐛 Bug Reports
Use the GitHub issue tracker to report bugs, including:

Issue description.
Steps to reproduce.
Expected vs. actual behavior.
Environment details (OS, Python version, TORCS version).

💡 Feature Requests
Submit feature requests via GitHub issues with:

Feature description.
Use case and benefits.
Proposed implementation (optional).


📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

📞 Support

GitHub Issues: Report bugs or request features
Documentation: See /docs for guides and manuals.
Competition Manual: Competition_Manual.pdf
TORCS Resources: TORCS Website, CIG Project Page


🙏 Acknowledgments

TORCS Team for the open-source racing simulator.
PyTorch Team for the machine learning framework.
AI 2002 Course for project inspiration and guidelines.
SourceForge for hosting TORCS downloads.
GitHub for version control and CI/CD support.


Empowering autonomous racing with AI-driven control and optimization
