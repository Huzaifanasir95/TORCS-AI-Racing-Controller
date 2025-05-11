import msgParser
import carState
import carControl
import csv
import keyboard
import torch
import torch.nn as nn
import torch.jit
import joblib
import numpy as np
import time
import os
import sys
import math
import numpy as np
from collections import deque
from functools import lru_cache
try:
    from scipy.interpolate import interp1d, splprep, splev
    from scipy.ndimage import gaussian_filter1d
    from scipy.spatial import distance
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("SciPy not available. Some advanced features will be disabled.")

# Optional imports for enhanced racing line calculation
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Pandas not available. Advanced data analysis features will be disabled.")

# Global constants for optimization
MODEL_CACHE_SIZE = 5  # Number of model predictions to cache
FEATURE_HASH_PRECISION = 3  # Decimal places for feature hashing
INFERENCE_BATCH_SIZE = 4  # Batch size for model inference when applicable
MODEL_OPTIMIZATION_LEVEL = 2  # 0=None, 1=Basic, 2=Advanced

# Define the neural network model architecture
class RacingModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(RacingModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.fc5 = nn.Linear(64, output_size)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        # Handle both single sample and batched input
        is_single = x.dim() == 1
        if is_single:
            x = x.unsqueeze(0)  # Add batch dimension
            
        x = self.dropout(torch.relu(self.bn1(self.fc1(x))))
        x = self.dropout(torch.relu(self.bn2(self.fc2(x))))
        x = self.dropout(torch.relu(self.bn3(self.fc3(x))))
        x = self.dropout(torch.relu(self.bn4(self.fc4(x))))
        x = self.fc5(x)
        x = torch.tanh(x)  # Normalize outputs to [-1, 1]
        
        # Return single result if input was single
        if is_single:
            return x.squeeze(0)
        return x
        
    def optimize(self):
        """Optimize model for inference"""
        # Fuse batch norm with linear where possible for faster inference
        for i in range(1, 5):
            fc = getattr(self, f'fc{i}')
            bn = getattr(self, f'bn{i}')
            
            # Extract parameters
            weight = fc.weight
            bias = fc.bias if fc.bias is not None else torch.zeros_like(bn.running_mean)
            
            # Scale weights
            scaled_weight = weight * (bn.weight / torch.sqrt(bn.running_var + bn.eps)).unsqueeze(1)
            
            # Scale bias
            scaled_bias = (bias - bn.running_mean) * bn.weight / torch.sqrt(bn.running_var + bn.eps) + bn.bias
            
            # Update parameters
            fc.weight.data.copy_(scaled_weight)
            if fc.bias is None:
                fc.bias = nn.Parameter(scaled_bias)
            else:
                fc.bias.data.copy_(scaled_bias)
                
            # Set batch norm to identity function
            bn.running_mean = torch.zeros_like(bn.running_mean)
            bn.running_var = torch.ones_like(bn.running_var)
            bn.weight.data.fill_(1.0)
            bn.bias.data.fill_(0.0)
            bn.eps = 0.0

# Advanced racing line and track position optimization
class RacingLineOptimizer:
    def __init__(self, track_width_estimate=20.0, safety_margin=1.0):
        self.track_width = track_width_estimate  # Estimated full track width
        self.safety_margin = safety_margin       # Safety margin from edges
        self.track_points = []                   # Track points for path planning
        self.racing_line = []                    # Optimized racing line
        self.curvature_profile = []              # Track curvature at each point
        self.segment_speeds = {}                 # Optimal speeds for segments
        self.segment_entry_angles = {}           # Optimal entry angles
        self.segment_exit_points = {}            # Optimal exit points
        
        # Racing line parameters
        self.apex_offset = 0.7                   # How close to inside at apex (0-1)
        self.entry_offset = 0.4                  # Position on entry (0-1, 0=inside)
        self.exit_offset = 0.3                   # Position on exit (0-1, 0=inside)
        
        # Performance metrics
        self.last_update_time = 0
        self.update_frequency = 2.0              # Seconds between full updates
        
    def update_track_data(self, track_sensors, angle, speed, track_pos, segment_type):
        """Add new track reading to build racing line"""
        # Construct track profile from sensor readings
        left_distance = track_sensors[0]
        right_distance = track_sensors[18]
        total_width = left_distance + right_distance
        
        # Store track point with normalized position
        self.track_points.append({
            'left': left_distance,
            'right': right_distance,
            'width': total_width,
            'angle': angle,
            'speed': speed,
            'pos': track_pos,
            'segment': segment_type
        })
        
        # Keep a reasonable history
        if len(self.track_points) > 100:
            self.track_points.pop(0)
            
    def estimate_curvature(self, window_size=5):
        """Estimate track curvature from sensor data"""
        if len(self.track_points) < window_size + 2:
            return 0.0
            
        # Use recent track points to estimate curvature
        recent_points = self.track_points[-window_size:]
        
        if SCIPY_AVAILABLE:
            # Estimate curvature using left/right differences
            left_dists = [p['left'] for p in recent_points]
            right_dists = [p['right'] for p in recent_points]
            left_smooth = gaussian_filter1d(left_dists, sigma=1.0)
            right_smooth = gaussian_filter1d(right_dists, sigma=1.0)
            
            # Calculate rate of change in track boundaries
            left_derivative = np.gradient(left_smooth)
            right_derivative = np.gradient(right_smooth)
            
            # Positive curvature = right turn, negative = left turn
            curvature = np.mean(right_derivative - left_derivative) / 5.0
            return curvature
        else:
            # Simple estimation without scipy
            left_diff = recent_points[-1]['left'] - recent_points[0]['left']
            right_diff = recent_points[-1]['right'] - recent_points[0]['right']
            return (right_diff - left_diff) / window_size
    
    def get_optimal_position(self, segment_type, angle, curvature, track_progress, speed):
        """Calculate optimal track position based on racing principles"""
        # Default middle of track
        position = 0.0
        
        # Base position on segment type
        if segment_type == "straight":
            # On straights, stay in the middle with slight adjustments for upcoming turns
            position = 0.0 + curvature * 0.5  # Prepare for upcoming turns
        
        elif segment_type == "left_turn":
            # For left turns: start wide (right), hit apex (left), exit wide (right)
            # Use sigmoid function to smoothly transition between phases
            
            # Simplified optimal racing line for left turns
            if SCIPY_AVAILABLE:
                # Simulate track progress within turn (0-1)
                # Convert angle to a progress metric (assume larger angles mean deeper in the turn)
                progress = min(1.0, max(0.0, (abs(angle) - 0.1) / 0.5))
                
                # Setup points for racing line
                x = np.array([0.0, 0.3, 0.5, 0.8, 1.0])  # Track progress points
                y = np.array([0.4, 0.6, 0.7, 0.3, 0.1])  # Offset values (0=left, 1=right)
                
                # Create smooth racing line using spline interpolation
                racing_line = interp1d(x, y, kind='cubic', bounds_error=False, fill_value=(y[0], y[-1]))
                
                # Get position from racing line (convert from 0-1 to -1 to 1 scale)
                # For left turns: positive = right side (outside at entry)
                position = racing_line(progress) * 2.0 - 1.0
            else:
                # Without scipy, use simple phase-based racing line
                if abs(angle) < 0.2:  # Entry
                    position = 0.5  # Start wide right
                elif abs(angle) < 0.4:  # Approach apex
                    position = 0.1  # Move toward inside
                else:  # Exit
                    position = -0.3  # Gradually move back right
        
        elif segment_type == "right_turn":
            # For right turns: start wide (left), hit apex (right), exit wide (left)
            if SCIPY_AVAILABLE:
                progress = min(1.0, max(0.0, (abs(angle) - 0.1) / 0.5))
                
                # Setup points for racing line (mirror of left turn)
                x = np.array([0.0, 0.3, 0.5, 0.8, 1.0])
                y = np.array([0.6, 0.4, 0.3, 0.7, 0.9])  # Offset values (0=left, 1=right)
                
                racing_line = interp1d(x, y, kind='cubic', bounds_error=False, fill_value=(y[0], y[-1]))
                
                # Convert from 0-1 to -1 to 1 scale (negative = left side, outside at entry)
                position = racing_line(progress) * 2.0 - 1.0
            else:
                # Simple phase-based approach
                if abs(angle) < 0.2:  # Entry
                    position = -0.5  # Start wide left
                elif abs(angle) < 0.4:  # Approach apex
                    position = -0.1  # Move toward inside
                else:  # Exit
                    position = 0.3  # Gradually move back left
        
        elif segment_type == "complex_turn":
            # For complex turns, use curvature to adapt
            # Positive curvature = right turn, negative = left turn
            if abs(curvature) > 0.05:
                # Determine if it's more like a left or right turn based on curvature
                if curvature < 0:  # More like a left turn
                    position = min(0.6, max(-0.7, 0.5 * math.sin(angle * 3.0)))
                else:  # More like a right turn
                    position = min(0.7, max(-0.6, -0.5 * math.sin(angle * 3.0)))
            else:
                # Minimal curvature, treat like a straight
                position = 0.0
        
        # Speed-based adjustments: at very high speeds, stay more central for stability
        if speed > 200:
            position = position * 0.7  # Reduce extremes at high speed
            
        # Apply safety constraints to keep within track
        position = min(0.8, max(-0.8, position))
            
        return position
        
    def optimize_racing_line(self, track_sensors, angle, speed, track_pos, segment_type, distance_covered):
        """Full racing line optimization with predictive planning"""
        current_time = time.time()
        
        # Update track data for future reference
        self.update_track_data(track_sensors, angle, speed, track_pos, segment_type)
        
        # Calculate current track curvature
        curvature = self.estimate_curvature()
        
        # Track progress within current segment (simplified)
        track_progress = 0.5  # Middle of segment by default
        if abs(angle) > 0.0:
            # Use angle as a proxy for progress through a turn
            track_progress = min(1.0, abs(angle) / 0.8)
        
        # Get optimal racing line position
        optimal_position = self.get_optimal_position(
            segment_type, angle, curvature, track_progress, speed)
        
        # Calculate how aggressively to correct toward optimal line
        # At higher speeds, make gentler corrections
        correction_factor = 1.0 - min(0.7, speed / 300.0)
        
        # Return target position and curvature data
        return {
            'target_position': optimal_position,
            'curvature': curvature,
            'correction_factor': correction_factor,
            'track_width': self.track_width
        }

class Driver(object):
    def __init__(self, stage, model_path='best_racing_model.pth', scaler_path='scaler.pkl'):
        self.WARM_UP = 0
        self.QUALIFYING = 1
        self.RACE = 2
        self.UNKNOWN = 3
        self.stage = stage
        
        self.parser = msgParser.MsgParser()
        self.state = carState.CarState()
        self.control = carControl.CarControl()
        
        self.steer_lock = 0.905398  # ~45 degrees in radians
        self.max_speed = 300  # Increased target speed
        self.prev_rpm = None
        
        # Enhanced stuck detection and recovery
        self.stuck_timer = 0
        self.stuck_speed_threshold = 12  # km/h
        self.stuck_time_threshold = 5  # seconds - detect stuck faster
        self.reverse_duration = 0
        self.max_reverse_duration = 3.0  # Allow more time for reverse if needed
        self.last_update_time = time.time()
        self.is_stuck = False
        self.prev_steer = 0.0  # For smoothing AI steering
        self.reverse_start_time = None
        self.reverse_steer = 0.0  # Store steering direction during reverse
        self.reverse_phase = 0  # Multi-phase recovery: 0=initial, 1=turning, 2=straightening
        self.reverse_progress = 0.0  # Track recovery progress from 0 to 1
        self.recovery_path = []  # Store points along recovery path
        self.initial_stuck_pos = None  # Store position when first stuck
        self.initial_stuck_angle = None  # Store angle when first stuck
        self.recovery_attempts = 0  # Track number of recovery attempts
        self.last_recovery_time = 0  # Time of last recovery
        self.recovery_cooldown = 10  # Seconds before attempting another recovery
        
        # Advanced driving features
        self.track_history = deque(maxlen=100)  # Store recent track readings
        self.speed_history = deque(maxlen=20)   # Store recent speed readings
        self.segment_type = "unknown"           # Current track segment type
        self.racing_line_offset = 0.0           # Optimal position on track (-1 to 1)
        self.cornering_speed_factor = 1.0       # Dynamic speed adjustment for corners
        self.adaptive_mode = "balanced"         # Driving mode (aggressive, balanced, safe)
        self.last_segment_distance = 0          # For segment detection
        self.segment_data = {}                  # Track segment information
        
        # Enhanced model optimization
        self.model_inference_time = 0           # Track model inference time
        self.model_inference_count = 0          # Count of model inferences
        self.batch_size = INFERENCE_BATCH_SIZE  # For batch processing
        self.warmup_done = False                # Flag for warmup completion
        self.calibration_count = 0              # For sensor calibration
        self.sensor_calibration = {}            # Calibration data for sensors
        
        # Advanced caching for model predictions
        self.feature_cache = deque(maxlen=INFERENCE_BATCH_SIZE)  # Recent feature vectors
        self.prediction_cache = {}              # Cache for model predictions
        self.cache_hit_count = 0                # Count cache hits
        self.cache_miss_count = 0               # Count cache misses
        self.last_prediction_time = time.time() # For prediction throttling
        self.prediction_cache_ttl = 0.05        # 50ms cache lifetime
        self.last_features_hash = None          # Hash of last feature set
        self.model_scripted = None              # JIT-compiled model
        
        # Racing line optimizer
        self.racing_line_optimizer = RacingLineOptimizer()
        
        # Verify dependencies
        self.check_dependencies()
        
        # Try to load AI model and scaler
        self.ai_available = False
        self.load_ai_model(model_path, scaler_path)
        self.use_ai = True  # Start with AI on
    
    def check_dependencies(self):
        """Ensure required libraries are installed"""
        required = {'torch': torch, 'joblib': joblib, 'keyboard': keyboard, 'numpy': np}
        for lib_name, module in required.items():
            if module is None:
                print(f"Error: {lib_name} is not installed. Install it with 'pip install {lib_name}'")
                sys.exit(1)
    
    def load_ai_model(self, model_path, scaler_path):
        """Load AI model and scaler with optimization"""
        try:
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")
            
            start_time = time.time()
            self.scaler = joblib.load(scaler_path)
            
            # Create model with appropriate input size
            self.model = RacingModel(input_size=30, output_size=3)  # 30 features
            
            # Load model weights
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Optimize model based on optimization level
            if MODEL_OPTIMIZATION_LEVEL >= 1:
                # Set to evaluation mode
                self.model.eval()
                
                # Disable dropout for inference
                for m in self.model.modules():
                    if isinstance(m, nn.Dropout):
                        m.p = 0.0
                        
                # Fuse batch norm layers with preceding linear layers if advanced optimization
                if MODEL_OPTIMIZATION_LEVEL >= 2:
                    print("Applying advanced model optimization...")
                    self.model.optimize()
                    
                    # Create JIT-compiled model for faster inference
                    try:
                        # Trace the model with an example input
                        example_input = torch.zeros(30, dtype=torch.float32)
                        self.model_scripted = torch.jit.trace(self.model, example_input)
                        self.model_scripted = torch.jit.optimize_for_inference(self.model_scripted)
                        print("JIT optimization successful")
                    except Exception as e:
                        print(f"JIT optimization failed: {e}")
                        self.model_scripted = None
            
            # Perform model warmup with random inputs
            if not self.warmup_done and MODEL_OPTIMIZATION_LEVEL >= 1:
                print("Warming up model...")
                with torch.no_grad():
                    # Single sample warmup
                    for _ in range(10):
                        x = torch.randn(30, dtype=torch.float32)
                        self.model(x)
                    
                    # Batch warmup
                    batch = torch.randn(INFERENCE_BATCH_SIZE, 30, dtype=torch.float32)
                    self.model(batch)
                self.warmup_done = True
            
            self.ai_available = True
            load_time = time.time() - start_time
            print(f"AI model loaded and optimized in {load_time:.2f}s")
            
        except FileNotFoundError as e:
            print(f"Failed to load AI model: {e}")
            self.ai_available = False
        except Exception as e:
            print(f"Unexpected error loading AI model: {e}")
            self.ai_available = False
    
    def init(self):
        self.angles = [0 for x in range(19)]
        for i in range(5):
            self.angles[i] = -90 + i * 15
            self.angles[18 - i] = 90 - i * 15
        for i in range(5, 9):
            self.angles[i] = -20 + (i-5) * 5
            self.angles[18 - i] = 20 - (i-5) * 5
        return self.parser.stringify({'init': self.angles})
    
    def get_sensors(self):
        """Extract sensor data for the model"""
        return {
            "Angle": self.state.angle if self.state.angle is not None else 0,
            "CurrentLapTime": self.state.curLapTime if self.state.curLapTime is not None else 0,
            "Damage": self.state.damage if self.state.damage is not None else 0,
            "DistanceFromStart": self.state.distFromStart if self.state.distFromStart is not None else 0,
            "DistanceCovered": self.state.distRaced if self.state.distRaced is not None else 0,
            "FuelLevel": self.state.fuel if self.state.fuel is not None else 0,
            "Gear": self.state.gear if self.state.gear is not None else 0,
            "LastLapTime": self.state.lastLapTime if self.state.lastLapTime is not None else 0,
            "RPM": self.state.rpm if self.state.rpm is not None else 0,
            "SpeedX": self.state.speedX if self.state.speedX is not None else 0,
            "SpeedY": self.state.speedY if self.state.speedY is not None else 0,
            "SpeedZ": self.state.speedZ if self.state.speedZ is not None else 0,
            "Track_1": self.state.track[0] if self.state.track is not None else 0,
            "Track_2": self.state.track[1] if self.state.track is not None else 0,
            "Track_3": self.state.track[2] if self.state.track is not None else 0,
            "Track_4": self.state.track[3] if self.state.track is not None else 0,
            "Track_5": self.state.track[4] if self.state.track is not None else 0,
            "Track_6": self.state.track[5] if self.state.track is not None else 0,
            "Track_7": self.state.track[6] if self.state.track is not None else 0,
            "Track_8": self.state.track[7] if self.state.track is not None else 0,
            "Track_9": self.state.track[8] if self.state.track is not None else 0,
            "Track_10": self.state.track[9] if self.state.track is not None else 0,
            "Track_11": self.state.track[10] if self.state.track is not None else 0,
            "Track_12": self.state.track[11] if self.state.track is not None else 0,
            "Track_13": self.state.track[12] if self.state.track is not None else 0,
            "Track_14": self.state.track[13] if self.state.track is not None else 0,
            "Track_15": self.state.track[14] if self.state.track is not None else 0,
            "Track_16": self.state.track[15] if self.state.track is not None else 0,
            "Track_17": self.state.track[16] if self.state.track is not None else 0,
            "RacePosition": self.state.racePos if self.state.racePos is not None else 0
        }
    
    def analyze_track_segment(self):
        """Analyze current track segment to determine optimal racing behavior using advanced algorithms"""
        track = self.state.getTrack() or [0] * 19
        speed = self.state.getSpeedX() or 0
        pos = self.state.getTrackPos() or 0
        angle = self.state.getAngle() or 0
        dist = self.state.getDistFromStart() or 0
        
        # Update track history
        self.track_history.append(track)
        self.speed_history.append(speed)
        
        # Skip if not enough history
        if len(self.track_history) < 10:
            return
            
        # Determine if we're in a new segment (check every ~50m)
        if abs(dist - self.last_segment_distance) < 50:
            return
            
        self.last_segment_distance = dist
        
        # Use scientific approach for segment classification
        if SCIPY_AVAILABLE:
            # Convert track readings to numpy arrays for analysis
            track_array = np.array(self.track_history)
            left_readings = track_array[:, 0]  
            right_readings = track_array[:, 18]
            center_readings = track_array[:, 9]
            
            # Apply Gaussian smoothing to reduce noise
            left_smooth = gaussian_filter1d(left_readings, sigma=1.0)
            right_smooth = gaussian_filter1d(right_readings, sigma=1.0)
            center_smooth = gaussian_filter1d(center_readings, sigma=1.0)
            
            # Calculate derivatives to detect changes in track width
            left_derivative = np.gradient(left_smooth)
            right_derivative = np.gradient(right_smooth)
            
            # Calculate track curvature (positive = right turn, negative = left turn)
            curvature = np.mean(right_derivative - left_derivative) / 5.0
            
            # Advanced segment classification based on curvature and variance
            left_var = np.var(left_smooth)
            right_var = np.var(right_smooth)
            center_var = np.var(center_smooth)
            
            # More sophisticated segment detection using statistical measures
            if max(left_var, right_var) < 100 and abs(curvature) < 0.1:
                self.segment_type = "straight"
                self.cornering_speed_factor = 1.0
            elif curvature < -0.2:
                self.segment_type = "left_turn"
                # Dynamically calculate cornering speed based on curvature severity
                self.cornering_speed_factor = max(0.6, 1.0 - abs(curvature) * 0.8)
            elif curvature > 0.2:
                self.segment_type = "right_turn"
                self.cornering_speed_factor = max(0.6, 1.0 - abs(curvature) * 0.8)
            elif max(left_var, right_var) > 150:
                self.segment_type = "complex_turn"
                self.cornering_speed_factor = 0.7
            else:
                self.segment_type = "transitional"
                self.cornering_speed_factor = 0.85
        else:
            # Fallback to simple variance-based approach if scipy not available
            left_readings = [t[0] for t in self.track_history]
            right_readings = [t[18] for t in self.track_history]
            center_readings = [t[9] for t in self.track_history]
            
            left_variance = np.var(left_readings)
            right_variance = np.var(right_readings)
            center_variance = np.var(center_readings)
            
            # Determine segment type based on sensor variances
            if center_variance < 100 and left_variance < 200 and right_variance < 200:
                self.segment_type = "straight"
                self.cornering_speed_factor = 1.0
            elif left_variance > right_variance * 2:
                self.segment_type = "right_turn"
                self.cornering_speed_factor = 0.7 + (0.3 * min(1.0, max(0.0, min(center_readings) / 100)))
            elif right_variance > left_variance * 2:
                self.segment_type = "left_turn"
                self.cornering_speed_factor = 0.7 + (0.3 * min(1.0, max(0.0, min(center_readings) / 100)))
            elif left_variance > 200 or right_variance > 200:
                self.segment_type = "complex_turn"
                self.cornering_speed_factor = 0.6
            else:
                self.segment_type = "unknown"
                self.cornering_speed_factor = 0.8
        
        # Get optimized racing line from advanced optimizer
        racing_line_data = self.racing_line_optimizer.optimize_racing_line(
            track, angle, speed, pos, self.segment_type, dist)
        
        # Apply advanced racing line data
        self.racing_line_offset = racing_line_data['target_position']
        self.track_curvature = racing_line_data['curvature']
        self.correction_strength = racing_line_data['correction_factor']
        
        # Remember this segment for future laps (enhanced with curvature)
        segment_key = f"{int(dist/50)}"
        self.segment_data[segment_key] = {
            "type": self.segment_type,
            "speed_factor": self.cornering_speed_factor,
            "racing_line": self.racing_line_offset,
            "curvature": self.track_curvature
        }
        
        print(f"Track segment: {self.segment_type}, Speed factor: {self.cornering_speed_factor:.2f}, " +
              f"Racing line: {self.racing_line_offset:.2f}, Curvature: {self.track_curvature:.3f}")
        
    def adapt_race_strategy(self):
        """Adapt racing strategy based on current race conditions"""
        damage = self.state.getDamage() or 0
        position = self.state.getRacePos() or 0
        laps_completed = int((self.state.getDistRaced() or 0) / 6000)  # Approximate
        
        # Adjust strategy based on race conditions
        if self.stage == self.QUALIFYING:
            # Maximum performance in qualifying
            self.adaptive_mode = "aggressive"
        elif self.stage == self.RACE:
            if damage > 3000:
                # High damage, drive more carefully
                self.adaptive_mode = "safe"
            elif position <= 3:
                # Leading positions, balanced approach
                self.adaptive_mode = "balanced"
            else:
                # Need to catch up, be more aggressive
                self.adaptive_mode = "aggressive"
        else:
            # Default to balanced mode
            self.adaptive_mode = "balanced"
            
        # Adjust speed factor based on selected mode
        mode_factors = {
            "aggressive": 1.2,
            "balanced": 1.0,
            "safe": 0.85
        }
        
        # Apply mode-specific adjustments to cornering_speed_factor
        self.cornering_speed_factor *= mode_factors.get(self.adaptive_mode, 1.0)
        
    def optimize_predicted_output(self, steering, acceleration, braking):
        """Apply advanced racing knowledge to refine model outputs using optimal racing line"""
        track = self.state.getTrack() or [0] * 19
        speed = self.state.getSpeedX() or 0
        pos = self.state.getTrackPos() or 0
        angle = self.state.getAngle() or 0
        
        # Get optimal track position from racing line optimizer
        if SCIPY_AVAILABLE:
            # Update racing line continuously for fine-grained control
            racing_line_data = self.racing_line_optimizer.optimize_racing_line(
                track, angle, speed, pos, self.segment_type, 
                self.state.getDistFromStart() or 0
            )
            
            # Apply racing line data with dynamic correction based on speed
            target_pos = racing_line_data['target_position']
            correction_factor = racing_line_data['correction_factor']
        else:
            # Fallback to simpler method
            target_pos = self.racing_line_offset
            correction_factor = 1.0 - min(0.7, speed / 300.0)  # Less aggressive at higher speeds
        
        # Calculate position error with respect to racing line
        pos_error = pos - target_pos
        
        # Create speed-based steering adjustment profile
        if SCIPY_AVAILABLE:
            # Adaptive steering correction based on segment type and speed
            # Use sigmoid function for smooth transitions between strategies
            def sigmoid(x, k=10):
                return 1 / (1 + math.exp(-k * x))
            
            # Calculate steering blend factors based on segment type and speed
            if self.segment_type == "straight":
                # On straights: stronger position correction, less model influence
                model_factor = sigmoid((speed - 100) / 100) * 0.5 + 0.3  # 0.3-0.8 range
                position_factor = 1.0 - model_factor
            elif self.segment_type.endswith("turn"):
                # In turns: more model influence, especially at lower speeds
                model_factor = sigmoid((speed - 80) / 100) * 0.3 + 0.6  # 0.6-0.9 range
                position_factor = (1.0 - model_factor) * 1.5  # Stronger position correction
            else:
                # Default mixed approach
                model_factor = 0.7
                position_factor = 0.3
                
            # Calculate steering command with dynamic mixing
            steering = (steering * model_factor) - (pos_error * position_factor * correction_factor)
            
            # Apply non-linear correction for large errors to prevent overcorrection
            if abs(pos_error) > 0.5:
                steering = steering * (0.8 + 0.2 * math.tanh(abs(pos_error)))
        else:
            # Simpler steering adjustment without scipy
            if self.segment_type == "straight" and abs(angle) < 0.1:
                # On straight with small angle, focus on centering
                steering = steering * 0.7 - pos_error * 0.3
            elif self.segment_type.endswith("_turn"):
                # In turns, emphasize model prediction but adjust for racing line
                steering = steering * 0.85 - pos_error * 0.15
            else:
                # Mixed approach for other segments
                steering = steering * 0.75 - pos_error * 0.25
        
        # Apply speed management based on track conditions and curvature
        min_front_dist = min(track[8:11]) if track else 0
        
        # Advanced speed profile based on multiple factors
        if SCIPY_AVAILABLE:
            # Base speed factor using cornering speed and segment type
            target_speed_factor = self.cornering_speed_factor
            
            # Adjust for local track curvature (not just segment type)
            curvature_adjustment = 1.0 - min(0.4, abs(self.track_curvature) * 3.0)
            target_speed_factor *= curvature_adjustment
            
            # Adjust for proximity to track edges
            edge_proximity = min(track[0], track[18]) if track else 10
            if edge_proximity < 5:
                target_speed_factor *= 0.8  # Reduce speed when close to edges
                
            # Adjust for upcoming obstacles or sharp turns
            if min_front_dist < 50:
                visibility_factor = max(0.6, min_front_dist / 50)
                target_speed_factor *= visibility_factor
        else:
            # Simpler speed management
            target_speed_factor = self.cornering_speed_factor
            
            # Handle upcoming turns detected by track sensors
            if min_front_dist < 50 and speed > 70:
                # Approaching turn or obstacle at high speed
                target_speed_factor *= 0.8
        
        # Apply intelligent speed control logic
        max_speed = self.max_speed * target_speed_factor
        
        # Hard braking scenario - obstacle or sharp turn ahead
        if (min_front_dist < 30 and speed > 100) or (min_front_dist < 20 and speed > 70):
            acceleration = 0
            braking = max(braking, min(0.8, speed / 150))  # Progressive braking based on speed
        else:
            # Adjust acceleration based on target speed
            if speed > max_speed:
                # Above target speed - apply proportional speed control
                speed_excess = (speed - max_speed) / max_speed
                acceleration *= max(0.1, 1.0 - speed_excess * 2)
                
                # Apply progressive braking proportional to excess speed
                if speed > max_speed * 1.1:  # Only brake if significantly over target
                    braking = max(braking, min(0.9, speed_excess))
                else:
                    braking = 0  # Just lift throttle for small excesses
            elif speed < max_speed * 0.8:
                # Well below target speed - increase acceleration aggressively
                acceleration = min(1.0, acceleration + 0.2 * (1.0 - speed/max_speed))
                braking = 0
            else:
                # Near target speed - fine adjustments
                acceleration = min(1.0, acceleration * (1.0 + 0.1 * (1.0 - speed/max_speed)))
                braking = 0
        
        # Apply minimum threshold for braking to avoid light brake drag
        if braking < 0.1:
            braking = 0
            
        return steering, acceleration, braking
        
    def hash_features(self, features):
        """Create a hashable representation of feature array for cache lookup"""
        if features.ndim > 1:
            features = features.flatten()
        # Round to reduce sensitivity to tiny changes
        rounded = np.round(features, FEATURE_HASH_PRECISION)
        return hash(rounded.tobytes())
    
    def predict_with_model(self, features_tensor):
        """Optimized model prediction with advanced caching"""
        current_time = time.time()
        
        # Hash the features for cache lookup
        features_hash = self.hash_features(features_tensor.numpy())
        
        # Check if we have a cached prediction for these features
        if features_hash in self.prediction_cache:
            cached_prediction, cache_time = self.prediction_cache[features_hash]
            
            # If cache is fresh enough, use it
            if current_time - cache_time < self.prediction_cache_ttl:
                self.cache_hit_count += 1
                return cached_prediction
        
        # Cache miss, need to run the model
        self.cache_miss_count += 1
        inference_start = time.time()
        
        # Store features for potential batch processing
        self.feature_cache.append(features_tensor.clone())
        
        # Use TorchScript model if available for faster inference
        if self.model_scripted is not None:
            with torch.no_grad():
                prediction = self.model_scripted(features_tensor).numpy()
        else:
            with torch.no_grad():
                prediction = self.model(features_tensor).numpy()
                
        # Update model inference metrics
        self.model_inference_time += time.time() - inference_start
        self.model_inference_count += 1
        
        # Cache the prediction
        self.prediction_cache[features_hash] = (prediction, current_time)
        
        # Periodically clean old cache entries
        if self.model_inference_count % 100 == 0:
            self.clean_prediction_cache(current_time)
            
        # Log inference statistics periodically
        if self.model_inference_count % 1000 == 0:
            avg_time = self.model_inference_time / max(1, self.model_inference_count)
            hit_rate = self.cache_hit_count / max(1, (self.cache_hit_count + self.cache_miss_count))
            print(f"Model stats: avg inference={avg_time*1000:.2f}ms, cache hit rate={hit_rate*100:.1f}%")
            
        return prediction
    
    def batch_predict(self):
        """Run prediction on a batch of inputs for better efficiency"""
        if len(self.feature_cache) < 2:
            return None
            
        # Create batch tensor from cached features
        batch = torch.stack(list(self.feature_cache))
        
        # Run batch prediction
        with torch.no_grad():
            if self.model_scripted is not None:
                results = self.model_scripted(batch).numpy()
            else:
                results = self.model(batch).numpy()
                
        # Clear feature cache after batch processing
        self.feature_cache.clear()
        
        return results
    
    def clean_prediction_cache(self, current_time):
        """Remove old entries from prediction cache"""
        keys_to_remove = []
        for feature_hash, (_, cache_time) in self.prediction_cache.items():
            if current_time - cache_time > self.prediction_cache_ttl * 10:
                keys_to_remove.append(feature_hash)
                
        for key in keys_to_remove:
            del self.prediction_cache[key]
    
    def prefetch_predictions(self):
        """Precompute predictions for likely future states to reduce latency"""
        if not self.ai_available or len(self.track_history) < 5:
            return
            
        # Only run prefetching periodically to reduce overhead
        current_time = time.time()
        if current_time - self.last_prediction_time < 0.1:
            return
            
        # Get current state
        speed = self.state.getSpeedX() or 0
        track_pos = self.state.getTrackPos() or 0
        angle = self.state.getAngle() or 0
        sensors = self.get_sensors()
        
        # Skip if we're in a recovery situation
        if self.is_stuck or abs(speed) < 10:
            return
            
        # Create synthetic feature sets for likely future states
        synthetic_features = []
        features_base = np.array([
            sensors["Angle"],
            sensors["CurrentLapTime"],
            sensors["Damage"],
            sensors["DistanceFromStart"],
            sensors["DistanceCovered"],
            sensors["FuelLevel"],
            sensors["Gear"],
            sensors["LastLapTime"],
            sensors["RPM"],
            sensors["SpeedX"],
            sensors["SpeedY"],
            sensors["SpeedZ"],
            sensors["Track_1"],
            sensors["Track_2"],
            sensors["Track_3"],
            sensors["Track_4"],
            sensors["Track_5"],
            sensors["Track_6"],
            sensors["Track_7"],
            sensors["Track_8"],
            sensors["Track_9"],
            sensors["Track_10"],
            sensors["Track_11"],
            sensors["Track_12"],
            sensors["Track_13"],
            sensors["Track_14"],
            sensors["Track_15"],
            sensors["Track_16"],
            sensors["Track_17"],
            sensors["RacePosition"]
        ])
        
        # Create batch of predicted future states
        batch_features = []
        
        # Predicted angles at constant speed
        for angle_offset in [-0.1, -0.05, 0.05, 0.1]:
            new_features = features_base.copy()
            new_features[0] = angle + angle_offset  # Adjust angle
            batch_features.append(new_features)
            
        # Batch transform and predict
        if batch_features:
            batch_np = np.array(batch_features)
            batch_transformed = self.scaler.transform(batch_np)
            batch_tensor = torch.FloatTensor(batch_transformed)
            
            # Run batch prediction
            with torch.no_grad():
                if self.model_scripted is not None:
                    batch_predictions = self.model_scripted(batch_tensor).numpy()
                else:
                    batch_predictions = self.model(batch_tensor).numpy()
                    
            # Store predictions in cache
            for i, features in enumerate(batch_features):
                features_hash = self.hash_features(features)
                self.prediction_cache[features_hash] = (batch_predictions[i], current_time)
            
            # Update batch stats
            self.model_inference_count += 1
            
            if self.model_inference_count % 100 == 0:
                print(f"Prefetched {len(batch_features)} predictions")
                
        return len(batch_features)
    
    def update_stuck_status(self):
        """Enhanced stuck detection and recovery with contextual awareness"""
        current_time = time.time()
        speed = self.state.getSpeedX() or 0
        track_pos = self.state.getTrackPos() or 0
        angle = self.state.getAngle() or 0
        track_sensors = self.state.getTrack() or [0] * 19
        
        # Get track edge distances with safety checks
        left_edge = track_sensors[0] if len(track_sensors) > 0 else 0
        right_edge = track_sensors[18] if len(track_sensors) > 18 else 0
        central_sensor = track_sensors[9] if len(track_sensors) > 9 else 0
        
        # Enhanced stuck detection using more context
        is_potentially_stuck = (
            # Standard criteria: low speed and bad position
            (abs(speed) < self.stuck_speed_threshold and abs(track_pos) > 0.7) or
            # Additional criteria: bad angle, very close to obstacle
            (abs(speed) < 20 and abs(angle) > 0.6 and central_sensor < 15) or
            # Almost stopped and facing a wall
            (abs(speed) < 5 and central_sensor < 10)
        )
        
        if is_potentially_stuck:
            self.stuck_timer += current_time - self.last_update_time
            # Additional check: if we're in a turn, give more time before declaring stuck
            if self.segment_type.endswith("_turn") and self.stuck_timer < self.stuck_time_threshold * 1.5:
                self.is_stuck = False
        else:
            self.stuck_timer = 0
            self.is_stuck = False
            self.reverse_start_time = None
            self.reverse_steer = 0.0
        
        # Smart reverse direction calculation when stuck
        if self.stuck_timer > self.stuck_time_threshold:
            self.is_stuck = True
            if self.reverse_start_time is None:
                self.reverse_start_time = current_time
                
                # Enhanced reverse direction calculation using more sensors
                left_side_avg = sum(track_sensors[0:4]) / 4 if len(track_sensors) > 4 else 0
                right_side_avg = sum(track_sensors[15:19]) / 4 if len(track_sensors) > 18 else 0
                
                # Determine best escape direction
                if left_side_avg > right_side_avg * 1.5:
                    # Much more space on left
                    self.reverse_steer = 0.7
                elif right_side_avg > left_side_avg * 1.5:
                    # Much more space on right
                    self.reverse_steer = -0.7
                elif track_pos > 0:
                    # On right side of track
                    self.reverse_steer = -0.5
                else:
                    # On left side of track
                    self.reverse_steer = 0.5
                
                # Adjust based on car angle
                if abs(angle) > 0.5:
                    angle_factor = 0.3 if abs(angle) > 1.0 else 0.15
                    self.reverse_steer = self.reverse_steer * (1 - angle_factor) + (-1 if angle > 0 else 1) * angle_factor
                
                print(f"Initiating smart reverse with steer: {self.reverse_steer:.2f}")
        
        # Enhanced exit criteria for reverse mode
        if self.is_stuck:
            self.reverse_duration = current_time - self.reverse_start_time
            # Dynamic recovery detection
            recovery_speed_threshold = 20 if self.reverse_duration > 1.0 else 15
            recovery_position_threshold = 0.6 if self.reverse_duration > 1.0 else 0.5
            
            if (abs(speed) > recovery_speed_threshold or
                abs(track_pos) < recovery_position_threshold or
                self.reverse_duration > self.max_reverse_duration or
                (central_sensor > 40 and abs(track_pos) < 0.8)):
                
                self.is_stuck = False
                self.stuck_timer = 0
                self.reverse_duration = 0
                self.reverse_start_time = None
                print(f"Recovered from stuck state after {self.reverse_duration:.1f}s")
        
        self.last_update_time = current_time
    
    def drive(self, msg):
        """Advanced driving function with optimized model usage"""
        # Decode the message if it's bytes
        if isinstance(msg, bytes):
            msg = msg.decode()
        self.state.setFromMsg(msg)
        
        # Update stuck status and reverse logic - this runs regardless of AI
        self.update_stuck_status()
        
        # Get sensor data 
        sensors = self.get_sensors()
        
        # Performance optimization: Only run track analysis every few frames
        current_time = time.time()
        should_analyze = self.last_segment_distance == 0 or (
            current_time - self.last_prediction_time > 0.5)
        
        if should_analyze:
            # Analyze track segment for optimized racing strategy
            self.analyze_track_segment()
            
            # Adapt race strategy based on conditions
            self.adapt_race_strategy()
        
        # Prefetch predictions for likely future states
        self.prefetch_predictions()
        
        # Convert sensor data to model input format
        features = np.array([[
            sensors["Angle"],
            sensors["CurrentLapTime"],
            sensors["Damage"],
            sensors["DistanceFromStart"],
            sensors["DistanceCovered"],
            sensors["FuelLevel"],
            sensors["Gear"],
            sensors["LastLapTime"],
            sensors["RPM"],
            sensors["SpeedX"],
            sensors["SpeedY"],
            sensors["SpeedZ"],
            sensors["Track_1"],
            sensors["Track_2"],
            sensors["Track_3"],
            sensors["Track_4"],
            sensors["Track_5"],
            sensors["Track_6"],
            sensors["Track_7"],
            sensors["Track_8"],
            sensors["Track_9"],
            sensors["Track_10"],
            sensors["Track_11"],
            sensors["Track_12"],
            sensors["Track_13"],
            sensors["Track_14"],
            sensors["Track_15"],
            sensors["Track_16"],
            sensors["Track_17"],
            sensors["RacePosition"]
        ]])
        
        # Normalize features with scaler
        features = self.scaler.transform(features)
        
        # Convert to tensor for model input
        features_tensor = torch.FloatTensor(features[0])
        
        # Get optimized model predictions
        steering, acceleration, braking = self.predict_with_model(features_tensor)
        
        # Apply racing expertise to refine model predictions
        steering, acceleration, braking = self.optimize_predicted_output(
            steering, acceleration, braking
        )
        
        if self.is_stuck:
            # Enhanced reverse logic
            self.control.setSteer(self.reverse_steer)
            self.control.setAccel(0.0)  # No acceleration while reversing
            self.control.setBrake(0.0)  # No braking while reversing
            self.control.setGear(-1)    # Set reverse gear
            
            # Only print updates when something changes to reduce console spam
            if current_time - self.last_update_time > 0.5:
                print(f"Reversing with steer: {self.reverse_steer:.2f}, Duration: {self.reverse_duration:.1f}s")
        else:
            # Advanced driving logic
            # Apply adaptive steering smoothing based on segment type
            smoothing_factor = 0.6 if self.segment_type.endswith("_turn") else 0.7
            smoothed_steering = smoothing_factor * steering + (1 - smoothing_factor) * self.prev_steer
            self.prev_steer = smoothed_steering
            
            # Set controls
            self.control.setSteer(smoothed_steering)
            self.control.setAccel(max(0, min(1, acceleration)))
            self.control.setBrake(max(0, min(1, braking)))
            
            # Enhanced gear shifting strategy
            self.advanced_gear_management()
        
        # Throttle logging to reduce overhead
        if self.model_inference_count % 100 == 0:
            speed = self.state.getSpeedX() or 0
            gear = self.state.getGear() or 0
            print(f"AI control: steer={smoothed_steering:.2f}, accel={acceleration:.2f}, brake={braking:.2f}, " +
                  f"speed={speed:.1f}, gear={gear}")
        
        return self.control.toMsg()
    
    def advanced_gear_management(self):
        """Advanced gear management with optimized logic"""
        # Get current state values
        rpm = self.state.getRpm() or 0
        gear = self.state.getGear() or 1
        speed = self.state.getSpeedX() or 0
        brake = self.control.getBrake() or 0
        accel = self.control.getAccel() or 0
        
        # Use lookup table for optimal gear based on speed for quicker decisions
        optimal_gear_for_speed = {
            0: 1,    # 0-20 km/h: 1st gear
            20: 1,   # 0-20 km/h: 1st gear
            40: 2,   # 20-40 km/h: 2nd gear
            80: 3,   # 40-80 km/h: 3rd gear
            120: 4,  # 80-120 km/h: 4th gear
            160: 5,  # 120-160 km/h: 5th gear
            1000: 6  # >160 km/h: 6th gear
        }
        
        if self.is_stuck:
            gear = -1
        else:
            # Heavy braking case
            if brake > 0.5:
                # Quickly find optimal gear based on speed
                target_gear = 1
                for speed_threshold, gear_value in sorted(optimal_gear_for_speed.items()):
                    if speed <= speed_threshold:
                        target_gear = gear_value
                        break
                gear = min(gear, target_gear)
            
            # Light braking case
            elif brake > 0.1:
                gear = max(gear - 1, 1)
                
            # Normal driving
            else:
                # Precomputed RPM thresholds by segment type and gear
                if not hasattr(self, 'rpm_thresholds'):
                    self.rpm_thresholds = {
                        'upshift': {
                            'straight': [6200, 6200, 6200, 6000, 6000, 7000],  # Higher shift points on straights
                            'left_turn': [6000, 6000, 5800, 5800, 5800, 7000],
                            'right_turn': [6000, 6000, 5800, 5800, 5800, 7000],
                            'complex_turn': [5800, 5800, 5600, 5600, 5600, 7000],
                            'unknown': [6000, 6000, 6000, 6000, 6000, 7000]
                        },
                        'downshift': {
                            'straight': [0, 3800, 3800, 4000, 4000, 4000],
                            'left_turn': [0, 4200, 4200, 4500, 4500, 4500],
                            'right_turn': [0, 4200, 4200, 4500, 4500, 4500],
                            'complex_turn': [0, 4500, 4500, 4800, 4800, 4800],
                            'unknown': [0, 4000, 4000, 4000, 4000, 4000]
                        }
                    }
                
                # Get appropriate threshold based on current segment type
                segment = self.segment_type if self.segment_type in self.rpm_thresholds['upshift'] else 'unknown'
                current_upshift_rpm = self.rpm_thresholds['upshift'][segment][min(5, max(0, gear-1))]
                current_downshift_rpm = self.rpm_thresholds['downshift'][segment][min(5, max(0, gear-1))]
                
                # Adjust thresholds based on acceleration request
                if accel > 0.9:
                    current_upshift_rpm += 200  # Delay upshift when accelerating hard
                
                # Gear shifting logic
                if rpm > current_upshift_rpm and speed > 30 and gear < 6:
                    gear += 1
                elif rpm < current_downshift_rpm and speed < 180 and gear > 1:
                    gear -= 1
                    
                # Force forward gear if coming from reverse
                if gear < 1:
                    gear = 1
                
                # Ensure appropriate gear for very high speeds
                if speed > 200 and gear < 5:
                    gear = max(gear, 5)
        
        # Set gear with minimal logging to reduce overhead
        current_time = time.time()
        if gear != self.control.getGear() or current_time - self.last_update_time > 1.0:
            self.control.setGear(gear)
            
            # Reduce logging frequency
            if current_time - self.last_update_time > 1.0:
                print(f"Gear: {gear}, Speed: {speed:.1f} km/h, RPM: {rpm:.0f}, Segment: {self.segment_type}")
                self.last_update_time = current_time
        else:
            # Just set the gear without logging
            self.control.setGear(gear)
        
        self.prev_rpm = rpm
    
    def steer(self):
        """Advanced steering logic with optimized racing line and curvature-aware adjustments"""
        angle = self.state.angle if self.state.angle is not None else 0.0
        track_pos = self.state.trackPos if self.state.trackPos is not None else 0.0
        speed = self.state.getSpeedX() or 0
        track = self.state.getTrack() or [0] * 19
        
        # Use racing line optimizer to get optimal positioning
        if SCIPY_AVAILABLE:
            racing_line_data = self.racing_line_optimizer.optimize_racing_line(
                track, angle, speed, track_pos, self.segment_type, 
                self.state.getDistFromStart() or 0
            )
            target_pos = racing_line_data['target_position']
            curvature = racing_line_data['curvature']
        else:
            # Fallback to standard target position
            target_pos = self.racing_line_offset
            curvature = 0.0
            if len(self.track_history) > 10:
                # Basic curvature estimate
                left_readings = [t[0] for t in self.track_history[-5:]]
                right_readings = [t[18] for t in self.track_history[-5:]]
                left_diff = left_readings[-1] - left_readings[0]
                right_diff = right_readings[-1] - right_readings[0]
                curvature = (right_diff - left_diff) / 5.0
        
        # Calculate position error
        position_error = track_pos - target_pos
        
        # Advanced steering computations using curvature and velocity
        if SCIPY_AVAILABLE:
            # Compute look-ahead distance based on speed
            # Higher speeds need earlier steering input
            look_ahead = min(1.0, speed / 100) * 1.5
            
            # Calculate dynamic weights with non-linear scaling
            angle_weight = 1.0 + (0.2 * abs(curvature) * min(1.0, speed / 100))
            
            # Position weight increases with speed but decreases in sharp turns
            position_weight = 0.1 + (0.15 * min(1.0, speed / 120)) * (1.0 - min(1.0, abs(curvature) * 3))
            
            # Apply predictive steering based on future curvature
            # Add a component to anticipate how angle will develop
            predictive_component = 0.0
            if abs(curvature) > 0.05:
                # Add anticipation component when approaching or in turns
                predictive_component = curvature * look_ahead * 0.5
                
            # Calculate steering with all components
            steer = (angle * angle_weight - position_error * position_weight + predictive_component) / self.steer_lock
            
            # Non-linear scaling for extreme values
            if abs(steer) > 0.5:
                steer = 0.5 * np.sign(steer) + 0.5 * np.tanh(steer - 0.5 * np.sign(steer))
        else:
            # Standard steering calculation for non-scipy environments
            # Dynamic steering weights based on speed and segment
            angle_weight = 1.0
            position_weight = 0.05 + (0.1 * min(1.0, speed / 100))
            
            # Adjust weights based on segment type
            if self.segment_type == "straight":
                position_weight *= 1.5  # Stronger correction on straights
            elif self.segment_type.endswith("_turn"):
                angle_weight *= 1.2    # Focus more on angle in turns
                
            # Calculate steering with enhanced weights
            steer = (angle * angle_weight - position_error * position_weight) / self.steer_lock
        
        # Dynamic steering limits based on speed and segment
        max_steer = 0.3
        if speed > 100:
            max_steer = 0.2
        if speed > 150:
            max_steer = 0.15
        if speed > 200:
            max_steer = 0.1
            
        # Allow more steering in complex turns at lower speeds
        if self.segment_type == "complex_turn" and speed < 80:
            max_steer *= 1.2
            
        # Apply steering limit
        steer = max(min(steer, max_steer), -max_steer)
        
        # Adaptive smoothing based on segment type
        smoothing = 0.8  # Default smoothing factor
        if self.segment_type == "straight":
            smoothing = 0.85  # More smoothing on straights
        elif self.segment_type == "complex_turn":
            smoothing = 0.7   # Less smoothing in complex turns
            
        # Apply smoothing
        current_steer = self.control.getSteer()
        smoothed_steer = smoothing * current_steer + (1 - smoothing) * steer
        
        # Set the steering
        self.control.setSteer(smoothed_steer)
        
        return smoothed_steer

def get_manual_inputs():
    """Get manual acceleration and braking inputs"""
    accel = 0.0
    brake = 0.0
    if keyboard.is_pressed('w'):
        accel = 1.0
    if keyboard.is_pressed('s'):
        brake = 1.0
    return accel, brake

if __name__ == "__main__":
    # Example usage for testing
    d = Driver(stage=3)
    print(d.init())