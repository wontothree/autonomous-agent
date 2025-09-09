from self_drive_sim.agent.interfaces import Observation, Info, MapInfo

import math
import numpy as np
import itertools
from dataclasses import dataclass, field
from scipy.ndimage import distance_transform_edt, zoom

class Map:
    ORIGINAL_STRING_MAP0 = """
1111111111111111111111111111111111111111
1111111111111111111111111111111111111111
1100000000000000000000001000000000000011
1100000000000000000000000000000000000011
1100000000000000000000000000000000000011
1100000000000000000000000000000000000011
1100000000000000000000000000000000000011
1100000000000000000000000000000000000011
1100000000000000000000000000000000000011
1100000000000000000000000000000000000011
1100000000000000000000000000000000000011
1100000000000000000000000000000000000011
1100000000000000000000001000000000000011
1111111111111111111111111000000000000011
1100000000000000000000000000000000000011
1100000000000000000000000000000000000011
1100000000000000000000000000000000000011
1100000000000000000000000000000000000011
1100000000000000000000000000000000000011
1100000000000000000000000000000000000011
1100000000000000000000000000000000000011
1100000000000000000000000000000000000011
1100000000000000000000000000000000000011
1100000000000000000000000000000000000011
1100000000000000000000000000000000000011
1100000000000000000000000000000000000011
1100000000000000000000000000000000000011
1100000000000000000000000000000000000011
1111111111111111111111111111111111111111
1111111111111111111111111111111111111111
"""

    ORIGINAL_STRING_MAP1 = """
11111111111111111111111111111111111111111111111111
11111111111111111111111111111111111111111111111111
11111100001000000000001111111000000000000000000011
11111100001000000000001111111000000000000000000011
11111100001000000000001111111000000000000000000011
11111100001000000000000001000000000000000000000011
11111100001000000000000001000000000000000000000011
11111100001000000000000001000000000000000000000011
11110000001000000000000001000000000000000000000011
11110000000000000000000001000000000000000000011111
11110000000000000000000001000000000000000000011111
11110000000000000000000001000000000000000000011111
11110000000000000000000001000000000000000000011111
11110000001111111111111111000000000000000000011111
11110000000000000000000001000000000000000000011111
11110000000000000000000000000000000000000000011111
11110000000000000000000000000000000000000000011111
11110000000000000000000000000000000000000000011111
11110000000000000000000000000000000000000000011111
11111000000000000000000001111111111111111111111111
11000000000000000000000000000000000001111111111111
11000000000000000000000000000000000001111111111111
11000000000000000000000000000000000001111111111111
11000000000000000000000000000000000000000000000011
11000000000000000000000001000000000000000000000011
11000000000000000000000001000000000000000000000011
11000000000000000000000001000000000000000000000011
11000000000000000000000001000000000000000000000011
11000000000000000000000001000000000000000000000011
11000000000000000000000001000000000000000000000011
11000000000000000000000001000000000000000000000011
11000000000000000000000001000000000000000000000011
11000000000000000000000001111000000000000000000011
11000000000000000000000001111000000000000000000011
11000000000000000000000001111000000000000000000011
11000000000000000000000001111000000000000000000011
11000000000000000000000001111000000000000000000011
11000000000000000000000001111000000000000000000011
11000000000000000000000001111000000000000000000011
11000000000000000000000001111000000000000000000011
11000000000000000000000001111000000000000000000011
11111111111111110000000011111111111111111111111111
11000000000000000000000000000000000000000000011111
11000000000000000000000000000000000000000000011111
11000000000000000000000000000000000000000000011111
11000000000000000000000000000000000000000000011111
11110000000000000000000000000000000000000000011111
11110000000000000000000000000000000000000000011111
11111111111111111111111111111111111111111111111111
11111111111111111111111111111111111111111111111111
"""

    ORIGINAL_STRING_MAP2 = """
111111111111111111111111111111111111111111111111111111111000000000000000000
111111111111111111111111111111111111111111111111111111111000000000000000000
110000000000000000000000111110000001000000000000000000011000000000000000000
110000000000000000000000111110000001000000000000000000011000000000000000000
110000000000000000000000111110000001000000000000000000011000000000000000000
110000000000000000000000111110000001000000000000000000011000000000000000000
110000000000000000000000111110000000000000000000000000011000000000000000000
110000000000000000000000111110000000000000000000000000011000000000000000000
110000000000000000000000111110000000000000000000000000011000000000000000000
110000000000000000000000111110000000000000000000000000011000000000000000000
110000000000000000000000000010000001000000000000000000011000000000000000000
110000000000000000000000000000000001000000000000000000011000000000000000000
110000000000000000000000000000000001000000000000000000011111111111111111111
110000000000000000000000000000000001000000000000000000011111111111111111111
110000000000000000000000000000000001111111111111111111111111111111111111111
110000000000000000000000000010000000000000000000011111111111111111111111111
111111111111111111111111111110000000000000000000011111111111111111111111111
111111111111000000000000000010000000000000000000011111111111111111111111111
111111111111000000000000000000000000000000000000011111111111111111111111111
111111111111000000000000000000000000000000000000000000000000000000000011111
111111111111000000000000000000000000000000000000000000000000000000000011111
111111111111000000000000000000000000000000000000000000000000000000000011111
111111111111000000000000000010000000000000000000000000000000000000000011111
111000000001000000000000000010000000000000000000000000000000000000000011111
111000000001000000000000000010000000000000000000000000000000000000000011111
111000000001000000000000000010000000000000000000000000000000000000000011111
111000000001000000000000000010000000000000000000000000000000000000000011111
111000000001000000000000000010000000000000000000000000000000000000000011111
111000000001000000000000000010000000000000000000000000000000000000000011111
111000000001000000000000000010000000000000000000000000000000000000000011111
111000000001000000000000000010000000000000000000011111111111111111111111111
111000000001000000000000000010000000000000000000011111111111111111111111111
111000000001000000000000000010000000000000000000011111111111111111111111111
111000000001000000000000000010000000000000000000011111111111111111111111111
111110000111111111111111111110000000000000000000000000000000000000000001111
111100000000000000000000000010000000000000000000000000000000000000000001111
111100000000000000000000000000000000000000000000000000000000000000000001111
111100000000000000000000000000000000000000000000000000000000000000000001111
111100000000000000000000000000000000000000000000000000000000000000000001111
111100000000000000000000000000000000000000000000000000000000000000000001111
111100000000000000000000000010000000000000000000000000000000000000000001111
111100000000000000000000000010000000000000000000000000000000000000000001111
111100000000000000000000000010000000000000000000000000000000000000000001111
111100000000000000000000000010000000000000000000000000000000000000000001111
111100000000000000000000000010000000000000000000000000000000000000000000011
111100000000000000000000000010000000000000000000000000000000000000000000011
111100000000000000000000000010000000000000000000000000000000000000000000011
111100000000000000000000000010000000000000000000000000000000000000000000011
111100000000000000000000000010000000000000000000000000000000000000000000011
111100000000000000000000000010000000000000000000000000000000000000000000011
111111111111111111111111111110000000000000000000000000000000000000000000011
111111111111111111111111111110000000000000000000000000000000000000000000011
000000000000000000000000000110000000000000000000000000000000000000000000011
000000000000000000000000000110000000000000000000000000000000000000000000011
000000000000000000000000000110000000000000000000000000000000000000000000011
000000000000000000000000000110000000000000000000000000000000000000000000011
000000000000000000000000000110000000000000000000000000000000000000000000011
000000000000000000000000000110000000000000000000000000000000000000000000011
000000000000000000000000000110000000000000000000000000000000000000000000011
000000000000000000000000000110000000000000000000000000000000000000000000011
000000000000000000000000000110000000000000000000000000000000000000000000011
000000000000000000000000000110000000000000000000000000000000000000000000011
000000000000000000000000000110000000000000000000000000000000000000000000011
000000000000000000000000000110000000000000000000000000000000000000000000011
000000000000000000000000000110000000000000000000000000000000000000000000011
000000000000000000000000000111111100000000111111111111110000000011111111111
000000000000000000000000000111100000000000000000000000000000000000000001111
000000000000000000000000000111100000000000000000000000000000000000000001111
000000000000000000000000000110000000000000000000000000000000000000000000011
000000000000000000000000000110000000000000000000000000000000000000000000011
000000000000000000000000000110000000000000000000000000000000000000000000011
000000000000000000000000000111100000000000000000000000000000000000000001111
000000000000000000000000000111100000000000000000000000000000000000000001111
000000000000000000000000000111111111111111111111111111111111111111111111111
000000000000000000000000000111111111111111111111111111111111111111111111111
"""

    ORIGINAL_STRING_MAP3 = """
1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111
1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111
1100000000000000000000000000000000111111111111111100000000111100000000000000001111111100000000000011
1100000000000000000000000000000000000000000000000000000000111100000000000000001111111100000000000011
1100000000000000000000000000000000000000000000000000000000111100000000000000001111111100000000000011
1100000000000000000000000000000000000000000000000000000000111100000000000000001111111100000000000011
1100000001000000000100000000000000000000000000000000000000111100000000000000001111111100000000000011
1100000001000000000100000000000000000000000000000000000000111100000000000000001111111100000000000011
1100000001111111111111111111111111111111111111111100000000111100000000000000001111111100000000000011
1100000001111111111111111111111111111111111111111100000000111100000000000000001111111100000000000011
1100000001111111111111111111111111111111111111111100000000111111111111110000111111111100000000000011
1100000000000100000000000000000011111111111111111000000000000000000000000000000000000100000000000011
1100000000000000000000000000000011111111111111111000000000000000000000000000000000000100000000000011
1100000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000011
1100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000011
1100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000011
1100000000000100000000111110000000000000000000000000000000000000000000000000000000000000000000000011
1111111111111100000000111110000000000000000000000000000000111111111111111111100000000000000000000011
1100000000000100000000111110000000000000000000000000000000100000000000000000100000000100000000000011
1100000000000100000000111110000000000000000000000000000000100000000000000000100000000100000000000011
1100000000000100000000111110000000000000000000000000000000000000000000000000000000000100000000000011
1100000000000100000000111110000000000000000000000000000000000000000000000000000000000100000000000011
1100000000000100000000111110000000000000000000000000000000000000000000000000000000000100000000000011
1100000000000100000000111110000000000000000000000000000000000000000000000000000000000100000000000011
1100000000000100000000111110000000000000000000000000000000100000000000000000100000000100000000000011
1100000000000100000000111110000000000000000000000000000000100000000000000000100000000100000000000011
1100000000000100000000111110000000000000000000000000000000100000000000000000100000000100000000000011
1100000000000100000000111110000000000000000000111100000011111111111111111111100000000111111111111111
1100000000000100000000111111111111111111111111111100000010000000000000000000100000000100000000000011
1100000000000100000000111111111111111111111111111100000010000000000000000000000000000000000000000011
1100000000000100000000111111111111111111111111111100000010000000000000000000000000000000000000000011
1100000000000100000000111111111111111111111111111100000010000000000000000000000000000000000000000011
1100000000000100000000100000000000000000000000000100000010000000000000000000000000000000000000000011
1100000000000000000000000000000000000000000000000100000010000000000000000000100000000100000000000011
1100000000000000000000000000000000000000000000000100000010000000000000000000100000000100000000000011
1100000000000000000000000000000000000000000000000100000010000000000000000000100000000100000000000011
1100000000000000000000000000000000000000000000000100000010000000000000000000100000000100000000000011
1100000000000100000000100000000000000000000000000100000010000000000000000000100000000100000000000011
1100000000000100000000100000000000000000000000000100000010000000000000000000100000000100000000000011
1100000000000100000000100000000000000000000000000100000010000000000000000000100000000100000000000011
1100000000000100000000100000000000000000000000000100000010000000000000000000100000000100000000000011
1100000000000100000000100000000000000000000000000100000010000000000000000000100000000100000000000011
1111111111111100000000100000000000000000000000000100000010000000000000000000100000000100000000000011
1100000000000100000000100000000000000000000000000100000010000000000000000000100000000100000000000011
1100000000000100000000100000000000000000000000000100000010000000000000000000100000000100000000000011
1100000000000100000000100000000000000000000000000100000010000000000000000000100000000100000000000011
1100000000000100000000100000000000000000000000000100000010000000000000000000100000000100000000000011
1100000000000100000000111111111111111111111100001100000011000011111111111111100000000100000000000011
1100000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000011
1100000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000011
1100000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000011
1100000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000011
1100000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000011
1100000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000011
1100000000000100000000100000000000000000100000000000000000100000000000000000100000000100000000000011
1100000000000100000000100000000000000000100000000000000000100000000000000000100000000100000000000011
1100000000000100000000100000000000000000100000000000000000100000000000000000100000000100000000000011
1100000000000100000000100000000000000000100000000000000000100000000000000000100000000100000000000011
1100000000000100000000100000000000000000100000000000000000100000000000000000100000000100000000000011
1111111111111100000000100000000000000000100000000000000000100000000000000000100000000100000000000011
1100000000000100000000100000000000000000100000000000000000100000000000000000100000000100000000000011
1100000000000100000000100000000000000000100000000000000000100000000000000000100000000100000000000011
1100000000000100000000100000000000000000100000000000000000100000000000000000100000000100000000000011
1100000000000100000000100000000000000000100000000000000000100000000000000000100000000100000000000011
1100000000000100000000100000000000000000100000000000000000100000000000000000100000000100000000000011
1100000000000100000000100000000000000000100000000000000000100000000000000000100000000100000000000011
1100000000000000000000100000000000000000100000000000000000100000000000000000100000000100000000000011
1100000000000000000000100000000000000000100000000000000000100000000000000000100000000100000000000011
1100000000000000000000100000000000000000100000000000000000100000000000000000100000000100000000000011
1100000000000000000000100000000000000000100000000000000000100000000000000000100000000100000000000011
1100000000000000000000100000000000000000100000000000000000100000000000000000100000000100000000000011
1100000000000000000000100000000000000000100000000000000000100000000000000000100000000100000000000011
1100000000000100000000100000000000000000100000000000000000100000000000000000100000000000000000000011
1100000000000100000000100000000000000000100000000000000000100000000000000000100000000000000000000011
1100000000000100000000100000000000000000100000000000000000100000000000000000100000000000000000000011
1100000000000100000000100000000000000000100000000000000000100000000000000000100000000000000000000011
1100000000000100000000100000000000000000100000000000000000100000000000000000100000000100000000000011
1100000000000100000000100000000000000000100000000000000000100000000000000000100000000100000000000011
1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111
1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111
"""

    def string_to_np_array_map(self, string_map: str) -> np.ndarray:
        return np.array([list(map(int, line)) 
                        for line in string_map.strip().splitlines() 
                        if line.strip()], dtype=np.int8)
        
    def np_array_to_string_map(self, array_map: np.ndarray) -> str:
        lines = ["".join(map(str, row)) for row in array_map]
        return "\n".join(lines)

    def occupancy_grid_to_distance_map(self, occupancy_grid_map: np.ndarray, map_resolution: float) -> np.ndarray:
        dist_map = distance_transform_edt(occupancy_grid_map == 0)
        dist_map = dist_map * map_resolution
        dist_map = np.round(dist_map, 2) 
        return dist_map

    def upscale_occupancy_grid_map(self, occupancy_grid_map: np.ndarray, scale: int) -> np.ndarray:
        if scale == 1:
            return occupancy_grid_map.copy()
        
        upscaled_map = zoom(occupancy_grid_map, zoom=scale, order=0) 
        return upscaled_map

@dataclass
class Pose:
    _x: float = 0.0
    _y: float = 0.0
    _yaw: float = 0.0

    def __post_init__(self):
        self.yaw = self._yaw

    def normalize_yaw(self):
        self._yaw = math.atan2(math.sin(self._yaw), math.cos(self._yaw))

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = value

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        self._y = value

    @property
    def yaw(self):
        return self._yaw

    @yaw.setter
    def yaw(self, value):
        self._yaw = value
        self.normalize_yaw()

    def update(self, x: float, y: float, yaw: float):
        self.x = x 
        self.y = y 
        self.yaw = yaw

@dataclass
class Particle:
    pose: Pose = field(default_factory=Pose)
    weight: float = 0.0

class MonteCarloLocalizer:
    def __init__(self,
            particle_num=100,
            initial_pose_noise=[0.02, 0.02, 0.02],
            odom_noise=[0.01, 0.002, 0.002, 0.01],

            # Weight
            scan_min_angle=-1.9,
            scan_angle_increment=0.016,
            scan_min_range=0.05,
            scan_max_range=8.0,
            scan_step=3,
            sigma_hit=0.2,
            z_hit=0.95,
            z_rand=0.05,

            # Resamping
            omega_slow=0.0,
            omega_fast=0.0,
            alpha_slow=0.001,
            alpha_fast=0.9,
            resample_ess_ratio=0.5,
            ):
        
        # Constants
        self.particle_num = particle_num
        self.initial_pose_noise = initial_pose_noise
        self.odom_noise = odom_noise
        self.scan_min_angle = scan_min_angle
        self.scan_angle_increment = scan_angle_increment
        self.scan_min_range = scan_min_range
        self.scan_max_range = scan_max_range
        self.scan_step = scan_step
        self.sigma_hit = sigma_hit
        self.z_hit = z_hit
        self.z_rand = z_rand
        self.omega_slow = omega_slow
        self.omega_fast = omega_fast
        self.alpha_slow = alpha_slow
        self.alpha_fast = alpha_fast
        self.resample_ess_ratio = resample_ess_ratio

        # Member variables
        self.particles = []   

    def initialize_particles(self, 
            initial_pose: Pose
        ):
        """
        Parameters
        ----------
        - initial_pose: initial pose of robot (x, y, yaw)
        
        Update
        ------
        - self.particles: initialized particles

        Using
        -----
        - self.particle_num
        - self.initial_pose_noise
        """
        particle_xs = initial_pose.x + np.random.normal(0, self.initial_pose_noise[0], self.particle_num)
        particle_ys = initial_pose.y + np.random.normal(0, self.initial_pose_noise[1], self.particle_num)
        particle_yaws = initial_pose.yaw + np.random.normal(0, self.initial_pose_noise[2], self.particle_num)
        
        initial_weight = 1.0 / self.particle_num
    
        self.particles = [
            Particle(
                pose=Pose(particle_xs[i], particle_ys[i], particle_yaws[i]),
                weight=initial_weight
            )
            for i in range(self.particle_num)
        ]
    
    def update_particles_by_motion_model(self, 
            delta_distance: float, 
            delta_yaw: float
        ):
        """
        Parameters
        ----------
        - delta_distance: measured by IMU
        - delta_yaw
        
        Update
        ------
        - self.particles:

        Using
        -----
        - self.odom_noise
        """
        # Standard deviation (noise)
        squared_delta_distance = delta_distance * delta_distance
        squared_delta_yaw = delta_yaw * delta_yaw
        std_dev_distance = math.sqrt(self.odom_noise[0] * squared_delta_distance + self.odom_noise[1] * squared_delta_yaw)
        std_dev_yaw = math.sqrt(self.odom_noise[2] * squared_delta_distance + self.odom_noise[3] * squared_delta_yaw)
        for particle in self.particles:
            # Differential drive model
            noisy_delta_distance = delta_distance + np.random.normal(0, std_dev_distance)
            noisy_delta_yaw = delta_yaw + np.random.normal(0, std_dev_yaw)
            yaw = particle.pose.yaw
            t = yaw + noisy_delta_yaw / 2.0
            
            updated_x = particle.pose.x + noisy_delta_distance * math.cos(t)
            updated_y = particle.pose.y + noisy_delta_distance * math.sin(t)
            updated_yaw = yaw + noisy_delta_yaw
						
			# update each particle	
            particle.pose.update(updated_x, updated_y, updated_yaw)

    def update_weights_by_measurement_model(
        self,
        scan_ranges: np.ndarray,
        occupancy_grid_map: np.ndarray,
        distance_map: np.ndarray,
        map_origin=(float, float),
        map_resolution= float,
        ):
        """
        Parameters
        ----------
        - scan
        - occupancy_grid_map
        - distance_map

        Update
        ------
        - self.particles

        Using
        -----
        - self.particle_num
        - self.scan_min_angle
        - self.scan_angle_increment
        - self.scan_min_range
        - self.scan_max_range
        - self.scan_step
        - self.sigma_hit
        - self.z_hit
        - self.z_rand
        """   

        eps = 1e-12

        beam_num = len(scan_ranges)
        sampled_beam_indices = np.arange(0, beam_num, self.scan_step)
        sampled_beam_angles = self.scan_min_angle + sampled_beam_indices * self.scan_angle_increment
        sampled_scan_ranges = scan_ranges[sampled_beam_indices]
        
        denominator = 2.0 * (self.sigma_hit ** 2)
        inv_denominator = 1.0 / denominator
        
        # Map constant
        map_height, map_width = occupancy_grid_map.shape
        map_origin_x, map_origin_y = map_origin
        
        # Initialize particle weight
        log_weights = np.zeros(self.particle_num, dtype=np.float64)
        
        # previous
        particle_yaws = np.array([particle.pose.yaw for particle in self.particles])
        particle_cos_yaws = np.cos(particle_yaws)
        particle_sin_yaws = np.sin(particle_yaws)
        
        particle_xs = np.array([particle.pose.x for particle in self.particles])
        particle_ys = np.array([particle.pose.y for particle in self.particles])
        
        # Previous calculation cos and sine for beam angle
        cos_sampled_beam_angles = np.cos(sampled_beam_angles)
        sin_sampled_beam_angles = np.sin(sampled_beam_angles)
        
        for particle_index in range(self.particle_num):
            particle_x = particle_xs[particle_index]
            particle_y = particle_ys[particle_index]
            cos_yaw = particle_cos_yaws[particle_index]
            sin_yaw = particle_sin_yaws[particle_index]
            
            log_likelihood = 0.0
            
            for beam_index in range(len(sampled_scan_ranges)):
                range_measurement = sampled_scan_ranges[beam_index]
                
                # Check validity of range
                if not (self.scan_min_range < range_measurement < self.scan_max_range) or np.isinf(range_measurement) or np.isnan(range_measurement):
                    log_likelihood += math.log(self.z_rand + eps)
                    continue
                
                # LiDAR endpoint
                direction_x = cos_yaw * cos_sampled_beam_angles[beam_index] - sin_yaw * sin_sampled_beam_angles[beam_index]
                direction_y = sin_yaw * cos_sampled_beam_angles[beam_index] + cos_yaw * sin_sampled_beam_angles[beam_index]
                
                lidar_hit_x = particle_x + range_measurement * direction_x
                lidar_hit_y = particle_y + range_measurement * direction_y
                
                # Index
                map_index_x = int(round((lidar_hit_x - map_origin_x) / map_resolution))
                map_index_y = int(round((lidar_hit_y - map_origin_y) / map_resolution))
                
                if 0 <= map_index_x < map_width and 0 <= map_index_y < map_height:
                    distance_in_cells = distance_map[map_index_y, map_index_x]
                    # distance_in_meters = float(distance_in_cells) * map_resolution
                    distance_in_meters = float(distance_in_cells)
                    
                    prob_hit = math.exp( -(distance_in_meters ** 2) * inv_denominator)
                    total_prob = self.z_hit * prob_hit + self.z_rand * (1.0 / self.scan_max_range)
                    
                    log_likelihood += math.log(total_prob + eps)
                else:
                    log_likelihood += math.log(self.z_rand + eps)
        
            log_weights[particle_index] = log_likelihood
            
        # Normalize log-sum-exp
        max_log_weight = np.max(log_weights)
        exp_weights = np.exp(log_weights - max_log_weight)
        normalized_weights = exp_weights / (np.sum(exp_weights) + eps)
        
        # allocate weight to particle
        for index, particle in enumerate(self.particles):
            particle.weight = max(normalized_weights[index], eps)
        
        total_weight = sum(particle.weight for particle in self.particles)
        if total_weight > 0:
            for particle in self.particles:
                particle.weight /= total_weight

    def estimate_robot_pose(self):
        """
        Returns
        -------
        - estimated_x
        - estimated_y
        - estimated_yaw

        Using
        ------
        - self.particles
        """        
        xs = np.array([particle.pose.x for particle in self.particles])
        ys = np.array([particle.pose.y for particle in self.particles])
        yaws = np.array([particle.pose.yaw for particle in self.particles])
        weights = np.array([particle.weight for particle in self.particles])
        
        # Weighted sum for x and y
        estimated_x = np.sum(xs * weights)
        estimated_y = np.sum(ys * weights)
        
        # Weighted sum for yaw
        cos_yaw = np.sum(np.cos(yaws) * weights)
        sin_yaw = np.sum(np.sin(yaws) * weights)
        estimated_yaw = math.atan2(sin_yaw, cos_yaw)
        
        return estimated_x, estimated_y, estimated_yaw
    
    def resample_particles(self):
        """
        Update
        ------
        - self.particles

        Using
        -----
        - self.particle_num
        - self.omega_slow
        - self.omega_fast
        - self.alpha_slow
        - self.alpha_slow
        - self.resample_ess_ratio
        """
        def calculate_amcl_random_particle_rate(average_likelihood):
            self.omega_slow += self.alpha_slow * (average_likelihood - self.omega_slow)
            self.omega_fast += self.alpha_fast * (average_likelihood - self.omega_fast)
            amcl_random_particle_rate = 1.0 - self.omega_fast / self.omega_slow
            return max(amcl_random_particle_rate, 0.0)

        # 수정해야 함. 변수명. sum 바꿔야 함
        def calculate_effective_sample_size():
            weights = np.array([particle.weight for particle in self.particles])
            total_likelihood = np.sum(weights)

            sum = 0.0
            for index in range(self.particle_num):
                weight = self.particles[index].weight / total_likelihood
                self.particles[index].weight = weight
                sum += weight * weight
            effective_sample_size = 1.0 / sum
            return effective_sample_size
        
        # Inspect effective sample size
        threshold = self.particle_num * self.resample_ess_ratio
        effective_sample_size = calculate_effective_sample_size()
        if effective_sample_size > threshold: return
        # --- 추가

        weights = np.array([particle.weight for particle in self.particles])
        weights /= np.sum(weights)  # normalize

        positions = (np.arange(self.particle_num) + np.random.uniform()) / self.particle_num
        cumulative_sum = np.cumsum(weights)
        indices = np.searchsorted(cumulative_sum, positions)

        # Resampled particles
        self.particles = [
            Particle(pose=Pose(
                _x=self.particles[i].pose.x,
                _y=self.particles[i].pose.y,
                _yaw=self.particles[i].pose.yaw),
                weight=1.0 / self.particle_num
            )
            for i in indices
        ]



class Agent:
    def __init__(self, logger):
        self.logger = logger
        self.steps = 0

        self.current_robot_pose = (0, 0, 0)
        self.true_robot_pose = (0, 0, 0)
        self.distance_error = 0

        # Map
        self.map_id = None
        self.room_num = None
        self.map_origin = None
        self.resolution = 0.01

        # Finite state machine
        self.current_fsm_state = "READY"
        self.waypoints = []
        self.current_waypoint_index = 0
        self.tmp_target_position = None
        self.optimal_next_node_index = None
        self.current_node_index = None
        self.visited_node_indices = []

        # Particle filter
        self.particle_filter = MonteCarloLocalizer()
        self.map_obj = Map()
        self.occupancy_grid_map = None
        self.distance_map = None

        # Controller
        self.last_angle_error = 0.0
        self.integral_angle_error = 0.0
        self.last_distance_error = 0.0
        self.integral_distance_error = 0.0

        # test
        self.target_position = None
        self.current_waypoint_index = 0
        self.tmp_start_node = 5
        self.tmp_end_node = 1
        self.max_localization_error = 0

    def initialize_map(self, map_info: MapInfo):
        """
        매핑 정보를 전달받는 함수
        시뮬레이션 (episode) 시작 시점에 호출됨

        MapInfo: 매핑 정보를 저장한 클래스
        ---
        height: int, 맵의 높이 (격자 크기 단위)
        width: int, 맵의 너비 (격자 크기 단위)
        wall_grid: np.ndarray, 각 칸이 벽인지 여부를 저장하는 2차원 배열
        room_grid: np.ndarray, 각 칸이 몇 번 방에 해당하는지 저장하는 2차원 배열
        num_rooms: int, 방의 개수
        grid_size: float, 격자 크기 (m)
        grid_origin: (float, float), 실제 world의 origin이 wall_grid 격자의 어디에 해당하는지 표시
        station_pos: (float, float), 청정 완료 후 복귀해야 하는 도크의 world 좌표
        room_names: list of str, 각 방의 이름
        pollution_end_time: float, 마지막 오염원 활동이 끝나는 시각
        starting_pos: (float, float), 시뮬레이션 시작 시 로봇의 world 좌표
        starting_angle: float, 시뮬레이션 시작 시 로봇의 각도 (x축 기준, 반시계 방향)

        is_wall: (x, y) -> bool, 해당 좌표가 벽인지 여부를 반환하는 함수
        get_room_id: (x, y) -> int, 해당 좌표가 속한 방의 ID를 반환하는 함수 (방에 속하지 않으면 -1 반환)
        get_cells_in_room: (room_id) -> list of (x, y), 해당 방에 속한 모든 격자의 좌표를 반환하는 함수
        grid2pos: (grid) -> (x, y), 격자 좌표를 실제 world 좌표로 변환하는 함수
        pos2grid: (pos) -> (grid_x, grid_y), 실제 world 좌표를 격자 좌표로 변환하는 함수
        ---
        """
        self.pollution_end_time = map_info.pollution_end_time

        # Initialize robot pose
        initial_robot_position = map_info.starting_pos
        initial_robot_yaw = map_info.starting_angle
        self.current_robot_pose = (initial_robot_position[0], initial_robot_position[1], initial_robot_yaw)

        # Identify map
        if map_info.num_rooms == 2: 
            self.map_id = 0
            self.room_num = 2
            self.map_origin = (14, 20)
            map = self.map_obj.ORIGINAL_STRING_MAP0
            self.current_node_index = 2
        elif map_info.num_rooms == 5: 
            self.map_id = 1
            self.room_num = 5
            self.map_origin = (25, 25)
            map = self.map_obj.ORIGINAL_STRING_MAP1
            self.current_node_index = 5
        elif map_info.num_rooms == 8:
            self.map_id = 2
            self.room_num = 8
            self.map_origin = (37, 37)
            map = self.map_obj.ORIGINAL_STRING_MAP2
            self.current_node_index = 8
        elif map_info.num_rooms == 13:
            self.map_id = 3
            self.room_num = 13
            self.map_origin = (40, 50)
            map = self.map_obj.ORIGINAL_STRING_MAP3
            self.current_node_index = 13

        self.log(self.room_num)

        # Finite state machine
        self.current_fsm_state = "READY"

        # Particle filter
        initial_pose = Pose(_x=self.current_robot_pose[0],
                            _y=self.current_robot_pose[1],
                            _yaw=self.current_robot_pose[2])
        self.particle_filter.initialize_particles(initial_pose)
        # Occupancy grid map and distance map
        original_map = self.map_obj.string_to_np_array_map(map)
        self.occupancy_grid_map = self.map_obj.upscale_occupancy_grid_map(original_map, 0.2 / self.resolution)
        self.distance_map = self.map_obj.occupancy_grid_to_distance_map(self.occupancy_grid_map, map_resolution=self.resolution)

    def act(self, observation: Observation):
        """
        env로부터 Observation을 전달받아 action을 반환하는 함수
        매 step마다 호출됨

        Observation: 로봇이 센서로 감지하는 정보를 저장한 dict
        ---
        sensor_lidar_front: np.ndarray, 전방 라이다 (241 x 1)
        sensor_lidar_back: np.ndarray, 후방 라이다 (241 x 1)
        sensor_tof_left: np.ndarray, 좌측 multi-tof (8 x 8)
        sensor_tof_right: np.ndarray, 우측 multi-tof (8 x 8)
        sensor_camera: np.ndarray, 전방 카메라 (480 x 640 x 3)
        sensor_ray: float, 상향 1D 라이다
        sensor_pollution: float, 로봇 내장 오염도 센서
        air_sensor_pollution: np.ndarray, 거치형 오염도 센서 (방 개수 x 1)
        disp_position: (float, float), 이번 step의 로봇의 위치 변위 (오차 포함)
        disp_angle: float, 이번 step의 로봇의 각도 변위 (오차 포함)
        ---

        action: (MODE, LINEAR, ANGULAR)
        MODE가 0인 경우: 이동 명령, Twist 컨트롤로 선속도(LINEAR) 및 각속도(ANGULAR) 조절. 최대값 1m/s, 2rad/s
        MODE가 1인 경우: 청정 명령, 제자리에서 공기를 청정. LINEAR, ANGULAR는 무시됨. 청정 명령을 유지한 후 1초가 지나야 실제로 청정이 시작됨
        """

        # --------------------------------------------------
        # Observation
        # pollution data
        air_sensor_pollution_data = observation['air_sensor_pollution']
        robot_sensor_pollution_data =  observation['sensor_pollution']
        pollution_end_time = self.pollution_end_time
        # IMU data
        delta_distance = np.linalg.norm(observation["disp_position"])
        delta_yaw = observation['disp_angle']
        # LiDAR data
        scan_ranges = observation['sensor_lidar_front']

        # Localization
        self.localizer(delta_distance, delta_yaw, scan_ranges, self.occupancy_grid_map, self.distance_map)
        current_robot_pose = self.current_robot_pose

        # Current time
        dt = 0.1
        current_time = self.steps * dt

        next_state, action = self.finite_state_machine(
            air_sensor_pollution_data,
            robot_sensor_pollution_data,
            current_time,
            pollution_end_time,
            current_robot_pose,
            self.current_fsm_state,
            self.map_id,
            self.room_num
        )
        self.current_fsm_state = next_state 
        # --------------------------------------------------

        # --------------------------------------------------
        # --------------------------------------------------
        # ------- 여기만 테스트 하세요 아빠 -------------------
        # --------------------------------------------------
        # --------------------------------------------------
        # action = self.move_along_path(1, 3, self.map_id)
        # --------------------------------------------------
        # --------------------------------------------------
        # --------------------------------------------------
        # --------------------------------------------------
        # --------------------------------------------------

        self.steps += 1 
        return action

    def learn(
            self,
            observation: Observation,
            info: Info,
            action,
            next_observation: Observation,
            next_info: Info,
            terminated,
            done,
            ):
        """
        실시간으로 훈련 상태를 전달받고 에이전트를 학습시키는 함수
        training 중에만 매 step마다 호출되며(act 호출 후), test 중에는 호출되지 않음
        강한 충돌(0.7m/s 이상 속도로 충돌)이 발생하면 시뮬레이션이 종료되고 terminated에 true가 전달됨 (실격)
        도킹 스테이션에 도착하면 시뮬레이션이 종료되고 done에 true가 전달됨

        Info: 센서 감지 정보 이외에 학습에 활용할 수 있는 정보 - 오직 training시에만 제공됨
        ---
        robot_position: (float, float), 로봇의 현재 world 좌표
        robot_angle: float, 로봇의 현재 각도
        collided: bool, 현재 로봇이 벽이나 물체에 충돌했는지 여부
        all_pollution: np.ndarray, 거치형 에어 센서가 없는 방까지 포함한 오염도 정보
        ---
        """
        # Only simulation
        self.true_robot_pose = (info["robot_position"][0], info["robot_position"][1], info["robot_angle"])

    def reset(self):
        """
        모델 상태 등을 초기화하는 함수
        training시, 각 episode가 끝날 때마다 호출됨 (initialize_map 호출 전)
        """
        self.steps = 0

    def log(self, msg):
        """
        터미널에 로깅하는 함수. print를 사용하면 비정상적으로 출력됨.
        ROS Node의 logger를 호출.
        """
        self.logger(str(msg))

    # ----------------------------------------------------------------------------------------------------
    # New defined functions

    def mission_planner(
            self, 
            air_sensor_pollution_data, 
            robot_sensor_pollution_data, 
            current_node_index,
            map_id,
            room_num,
            pollution_threshold=0.05
            ):
        """
        미션 플래너: 오염 감지된 방들을 기반으로 TSP 순서에 따라 task queue 생성

        Parameters:
        - air_sensor_pollution_data: list of float, 각 방의 공기 센서 오염 수치
        - robot_sensor_pollution_data: list of float, 로봇 센서 오염 수치 (현재 사용 안 함)
        - current_node_index: int, 현재 방(room)의 ID
        - map_id: 0, 1, 2, 3

        Return
        ------
        - best_path: List[int], 방문해야 할 방의 순서
        """
        distance_matrices = {
            0: [
                [0.0, 4.0, 0.0, 5.5],
                [5.6, 0.0, 0.0, 9.1],
                [2.1, 7.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            1: [
                [0.0, 4.6, 4.7, 5.0, 8.0, 0.2, 4.1],
                [4.6, 0.0, 8.1, 9.0, 13.1, 4.7, 7.4],
                [4.7, 8.1, 0.0, 7.1, 11.6, 4.8, 8.8],
                [5.0, 9.0, 7.1, 0.0, 11.6, 5.0, 9.0],
                [8.0, 12.5, 11.6, 11.6, 0.0, 7.8, 7.8],
                [0.2, 4.7, 4.9, 5.0, 7.8, 0.0, 4.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            2: [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            3: [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
        }

        unobserved_potential_regions = []

        # Polluted regions
        observed_polluted_regions = [
            room_id for room_id in range(room_num)
            if air_sensor_pollution_data[room_id] > pollution_threshold
        ]

        if not observed_polluted_regions:
            return []

        distance_matrix = distance_matrices.get(map_id) 
        dock_station_id = len(distance_matrix) - 1  # 마지막 인덱스가 도킹 스테이션

        min_cost = float('inf')
        optimal_visit_order = []

        # Calculate cost for every cases
        for perm in itertools.permutations(observed_polluted_regions):
            total_cost = 0
            last_visited = current_node_index

            for room_id in perm:
                total_cost += distance_matrix[last_visited][room_id]
                last_visited = room_id

            # Add Cost for last_visited to docking station
            total_cost += distance_matrix[last_visited][dock_station_id]

            if total_cost < min_cost:
                min_cost = total_cost
                optimal_visit_order = list(perm)

        return optimal_visit_order
    
    def global_planner(self, start_node_index, end_node_index, map_id):
        """
        Index rules
        - index of region is a index of matrix
        - last index is for docking station
        - (last - 1) index is for start position
        
        reference_waypoint_matrix[i][j] : waypoints from start node i to end node j
        """

        reference_waypoints = {
            0: {
                (0, 1): [(-1, 2), (-1, -2)],
                (0, 3): [(1.4, -3)],
                (1, 0): [(-1, 1.6), (1.2, 2)],
                (1, 3): [(-0.8, 1.6), (0.2, 1.2), (1.4, -3)],
                (2, 0): [(1.2, 2)],
                (2, 1): [(0.2, 1.6), (-0.8, 1.6), (-1, -2)],
            },
            1: {
                (0, 1): [(-0.2, -2.0), (-1.6, -3.4), (-2.77, -3.4), (-2.77, -2.2)],               
                (0, 2): [(-0.2, -2.0), (-1.57, -0.8), (-1.57, 0.8), (-2.4, 2.4)],
                (0, 3): [(-0.2, -2.0), (-0.62, -0.8), (-0.61, 0.8), (0.9, 0.9), (1.0, 2.8)],
                (0, 4): [(-0.2, -2.0), (0.8, -0.8), (3.9, -0.8), (3.9, 2.2)],
                (0, 5): [(-0.2, -2.0), (0, -2)],
                (0, 6): [(-0.2, -2.0), (0.8, -3.6), (2.8, -4.2)],

                (1, 0): [(-2.77, -2.2), (-2.77, -3.4), (-1.6, -3.4), (-0.2, -2.0)],
                (1, 2): [(-2.77, -2.2), (-2.77, -3.4), (-1.6, -3.4), (-1.57, -0.8), (-1.57, 0.8), (-2.4, 2.4)],
                (1, 3): [(-2.77, -2.2), (-2.77, -3.4), (-1.6, -3.4), (-0.62, -0.8), (-0.61, 0.8), (0.9, 0.9), (1.0, 2.8)],
                (1, 4): [(-2.77, -2.2), (-2.77, -3.4), (-1.6, -3.4), (0.8, -0.8), (3.9, -0.8), (3.9, 2.2)],
                (1, 5): [(-2.77, -2.2), (-2.77, -3.4), (-1.6, -3.4), (0, -2)],
                (1, 6): [(-2.77, -2.2), (-2.77, -3.4), (-1.6, -3.4), (0.8, -3.6), (2.8, -4.2)],
                            
                (2, 0): [(-2.4, 2.4), (-1.57, 0.8), (-1.57, -0.8), (-0.2, -2.0)],			
                (2, 1): [(-2.4, 2.4), (-1.57, 0.8), (-1.57, -0.8), (-1.6, -3.4), (-2.77, -3.4), (-2.77, -2.2)],
                (2, 3): [(-2.4, 2.4), (-1.57, 0.8), (-1.57, -0.8), (-0.62, -0.8), (-0.61, 0.8), (0.9, 0.9), (1.0, 2.8)],
                (2, 4): [(-2.4, 2.4), (-1.57, 0.8), (-1.57, -0.8), (3.9, -0.8), (3.9, 2.2)],
                (2, 5): [(-2.4, 2.4), (-1.57, 0.8), (-1.57, -0.8), (0, -2)],
                (2, 6): [(-2.4, 2.4), (-1.57, 0.8), (-1.57, -0.8), (0.8, -3.6), (2.8, -4.2)],			
                    
                (3, 0): [(1.0, 2.8), (0.9, 0.9), (-0.61, 0.8), (-0.62, -0.8), (-0.2, -2.0)],	
                (3, 1): [(1.0, 2.8), (0.9, 0.9), (-0.61, 0.8), (-0.62, -0.8), (-1.6, -3.4), (-2.77, -3.4), (-2.77, -2.2)],
                (3, 2): [(1.0, 2.8), (0.9, 0.9), (-0.61, 0.8), (-0.62, -0.8), (-1.57, -0.8), (-1.57, 0.8), (-2.4, 2.4)],
                (3, 4): [(1.0, 2.8), (0.9, 0.9), (-0.61, 0.8), (-0.62, -0.8), (3.9, -0.8), (3.9, 2.2)],
                (3, 5): [(1.0, 2.8), (0.9, 0.9), (-0.61, 0.8), (-0.62, -0.8), (0, -2)],
                (3, 6): [(1.0, 2.8), (0.9, 0.9), (-0.61, 0.8), (-0.62, -0.8), (0.8, -3.6), (2.8, -4.2)],

                (4, 0): [(3.9, 2.2), (3.9, -0.8), (0.8, -0.8), (-0.2, -2.0)],
                (4, 1): [(3.9, 2.2), (3.9, -0.8), (0.8, -0.8), (-1.6, -3.4), (-2.77, -3.4), (-2.77, -2.2)],
                (4, 2): [(3.9, 2.2), (3.9, -0.8), (-1.57, -0.8), (-1.57, 0.8), (-2.4, 2.4)],
                (4, 3): [(3.9, 2.2), (3.9, -0.8), (-0.62, -0.8), (-0.61, 0.8), (0.9, 0.9), (1.0, 2.8)],
                (4, 5): [(3.9, 2.2), (3.9, -0.8), (0.8, -0.8), (0, -2)],
                (4, 6): [(3.9, 2.2), (3.9, -0.8), (2.6, -0.8), (2.6, -4.2)],

                (5, 0): [(0, -2), (-0.2, -2.0)],
                (5, 1): [(0, -2), (-1.6, -3.4), (-2.77, -3.4), (-2.77, -2.2)],               
                (5, 2): [(0, -2), (-1.57, -0.8), (-1.57, 0.8), (-2.4, 2.4)],
                (5, 3): [(0, -2), (-0.62, -0.8), (-0.61, 0.8), (0.9, 0.9), (1.0, 2.8)],
                (5, 4): [(0, -2), (0.8, -0.8), (3.9, -0.8), (3.9, 2.2)],
                (5, 6): [(0, -2), (0.8, -3.6), (2.8, -4.2)],
            },
            2: {
                # (0, 1): [(3.8, 1.8), (-1.0, 1.6), (-2.4, 2.4), (-2.4, 4.8)],               
                # (0, 2): [(3.8, 1.8), (-4.2, -1.0), (-5.8, -1.0), (-5.8, 1.8)],
                # (0, 3): [(3.8, 1.8), (-4.2, -1.0), (-4.8, -1.6), (-5.2, -5.0)],
                # (0, 4): [(3.8, 1.8), (-2.8, -1.0), (-3.4, -1.6), (-3.4, -2.2), (-1.8, -3.4)],
                # (0, 5): [(3.8, 1.8), (0.2, -1.6), (0.8, -4.4), (0.2, -6.0), (-1.2, -6.0)],
                # (0, 6): [(3.8, 1.8), (0.2, -1.6), (0.8, -5.0)],
                # (0, 7): [(3.8, 1.8), (5.6, 4.6), (6.0, 5.2)],
                # (0, 8): [(3.8, 1.8), (3, 3)],
                # (0, 9): [(3.8, 1.8), (-0.2, 6.4)],

                # (1, 0): [(-2.4, 4.8), (-2.4, 2.4), (-1.0, 1.6), (3.8, 1.8)],
                # (1, 2): [(-2.4, 4.8), (-2.4, 2.4), (-4.2, -1.0), (-5.8, -1.0), (-5.8, 1.8)],
                # (1, 3): [(-2.4, 4.8), (-2.4, 2.4), (-4.2, -1.0), (-4.8, -1.6), (-5.2, -5.0)],
                # (1, 4): [(-2.4, 4.8), (-2.4, 2.4), (-3.4, -1.6), (-3.4, -2.2), (-1.8, -3.4)],
                # (1, 5): [(-2.4, 4.8), (-2.4, 2.4), (-0.4, -1.0), (0.2, -1.6), (0.8, -4.4), (0.2, -6.0), (-1.2, -6.0)],
                # (1, 6): [(-2.4, 4.8), (-2.4, 2.4), (-0.4, -1.0), (0.2, -1.6), (0.8, -5.0)],
                # (1, 7): [(-2.4, 4.8), (-2.4, 2.4), (-1.0, 1.6), (5.6, 4.6), (6.0, 5.2)],
                # (1, 8): [(-2.4, 4.8), (-2.4, 2.4), (-1.0, 1.6), (3, 3)],
                # (1, 9): [(-2.4, 4.8), (-2.4, 2.4), (-1.0, 1.6), (-0.2, 1.0), (-0.2, 6.4)],
                            
                # (2, 0): [(-5.8, 1.8), (-5.8, -1.0), (-4.2, -1.0), (3.8, 1.8)],			
                # (2, 1): [(-5.8, 1.8), (-5.8, -1.0), (-4.2, -1.0), (-2.4, 2.4), (-2.4, 4.8)],
                # (2, 3): [(-5.8, 1.8), (-5.8, -1.0), (-4.8, -1.6), (-5.2, -5.0)],
                # (2, 4): [(-5.8, 1.8), (-5.8, -1.0), (-4.2, -1.0), (-3.4, -1.6), (-3.4, -2.2), (-1.8, -3.4)],
                # (2, 5): [(-5.8, 1.8), (-5.8, -1.0), (-0.4, -1.0), (0.2, -1.6), (0.8, -4.4), (0.2, -6.0), (-1.2, -6.0)],
                # (2, 6): [(-5.8, 1.8), (-5.8, -1.0), (-0.4, -1.0), (0.2, -1.6), (0.8, -5.0)],
                # (2, 7): [(-5.8, 1.8), (-5.8, -1.0), (-4.2, -1.0), (5.6, 4.6), (6.0, 5.2)],
                # (2, 8): [(-5.8, 1.8), (-5.8, -1.0), (-4.2, -1.0), (3, 3)],
                # (2, 9): [(-5.8, 1.8), (-5.8, -1.0), (-4.2, -1.0), (-0.2, 1.0), (-0.2, 6.4)],		
                    
                # (3, 0): [(-5.2, -5.0), (-4.8, -1.6), (-4.2, -1.0), (3.8, 1.8)],	
                # (3, 1): [(-5.2, -5.0), (-4.8, -1.6), (-4.2, -1.0), (-2.4, 2.4), (-2.4, 4.8)],
                # (3, 2): [(-5.2, -5.0), (-4.8, -1.6), (-5.8, -1.0), (-5.8, 1.8)],
                # (3, 4): [(-5.2, -5.0), (-4.8, -1.6), (-4.2, -1.0), (-3.4, -1.6), (-3.4, -2.2), (-1.8, -3.4)],
                # (3, 5): [(-5.2, -5.0), (-4.8, -1.6), (-4.2, -1.0), (-0.4, -1.0), (0.2, -1.6), (0.8, -4.4), (0.2, -6.0), (-1.2, -6.0)],
                # (3, 6): [(-5.2, -5.0), (-4.8, -1.6), (-4.2, -1.0), (-0.4, -1.0), (0.2, -1.6), (0.8, -5.0)],
                # (3, 7): [(-5.2, -5.0), (-4.8, -1.6), (-4.2, -1.0), (5.6, 4.6), (6.0, 5.2)],
                # (3, 8): [(-5.2, -5.0), (-4.8, -1.6), (-4.2, -1.0), (3, 3)],
                # (3, 9): [(-5.2, -5.0), (-4.8, -1.6), (-4.2, -1.0), (-0.2, 1.0), (-0.2, 6.4)],

                # (4, 0): [(-1.8, -3.4), (-3.4, -2.2), (-3.4, -1.6), (-2.8, -1.0), (3.8, 1.8)],
                # (4, 1): [(-1.8, -3.4), (-3.4, -2.2), (-3.4, -1.6), (-2.4, 2.4), (-2.4, 4.8)],
                # (4, 2): [(-1.8, -3.4), (-3.4, -2.2), (-3.4, -1.6), (-4.2, -1.0), (-5.8, -1.0), (-5.8, 1.8)],
                # (4, 3): [(-1.8, -3.4), (-3.4, -2.2), (-3.4, -1.6), (-4.2, -1.0), (-4.8, -1.6), (-5.2, -5.0)],
                # (4, 5): [(-1.8, -3.4), (-3.4, -2.2), (-3.4, -1.6), (-2.8, -1.0), (-0.4, -1.0), (0.2, -1.6), (0.8, -4.4), (0.2, -6.0), (-1.2, -6.0)],
                # (4, 6): [(-1.8, -3.4), (-3.4, -2.2), (-3.4, -1.6), (-2.8, -1.0), (-0.4, -1.0), (0.2, -1.6), (0.8, -5.0)],
                # (4, 7): [(-1.8, -3.4), (-3.4, -2.2), (-3.4, -1.6), (-2.8, -1.0), (5.6, 4.6), (6.0, 5.2)],
                # (4, 8): [(-1.8, -3.4), (-3.4, -2.2), (-3.4, -1.6), (-2.8, -1.0), (3, 3)],
                # (4, 9): [(-1.8, -3.4), (-3.4, -2.2), (-3.4, -1.6), (-2.8, -1.0), (-0.2, 1.0), (-0.2, 6.4)],

                # (5, 0): [(-1.2, -6.0), (0.2, -6.0), (0.8, -4.4), (0.2, -1.6), (3.8, 1.8)],
                # (5, 1): [(-1.2, -6.0), (0.2, -6.0), (0.8, -4.4), (0.2, -1.6), (-2.4, 2.4), (-2.4, 4.8)],           
                # (5, 2): [(-1.2, -6.0), (0.2, -6.0), (0.8, -4.4), (0.2, -1.6), (-0.4, -1.0), (-5.8, -1.0), (-5.8, 1.8)],
                # (5, 3): [(-1.2, -6.0), (0.2, -6.0), (0.8, -4.4), (0.2, -1.6), (-0.4, -1.0), (-4.2, -1.0), (-4.8, -1.6), (-5.2, -5.0)],
                # (5, 4): [(-1.2, -6.0), (0.2, -6.0), (0.8, -4.4), (0.2, -1.6), (-0.4, -1.0), (-2.8, -1.0), (-3.4, -1.6), (-3.4, -2.2), (-1.8, -3.4)],
                # (5, 6): [(-1.2, -6.0), (0.2, -6.0), (0.8, -5.0)],
                # (5, 7): [(-1.2, -6.0), (0.2, -6.0), (0.8, -4.4), (0.2, -1.6), (5.6, 4.6), (6.0, 5.2)],
                # (5, 8): [(-1.2, -6.0), (0.2, -6.0), (0.8, -4.4), (0.2, -1.6), (3, 3)],
                # (5, 9): [(-1.2, -6.0), (0.2, -6.0), (0.8, -4.4), (0.2, -1.6), (-0.2, 6.4)],

                # (6, 0): [(0.8, -5.0), (0.2, -1.6), (3.8, 1.8)],
                # (6, 1): [(0.8, -5.0), (0.2, -1.6), (-2.4, 2.4), (-2.4, 4.8)],              
                # (6, 2): [(0.8, -5.0), (0.2, -1.6), (-0.4, -1.0), (-5.8, -1.0), (-5.8, 1.8)],
                # (6, 3): [(0.8, -5.0), (0.2, -1.6), (-0.4, -1.0), (-4.2, -1.0), (-4.8, -1.6), (-5.2, -5.0)],
                # (6, 4): [(0.8, -5.0), (0.2, -1.6), (-0.4, -1.0), (-2.8, -1.0), (-3.4, -1.6), (-3.4, -2.2), (-1.8, -3.4)],
                # (6, 5): [(0.8, -5.0), (0.2, -6.0), (-1.2, -6.0)],
                # (6, 7): [(0.8, -5.0), (0.2, -1.6), (5.6, 4.6), (6.0, 5.2)],
                # (6, 8): [(0.8, -5.0), (0.2, -1.6), (3, 3)],
                # (6, 9): [(0.8, -5.0), (0.2, -1.6), (-0.2, 6.4)],

                # (7, 0): [(6.0, 5.2), (5.6, 4.6), (3.8, 1.8)],
                # (7, 1): [(6.0, 5.2), (5.6, 4.6), (-1.0, 1.6), (-2.4, 2.4), (-2.4, 4.8)],                
                # (7, 2): [(6.0, 5.2), (5.6, 4.6), (-4.2, -1.0), (-5.8, -1.0), (-5.8, 1.8)],
                # (7, 3): [(6.0, 5.2), (5.6, 4.6), (-4.2, -1.0), (-4.8, -1.6), (-5.2, -5.0)],
                # (7, 4): [(6.0, 5.2), (5.6, 4.6), (-2.8, -1.0), (-3.4, -1.6), (-3.4, -2.2), (-1.8, -3.4)],
                # (7, 5): [(6.0, 5.2), (5.6, 4.6), (0.2, -1.6), (0.8, -4.4), (0.2, -6.0), (-1.2, -6.0)],
                # (7, 6): [(6.0, 5.2), (5.6, 4.6), (0.2, -1.6), (0.8, -5.0)],
                # (7, 8): [(6.0, 5.2), (5.6, 4.6), (3, 3)],
                # (7, 9): [(6.0, 5.2), (5.6, 4.6), (-0.2, 6.4)],

                # (8, 0): [(3, 3), (3.8, 1.8)],
                # (8, 1): [(3, 3), (-1.0, 1.6), (-2.4, 2.4), (-2.4, 4.8)],               
                # (8, 2): [(3, 3), (-4.2, -1.0), (-5.8, -1.0), (-5.8, 1.8)],
                # (8, 3): [(3, 3), (-4.2, -1.0), (-4.8, -1.6), (-5.2, -5.0)],
                # (8, 4): [(3, 3), (-2.8, -1.0), (-3.4, -1.6), (-3.4, -2.2), (-1.8, -3.4)],
                # (8, 5): [(3, 3), (0.2, -1.6), (0.8, -4.4), (0.2, -6.0), (-1.2, -6.0)],
                # (8, 6): [(3, 3), (0.2, -1.6), (0.8, -5.0)],
                # (8, 7): [(3, 3), (5.6, 4.6), (6.0, 5.2)],
                # (8, 9): [(3, 3), (-0.2, 6.4)],

                (0, 0): [(-40, 0)],
                (0, 1): [(-40, 0)],
                (0, 2): [(-40, 0)],
                (0, 3): [(-40, 0)],
                (0, 4): [(-40, 0)],
                (0, 5): [(-40, 0)],
                (0, 6): [(-40, 0)],
                (0, 7): [(-40, 0)],
                (0, 8): [(-40, 0)],
                (0, 9): [(-40, 0)],
                (0, 10): [(-40, 0)],
                (0, 11): [(-40, 0)],
                (0, 12): [(-40, 0)],
                (0, 13): [(-40, 0)],
                (0, 14): [(-40, 0)],

                (1, 0): [(-40, 0)],
                (1, 1): [(-40, 0)],
                (1, 2): [(-40, 0)],
                (1, 3): [(-40, 0)],
                (1, 4): [(-40, 0)],
                (1, 5): [(-40, 0)],
                (1, 6): [(-40, 0)],
                (1, 7): [(-40, 0)],
                (1, 8): [(-40, 0)],
                (1, 9): [(-40, 0)],

                (2, 0): [(-40, 0)],
                (2, 1): [(-40, 0)],
                (2, 2): [(-40, 0)],
                (2, 3): [(-40, 0)],
                (2, 4): [(-40, 0)],
                (2, 5): [(-40, 0)],
                (2, 6): [(-40, 0)],
                (2, 7): [(-40, 0)],
                (2, 8): [(-40, 0)],
                (2, 9): [(-40, 0)],

                (3, 0): [(-40, 0)],
                (3, 1): [(-40, 0)],
                (3, 2): [(-40, 0)],
                (3, 3): [(-40, 0)],
                (3, 4): [(-40, 0)],
                (3, 5): [(-40, 0)],
                (3, 6): [(-40, 0)],
                (3, 7): [(-40, 0)],
                (3, 8): [(-40, 0)],
                (3, 9): [(-40, 0)],

                (4, 0): [(-40, 0)],
                (4, 1): [(-40, 0)],
                (4, 2): [(-40, 0)],
                (4, 3): [(-40, 0)],
                (4, 4): [(-40, 0)],
                (4, 5): [(-40, 0)],
                (4, 6): [(-40, 0)],
                (4, 7): [(-40, 0)],
                (4, 8): [(-40, 0)],
                (4, 9): [(-40, 0)],

                (5, 0): [(-40, 0)],
                (5, 1): [(-40, 0)],
                (5, 2): [(-40, 0)],
                (5, 3): [(-40, 0)],
                (5, 4): [(-40, 0)],
                (5, 5): [(-40, 0)],
                (5, 6): [(-40, 0)],
                (5, 7): [(-40, 0)],
                (5, 8): [(-40, 0)],
                (5, 9): [(-40, 0)],

                (6, 0): [(-40, 0)],
                (6, 1): [(-40, 0)],
                (6, 2): [(-40, 0)],
                (6, 3): [(-40, 0)],
                (6, 4): [(-40, 0)],
                (6, 5): [(-40, 0)],
                (6, 6): [(-40, 0)],
                (6, 7): [(-40, 0)],
                (6, 8): [(-40, 0)],
                (6, 9): [(-40, 0)],

                (7, 0): [(-40, 0)],
                (7, 1): [(-40, 0)],
                (7, 2): [(-40, 0)],
                (7, 3): [(-40, 0)],
                (7, 4): [(-40, 0)],
                (7, 5): [(-40, 0)],
                (7, 6): [(-40, 0)],
                (7, 7): [(-40, 0)],
                (7, 8): [(-40, 0)],
                (7, 9): [(-40, 0)],

                (8, 0): [(-40, 0)],
                (8, 1): [(-40, 0)],
                (8, 2): [(-40, 0)],
                (8, 3): [(-40, 0)],
                (8, 4): [(-40, 0)],
                (8, 5): [(-40, 0)],
                (8, 6): [(-40, 0)],
                (8, 7): [(-40, 0)],
                (8, 8): [(-40, 0)],
                (8, 9): [(-40, 0)],
            },
            3: {
                (0, 0): [(0, 0)],
                (0, 1): [(0, 0)],
                (0, 2): [(0, 0)],
                (0, 3): [(0, 0)],
                (0, 4): [(0, 0)],
                (0, 5): [(0, 0)],
                (0, 6): [(0, 0)],
                (0, 7): [(0, 0)],
                (0, 8): [(0, 0)],
                (0, 9): [(0, 0)],
                (0, 10): [(0, 0)],
                (0, 11): [(0, 0)],
                (0, 12): [(0, 0)],
                (0, 13): [(0, 0)],
                (0, 14): [(0, 0)],

                (1, 0): [(0, 0)],
                (1, 1): [(0, 0)],
                (1, 2): [(0, 0)],
                (1, 3): [(0, 0)],
                (1, 4): [(0, 0)],
                (1, 5): [(0, 0)],
                (1, 6): [(0, 0)],
                (1, 7): [(0, 0)],
                (1, 8): [(0, 0)],
                (1, 9): [(0, 0)],
                (1, 10): [(0, 0)],
                (1, 11): [(0, 0)],
                (1, 12): [(0, 0)],
                (1, 13): [(0, 0)],
                (1, 14): [(0, 0)],

                (2, 0): [(0, 0)],
                (2, 1): [(0, 0)],
                (2, 2): [(0, 0)],
                (2, 3): [(0, 0)],
                (2, 4): [(0, 0)],
                (2, 5): [(0, 0)],
                (2, 6): [(0, 0)],
                (2, 7): [(0, 0)],
                (2, 8): [(0, 0)],
                (2, 9): [(0, 0)],
                (2, 10): [(0, 0)],
                (2, 11): [(0, 0)],
                (2, 12): [(0, 0)],
                (2, 13): [(0, 0)],
                (2, 14): [(0, 0)],

                (3, 0): [(0, 0)],
                (3, 1): [(0, 0)],
                (3, 2): [(0, 0)],
                (3, 3): [(0, 0)],
                (3, 4): [(0, 0)],
                (3, 5): [(0, 0)],
                (3, 6): [(0, 0)],
                (3, 7): [(0, 0)],
                (3, 8): [(0, 0)],
                (3, 9): [(0, 0)],
                (3, 10): [(0, 0)],
                (3, 11): [(0, 0)],
                (3, 12): [(0, 0)],
                (3, 13): [(0, 0)],
                (3, 14): [(0, 0)],

                (4, 0): [(0, 0)],
                (4, 1): [(0, 0)],
                (4, 2): [(0, 0)],
                (4, 3): [(0, 0)],
                (4, 4): [(0, 0)],
                (4, 5): [(0, 0)],
                (4, 6): [(0, 0)],
                (4, 7): [(0, 0)],
                (4, 8): [(0, 0)],
                (4, 9): [(0, 0)],
                (4, 10): [(0, 0)],
                (4, 11): [(0, 0)],
                (4, 12): [(0, 0)],
                (4, 13): [(0, 0)],
                (4, 14): [(0, 0)],

                (5, 0): [(0, 0)],
                (5, 1): [(0, 0)],
                (5, 2): [(0, 0)],
                (5, 3): [(0, 0)],
                (5, 4): [(0, 0)],
                (5, 5): [(0, 0)],
                (5, 6): [(0, 0)],
                (5, 7): [(0, 0)],
                (5, 8): [(0, 0)],
                (5, 9): [(0, 0)],
                (5, 10): [(0, 0)],
                (5, 11): [(0, 0)],
                (5, 12): [(0, 0)],
                (5, 13): [(0, 0)],
                (5, 14): [(0, 0)],

                (6, 0): [(0, 0)],
                (6, 1): [(0, 0)],
                (6, 2): [(0, 0)],
                (6, 3): [(0, 0)],
                (6, 4): [(0, 0)],
                (6, 5): [(0, 0)],
                (6, 6): [(0, 0)],
                (6, 7): [(0, 0)],
                (6, 8): [(0, 0)],
                (6, 9): [(0, 0)],
                (6, 10): [(0, 0)],
                (6, 11): [(0, 0)],
                (6, 12): [(0, 0)],
                (6, 13): [(0, 0)],
                (6, 14): [(0, 0)],

                (7, 0): [(0, 0)],
                (7, 1): [(0, 0)],
                (7, 2): [(0, 0)],
                (7, 3): [(0, 0)],
                (7, 4): [(0, 0)],
                (7, 5): [(0, 0)],
                (7, 6): [(0, 0)],
                (7, 7): [(0, 0)],
                (7, 8): [(0, 0)],
                (7, 9): [(0, 0)],
                (7, 10): [(0, 0)],
                (7, 11): [(0, 0)],
                (7, 12): [(0, 0)],
                (7, 13): [(0, 0)],
                (7, 14): [(0, 0)],

                (8, 0): [(0, 0)],
                (8, 1): [(0, 0)],
                (8, 2): [(0, 0)],
                (8, 3): [(0, 0)],
                (8, 4): [(0, 0)],
                (8, 5): [(0, 0)],
                (8, 6): [(0, 0)],
                (8, 7): [(0, 0)],
                (8, 8): [(0, 0)],
                (8, 9): [(0, 0)],
                (8, 10): [(0, 0)],
                (8, 11): [(0, 0)],
                (8, 12): [(0, 0)],
                (8, 13): [(0, 0)],
                (8, 14): [(0, 0)],

                (9, 0): [(0, 0)],
                (9, 1): [(0, 0)],
                (9, 2): [(0, 0)],
                (9, 3): [(0, 0)],
                (9, 4): [(0, 0)],
                (9, 5): [(0, 0)],
                (9, 6): [(0, 0)],
                (9, 7): [(0, 0)],
                (9, 8): [(0, 0)],
                (9, 9): [(0, 0)],
                (9, 10): [(0, 0)],
                (9, 11): [(0, 0)],
                (9, 12): [(0, 0)],
                (9, 13): [(0, 0)],
                (9, 14): [(0, 0)],

                (10, 0): [(0, 0)],
                (10, 1): [(0, 0)],
                (10, 2): [(0, 0)],
                (10, 3): [(0, 0)],
                (10, 4): [(0, 0)],
                (10, 5): [(0, 0)],
                (10, 6): [(0, 0)],
                (10, 7): [(0, 0)],
                (10, 8): [(0, 0)],
                (10, 9): [(0, 0)],
                (10, 10): [(0, 0)],
                (10, 11): [(0, 0)],
                (10, 12): [(0, 0)],
                (10, 13): [(0, 0)],
                (10, 14): [(0, 0)],

                (11, 0): [(0, 0)],
                (11, 1): [(0, 0)],
                (11, 2): [(0, 0)],
                (11, 3): [(0, 0)],
                (11, 4): [(0, 0)],
                (11, 5): [(0, 0)],
                (11, 6): [(0, 0)],
                (11, 7): [(0, 0)],
                (11, 8): [(0, 0)],
                (11, 9): [(0, 0)],
                (11, 10): [(0, 0)],
                (11, 11): [(0, 0)],
                (11, 12): [(0, 0)],
                (11, 13): [(0, 0)],
                (11, 14): [(0, 0)],

                (12, 0): [(0, 0)],
                (12, 1): [(0, 0)],
                (12, 2): [(0, 0)],
                (12, 3): [(0, 0)],
                (12, 4): [(0, 0)],
                (12, 5): [(0, 0)],
                (12, 6): [(0, 0)],
                (12, 7): [(0, 0)],
                (12, 8): [(0, 0)],
                (12, 9): [(0, 0)],
                (12, 10): [(0, 0)],
                (12, 11): [(0, 0)],
                (12, 12): [(0, 0)],
                (12, 13): [(0, 0)],
                (12, 14): [(0, 0)],

                (13, 0): [(0, 0)],
                (13, 1): [(0, 0)],
                (13, 2): [(0, 0)],
                (13, 3): [(0, 0)],
                (13, 4): [(0, 0)],
                (13, 5): [(0, 0)],
                (13, 6): [(0, 0)],
                (13, 7): [(0, 0)],
                (13, 8): [(0, 0)],
                (13, 9): [(0, 0)],
                (13, 10): [(0, 0)],
                (13, 11): [(0, 0)],
                (13, 12): [(0, 0)],
                (13, 13): [(0, 0)],
                (13, 14): [(0, 0)],

                (14, 0): [(0, 0)],
                (14, 1): [(0, 0)],
                (14, 2): [(0, 0)],
                (14, 3): [(0, 0)],
                (14, 4): [(0, 0)],
                (14, 5): [(0, 0)],
                (14, 6): [(0, 0)],
                (14, 7): [(0, 0)],
                (14, 8): [(0, 0)],
                (14, 9): [(0, 0)],
                (14, 10): [(0, 0)],
                (14, 11): [(0, 0)],
                (14, 12): [(0, 0)],
                (14, 13): [(0, 0)],
                (14, 14): [(0, 0)]
            },
        }

        map_reference_waypoints = reference_waypoints[map_id]
        
        # Check validity of map_id
        if map_id not in reference_waypoints:
            return None

        # Check validity of node indexes
        if (start_node_index, end_node_index) not in map_reference_waypoints:
            return None

        return map_reference_waypoints[(start_node_index, end_node_index)]

    def controller(self, current_robot_pose, target_position, 
                linear_gain=1.0, angular_gain=2.0, 
                max_linear=1, max_angular=1.0,
                angle_threshold=0.2):
        """
        목표 방향을 먼저 향하도록 하고, 방향이 맞으면 직진.
        """
        x, y, theta = current_robot_pose
        target_x, target_y = target_position

        dx = target_x - x
        dy = target_y - y
        distance = math.hypot(dx, dy)
        target_angle = math.atan2(dy, dx)

        # 방향 오차
        angle_error = target_angle - theta
        angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))  # -pi ~ pi

        # 방향 우선 제어
        if abs(angle_error) > angle_threshold:
            linear_velocity = 0.0  # 먼저 회전
            angular_velocity = max(-max_angular, min(max_angular, angular_gain * angle_error))
        else:
            linear_velocity = max(-max_linear, min(max_linear, linear_gain * distance))
            angular_velocity = max(-max_angular, min(max_angular, angular_gain * angle_error))

        return linear_velocity, angular_velocity

    def localizer(self, delta_distance, delta_yaw, scan_ranges, occupancy_grid_map, distance_map):
        """
        Localization by Particle Filter
        """
        # Basic
        # x, y, yaw = self.current_robot_pose

        # new_x = x + delta_distance * math.cos(yaw)
        # new_y = y + delta_distance * math.sin(yaw)
        # new_yaw = (yaw + delta_yaw + math.pi) % (2 * math.pi) - math.pi
        # self.current_robot_pose = (new_x, new_y, new_yaw)

        # Localization by Particle Filter
        self.particle_filter.update_particles_by_motion_model(delta_distance, delta_yaw)
        self.particle_filter.update_weights_by_measurement_model(
            scan_ranges=scan_ranges,
            occupancy_grid_map=occupancy_grid_map,
            distance_map=distance_map,
            map_origin=self.map_origin,
            map_resolution=self.resolution
        )
        estimated_x, estimated_y, estimated_yaw = self.particle_filter.estimate_robot_pose()
        self.current_robot_pose = (estimated_x, estimated_y, estimated_yaw)
        self.particle_filter.resample_particles()

        # Print error
        if True:
            if (
                self.true_robot_pose is None 
                or self.current_robot_pose is None
                or None in self.true_robot_pose
                or None in self.current_robot_pose
            ):
                return

            dx = self.current_robot_pose[0] - self.true_robot_pose[0]
            dy = self.current_robot_pose[1] - self.true_robot_pose[1]
            self.distance_error = math.hypot(dx, dy)
            if self.distance_error > self.max_localization_error: self.max_localization_error = self.distance_error
            # self.log(f"PF Error: {distance_error:.3f}, MAX Error: {self.max_localization_error}")

    # Main Logic
    def finite_state_machine(self,
            air_sensor_pollution_data,
            robot_sensor_pollution_data,
            current_time,
            pollution_end_time,
            current_robot_pose, 
            current_fsm_state,
            map_id,
            room_num,
            pollution_threshold=0.05 
            ):
        """
        current_fsm_state -> next_fsm_state, action
        """
        # Define states of finite state machine
        FSM_READY = "READY"
        FSM_CLEANING = "CLEANING"
        FSM_NAVIGATING = "NAVIGATING"
        FSM_RETURNING = "RETURNING"

        # Indexes of initial node and docking station node by map
        initial_node_index = room_num
        docking_station_node_index = room_num + 1

        def calculate_distance_to_target_position(current_position, target_position):
            """
            Euclidean distance
            """
            dx = target_position[0] - current_position[0]
            dy = target_position[1] - current_position[1]
            distance = math.hypot(dx, dy)
            return distance

        def is_target_reached(current_position, target_position, threshold=0.05):
            """
            Decide if robot reached in target position by threshold
            """
            return calculate_distance_to_target_position(current_position, target_position) < threshold
        
        def are_no_polluted_rooms(air_sensor_pollution_data):
            """
            Decide if there are polluted rooms
            """
            return all(pollution <= 0 for pollution in air_sensor_pollution_data) # True: there are no polluted rooms

        current_robot_position = current_robot_pose[0], current_robot_pose[1]

        # ---------------------------------------------------------------------------- #
        # [State] READY (start state) ------------------------------------------------ #
        # ---------------------------------------------------------------------------- #
        if current_fsm_state == FSM_READY:
            # Mission planning
            optimal_visit_order = self.mission_planner(
                air_sensor_pollution_data, 
                robot_sensor_pollution_data, 
                current_node_index=initial_node_index,
                map_id=map_id,
                room_num=room_num
                )

            # State transition
            # READY -> NAVIGATING
            if optimal_visit_order: # 목표 구역이 있음
                next_fsm_state = FSM_NAVIGATING
                self.optimal_next_node_index = optimal_visit_order[0]
                self.waypoints = self.global_planner(start_node_index=initial_node_index, end_node_index=self.optimal_next_node_index, map_id=map_id)
                self.current_waypoint_index = 0
                self.tmp_target_position = self.waypoints[0]
            
            # READY -> READY
            else:
                next_fsm_state = FSM_READY

            action = (0, 0, 0) # Stop

        # ---------------------------------------------------------------------------- #
        # [State] Navigating --------------------------------------------------------- #
        # ---------------------------------------------------------------------------- #
        elif current_fsm_state == FSM_NAVIGATING:
            # State transition
            # NAVIGATING -> CLEANING
            if is_target_reached(current_robot_position, self.waypoints[-1]): # 목표 구역에 도달함
                next_fsm_state = FSM_CLEANING
                self.current_waypoint_index = 0
                self.current_node_index = self.optimal_next_node_index
            
            # NAVIGATING -> NAVIGATING
            else:
                next_fsm_state = FSM_NAVIGATING

            # set next tempt target point 이 순서를 바꾸면 왜 문제가 생길까?
            if is_target_reached(current_robot_position, self.tmp_target_position) and self.current_waypoint_index < len(self.waypoints) - 1:
                self.current_waypoint_index += 1
                self.tmp_target_position = self.waypoints[self.current_waypoint_index]

            linear_velocity, angular_velocity = self.controller(current_robot_pose, self.tmp_target_position)
            action = (0, linear_velocity, angular_velocity)

        # ---------------------------------------------------------------------------- #
        # [State] CLEANING ----------------------------------------------------------- #
        # ---------------------------------------------------------------------------- #
        elif current_fsm_state == FSM_CLEANING:
            # Mission planning
            optimal_visit_order = self.mission_planner(
                air_sensor_pollution_data, 
                robot_sensor_pollution_data, 
                current_node_index=self.current_node_index,
                map_id=map_id,
                room_num=room_num
                )          

            # State transition
            # CLEANING -> RETURNING
            if are_no_polluted_rooms(air_sensor_pollution_data) and current_time >= pollution_end_time:     # 오염 구역이 없음
                next_fsm_state = FSM_RETURNING

                self.current_waypoint_index = 0
                self.waypoints = self.global_planner(start_node_index=self.current_node_index, end_node_index=docking_station_node_index, map_id=map_id)
                self.tmp_target_position = self.waypoints[0]

            # tmp
            elif current_time >= pollution_end_time and map_id != 0:
                next_fsm_state = FSM_RETURNING

                self.current_waypoint_index = 0
                self.waypoints = self.global_planner(start_node_index=self.current_node_index, end_node_index=docking_station_node_index, map_id=map_id)
                self.tmp_target_position = self.waypoints[0]

            # CLEANING -> NAVIGATING
            elif air_sensor_pollution_data[self.optimal_next_node_index] < pollution_threshold and optimal_visit_order:       # 청정 완료함
                next_fsm_state = FSM_NAVIGATING

                # 꼭 이때 해야 할까?
                self.optimal_next_node_index = optimal_visit_order[0]
                self.waypoints = self.global_planner(start_node_index=self.current_node_index, end_node_index=self.optimal_next_node_index, map_id=map_id)
                self.current_waypoint_index = 0
                self.tmp_target_position = self.waypoints[0]  

            # CLEANING -> CLEANING
            else:
                next_fsm_state = FSM_CLEANING
            
            action = (1, 0, 0) # Stop and clean

        # ---------------------------------------------------------------------------- #
        # [State] RETURNING (end state) ---------------------------------------------- #
        # ---------------------------------------------------------------------------- #
        elif current_fsm_state == FSM_RETURNING:
            next_fsm_state = FSM_RETURNING
            
            linear_velocity, angular_velocity = self.controller(current_robot_pose, self.tmp_target_position)
            action = (0, linear_velocity, angular_velocity)

            if is_target_reached(current_robot_position, self.tmp_target_position) and self.current_waypoint_index < len(self.waypoints) - 1:
                self.current_waypoint_index += 1
                self.tmp_target_position = self.waypoints[self.current_waypoint_index]

        # log
        if current_fsm_state == FSM_NAVIGATING:
            self.log(f"[{current_time:.1f}] [NAVIGATING] {self.current_node_index} -> {self.optimal_next_node_index} | PF Error: {self.distance_error:.3f}")
        elif current_fsm_state == FSM_CLEANING:
            self.log(f"[{current_time:.1f}] [CLEANING] {self.current_node_index}: {air_sensor_pollution_data[self.current_node_index]:.3f} | PF Error: {self.distance_error:.3f}")
        else:
            self.log(f"[{current_time:.1f}] [{current_fsm_state}] | PF Error: {self.distance_error:.3f}")


        return next_fsm_state, action


    def move_along_path(self, start_node, end_node, map_id, distance_threshold=0.1):
        """
        주어진 start_node → end_node 경로를 따라가며 이동하는 함수

        just for obtaining global plan
        """
        waypoints = self.global_planner(self.tmp_start_node, self.tmp_end_node, map_id=map_id)
        if not waypoints:
            print(f"[ERROR] No path from {start_node} to {end_node}")
            return (0, 0, 0)

        # 현재 목표 웨이포인트
        self.target_position = waypoints[self.current_waypoint_index]
        linear_velocity, angular_velocity = self.controller(self.true_robot_pose, self.target_position)
        action = (0, linear_velocity, angular_velocity)

        # 현재 위치와 목표 위치 거리 계산
        distance_to_target = math.hypot(
            self.target_position[0] - self.true_robot_pose[0],
            self.target_position[1] - self.true_robot_pose[1]
        )
        if distance_to_target < distance_threshold:
            if self.current_waypoint_index < len(waypoints) - 1:
                self.current_waypoint_index += 1
                self.target_position = waypoints[self.current_waypoint_index]
            else:
                print("Final waypoint reached!")
                self.current_waypoint_index = 0

                self.tmp_start_node = start_node
                self.tmp_end_node = end_node
                waypoints = self.global_planner(self.tmp_start_node, self.tmp_start_node, map_id)
                if waypoints:
                    self.target_position = waypoints[self.current_waypoint_index]
                else:
                    action = (0, 0, 0)

        return action
