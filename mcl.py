import math
import numpy as np
from dataclasses import dataclass, field

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
            omega_slow=0.0,
            omega_fast=0.0,
            alpha_slow=0.001,
            alpha_fast=0.9
            ):
        
        # Constants
        self.odom_noise = odom_noise
        self.initial_pose_noise = initial_pose_noise
        self.particle_num = particle_num
        self.omega_slow = omega_slow
        self.omega_fast = omega_fast
        self.alpha_slow = alpha_slow
        self.alpha_fast = alpha_fast

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

    # 수정 안 함

    def update_weights_by_measurement_model(
        self,
        scan_ranges: np.ndarray,
        occupancy_grid_map: np.ndarray,
        distance_map: np.ndarray,

        # Map
        map_origin=(float, float),
        map_resolution= float,

        # LiDAR
        min_angle=-1.9,
        angle_increment=0.016,
        min_range=0.05,
        max_range=8.0,
        scan_step=3, # or beam_sample_count

        sigma_hit=0.2,
        z_hit=0.95,
        z_rand=0.05
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
        """   

        eps = 1e-12

        beam_num = len(scan_ranges)
        sampled_beam_indices = np.arange(0, beam_num, scan_step)
        sampled_beam_angles = min_angle + sampled_beam_indices * angle_increment
        sampled_scan_ranges = scan_ranges[sampled_beam_indices]
        
        denominator = 2.0 * (sigma_hit ** 2)
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
                if not (min_range < range_measurement < max_range) or np.isinf(range_measurement) or np.isnan(range_measurement):
                    log_likelihood += math.log(z_rand + eps)
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
                    total_prob = z_hit * prob_hit + z_rand * (1.0 / max_range)
                    
                    log_likelihood += math.log(total_prob + eps)
                else:
                    log_likelihood += math.log(z_rand + eps)
        
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
        """
        def calculate_amcl_random_particle_rate(average_likelihood):
            self.omega_slow += self.alpha_slow * (average_likelihood - self.omega_slow)
            self.omega_fast += self.alpha_fast * (average_likelihood - self.omega_fast)
            amcl_random_particle_rate = 1.0 - self.omega_fast / self.omega_slow
            return max(amcl_random_particle_rate, 0.0)

        def calculate_effective_sample_size(total_likelihood):
            sum = 0.0
            for index in range(self.particle_num):
                weight = self.particles[index].weight / total_likelihood
                self.particles[index].weight = weight
                sum += weight * weight
            effective_sample_size = 1.0 / sum
            return effective_sample_size

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

if __name__ == '__main__':
    mcl = MonteCarloLocalizer()
    mcl.initialize_particles(Pose(10, 10, 0), Pose(0.1, 0.1, 0.1))
    print(mcl.particles[0])
    mcl.update_particles_by_motion_model(0.1, 0.2)
    print(mcl.particles[0])
    robot_pose = mcl.estimate_robot_pose()
    print(robot_pose)

    mcl.resample_particles()