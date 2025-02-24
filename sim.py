import scipy.spatial.transform
import numpy as np
from animate_function import QuadPlotter
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from model import Phi_Net, load_model
import torch

def quat_mult(q, p):
    # q * p
    # p,q = [w x y z]
    return np.array(
        [
            p[0] * q[0] - q[1] * p[1] - q[2] * p[2] - q[3] * p[3],
            q[1] * p[0] + q[0] * p[1] + q[2] * p[3] - q[3] * p[2],
            q[2] * p[0] + q[0] * p[2] + q[3] * p[1] - q[1] * p[3],
            q[3] * p[0] + q[0] * p[3] + q[1] * p[2] - q[2] * p[1],
        ]
    )
    
def quat_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quaternion_from_vectors(v_from, v_to):
    v_from = normalized(v_from)
    v_to = normalized(v_to)
    v_mid = normalized(v_from + v_to)
    q = np.array([np.dot(v_from, v_mid), *np.cross(v_from, v_mid)])
    return q

def normalized(v):
    norm = np.linalg.norm(v)
    return v / norm

NO_STATES = 13
IDX_POS_X = 0
IDX_POS_Y = 1
IDX_POS_Z = 2
IDX_VEL_X = 3
IDX_VEL_Y = 4
IDX_VEL_Z = 5
IDX_QUAT_W = 6
IDX_QUAT_X = 7
IDX_QUAT_Y = 8
IDX_QUAT_Z = 9
IDX_OMEGA_X = 10
IDX_OMEGA_Y = 11
IDX_OMEGA_Z = 12

class Robot:
    
    '''
    frames:
        B - body frame
        I - inertial frame
    states:
        p_I - position of the robot in the inertial frame (state[0], state[1], state[2])
        v_I - velocity of the robot in the inertial frame (state[3], state[4], state[5])
        q - orientation of the robot (w=state[6], x=state[7], y=state[8], z=state[9])
        omega - angular velocity of the robot (state[10], state[11], state[12])
    inputs:
        omega_1, omega_2, omega_3, omega_4 - angular velocities of the motors
    '''
    def __init__(self, phi_net : Phi_Net=None):
        self.m = 1.0 # mass of the robot
        self.arm_length = 0.25 # length of the quadcopter arm (motor to center)
        self.height = 0.05 # height of the quadcopter
        self.body_frame = np.array([(self.arm_length, 0, 0, 1),
                                    (0, self.arm_length, 0, 1),
                                    (-self.arm_length, 0, 0, 1),
                                    (0, -self.arm_length, 0, 1),
                                    (0, 0, 0, 1),
                                    (0, 0, self.height, 1)])

        self.J = 0.025 * np.eye(3) # [kg m^2]
        self.J_inv = np.linalg.inv(self.J)
        self.constant_thrust = 10e-4
        self.constant_drag = 10e-6
        self.state = self.reset_state_and_input(np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0]))
        self.time = 0.0
        self.a_adapt = np.array([0.0, 0.0, 0.0])

        # for storing the trajectory
        self.trajectory = []
        self.omegas_motors = []
        self.F_winds = []

        if phi_net is not None:
            self.phi_net = phi_net
            self.a_hat = np.zeros(self.phi_net.dim_a)
            self.P = np.eye(self.phi_net.dim_a) * 0.1
        

    def reset_state_and_input(self, init_xyz, init_quat_wxyz):
        state0 = np.zeros(NO_STATES)
        state0[IDX_POS_X:IDX_POS_Z+1] = init_xyz
        state0[IDX_VEL_X:IDX_VEL_Z+1] = np.array([0.0, 0.0, 0.0])
        state0[IDX_QUAT_W:IDX_QUAT_Z+1] = init_quat_wxyz
        state0[IDX_OMEGA_X:IDX_OMEGA_Z+1] = np.array([0.0, 0.0, 0.0])
        self.trajectory = []
        self.omegas_motors = []
        self.F_winds = []
        return state0

    def update(self, omegas_motor, dt, F_wind=np.zeros(3)):
        p_I = self.state[IDX_POS_X:IDX_POS_Z+1]
        v_I = self.state[IDX_VEL_X:IDX_VEL_Z+1]
        q = self.state[IDX_QUAT_W:IDX_QUAT_Z+1]
        omega = self.state[IDX_OMEGA_X:IDX_OMEGA_Z+1]
        R = scipy.spatial.transform.Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()

        thrust = self.constant_thrust * np.sum(omegas_motor**2)
        f_b = np.array([0, 0, thrust])

        
        tau_x = self.constant_thrust * (omegas_motor[3]**2 - omegas_motor[1]**2) * 2 * self.arm_length
        tau_y = self.constant_thrust * (omegas_motor[2]**2 - omegas_motor[0]**2) * 2 * self.arm_length
        tau_z = self.constant_drag * (omegas_motor[0]**2 - omegas_motor[1]**2 + omegas_motor[2]**2 - omegas_motor[3]**2)
        tau_b = np.array([tau_x, tau_y, tau_z])

        v_dot = 1 / self.m * (R @ f_b) + np.array([0, 0, -9.81]) + 1/self.m * F_wind
        omega_dot = self.J_inv @ (np.cross(self.J @ omega, omega) + tau_b)
        q_dot = 1 / 2 * quat_mult(q, [0, *omega])
        p_dot = v_I
        
        x_dot = np.concatenate([p_dot, v_dot, q_dot, omega_dot])
        self.state += x_dot * dt
        self.state[IDX_QUAT_W:IDX_QUAT_Z+1] /= np.linalg.norm(self.state[IDX_QUAT_W:IDX_QUAT_Z+1]) # Re-normalize quaternion.
        self.time += dt

        self.trajectory.append(self.state.copy())
        self.omegas_motors.append(omegas_motor.copy())
        self.F_winds.append(F_wind.copy())

    def control(self, p_d_I, v_d_I=None):
        '''
        This is the controller without the neural fly component. Implements an integral controller
        '''
        assert p_d_I.shape == (3,), p_d_I.shape
        assert v_d_I.shape == (3,), v_d_I.shape
        p_I = self.state[IDX_POS_X:IDX_POS_Z+1]
        v_I = self.state[IDX_VEL_X:IDX_VEL_Z+1]
        q = self.state[IDX_QUAT_W:IDX_QUAT_Z+1]
        omega_b = self.state[IDX_OMEGA_X:IDX_OMEGA_Z+1]

        # Position controller.
        k_p = 0.3
        k_d = 3.5
        k_I = 2
        
        if v_d_I is None:
            v_d_I = np.zeros(3)
        
        s = (v_I - v_d_I) + k_p * (p_I - p_d_I)
        a = -k_d * s + np.array([0, 0, 9.81]) - k_I * self.a_adapt
        self.a_adapt = self.a_adapt + s * dt

        f = self.m * a      # force in inertial frame
        f_b = scipy.spatial.transform.Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix().T @ f     # force in body frame
        thrust = np.max([0, f_b[2]])
        
        # Attitude controller.
        q_ref = quaternion_from_vectors(np.array([0, 0, 1]), normalized(f))
        q_err = quat_mult(quat_conjugate(q_ref), q) # error from Body to Reference.
        if q_err[0] < 0:
            q_err = -q_err
        k_q = 20.0
        k_omega = 100.0
        omega_ref = - k_q * 2 * q_err[1:]
        alpha = - k_omega * (omega_b - omega_ref)
        tau = self.J @ alpha + np.cross(omega_b, self.J @ omega_b) # + self.J @ omega_ref_dot
        
        # Compute the motor speeds.
        B = np.array([
            [self.constant_thrust, self.constant_thrust, self.constant_thrust, self.constant_thrust],
            [0, -self.arm_length * self.constant_thrust, 0, self.arm_length * self.constant_thrust],
            [-self.arm_length * self.constant_thrust, 0, self.arm_length * self.constant_thrust, 0],
            [self.constant_drag, -self.constant_drag, self.constant_drag, -self.constant_drag]
        ])
        B_inv = np.linalg.inv(B)
        omega_motor_square = B_inv @ np.concatenate([np.array([thrust]), tau])
        omega_motor = np.sqrt(np.clip(omega_motor_square, 0, None))
        return omega_motor

    def control_adapt(self, p_d_I, v_d_I=None):
        '''
        This is the controller with the neural fly component. implements the controller specified in the paper
        '''
        assert p_d_I.shape == (3,), p_d_I.shape
        assert v_d_I.shape == (3,), v_d_I.shape
        

        p_I = self.state[IDX_POS_X:IDX_POS_Z+1]
        v_I = self.state[IDX_VEL_X:IDX_VEL_Z+1]
        q = self.state[IDX_QUAT_W:IDX_QUAT_Z+1]
        assert np.isclose(np.linalg.norm(q), 1), f"Quaternion is not normalized: {np.linalg.norm(q)}"
        omega_b = self.state[IDX_OMEGA_X:IDX_OMEGA_Z+1]

        # Position controller.
        k_p = 0.3
        k_d = 3.5
        k_I = 2

        # adaptation parameters
        lamb = 0.9
        R_inv = np.eye(3)
        Q = np.eye(3)
        
        if v_d_I is None:
            v_d_I = np.zeros(3)
        
        s = (v_I - v_d_I) + k_p * (p_I - p_d_I)
        self.a_adapt = self.a_adapt + s * dt
        if len(self.omegas_motors) == 0:    # the first timestamp there's no omegas_motors
            phi_net_input = np.concatenate([self.state[IDX_VEL_X:IDX_VEL_Z+1], self.state[IDX_QUAT_W:IDX_QUAT_Z+1], np.zeros(4)])
        else:
            phi_net_input = np.concatenate([self.state[IDX_VEL_X:IDX_VEL_Z+1], self.state[IDX_QUAT_W:IDX_QUAT_Z+1], self.omegas_motors[-1]])
        phi = self.phi_net(torch.tensor(phi_net_input, dtype=torch.double)).detach().numpy()
        phi = np.diag(phi.flatten())
        self.a_hat = self.a_hat.reshape(3, 1)
        f = self.m * -k_d * s + np.array([0, 0, 9.81]) - k_I * self.a_adapt - (phi @ self.a_hat).reshape(-1)     # force in inertial frame
        f_b = scipy.spatial.transform.Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix().T @ f     # force in body frame
        thrust = np.max([0, f_b[2]])

        # Adaptation law
        self.a_hat = (self.a_hat).reshape(-1) + (- lamb * (self.a_hat).reshape(-1) - (self.P @ phi.T @ R_inv @ phi @ self.a_hat).reshape(-1) + self.P @ phi.T @ s) * dt
        self.P = self.P + (- 2 * lamb * self.P + Q  - self.P @ phi.T @ R_inv @ phi @ self.P) * dt
        
        # Attitude controller.
        q_ref = quaternion_from_vectors(np.array([0, 0, 1]), normalized(f))
        q_err = quat_mult(quat_conjugate(q_ref), q) # error from Body to Reference.
        if q_err[0] < 0:
            q_err = -q_err
        k_q = 20.0
        k_omega = 100.0
        omega_ref = - k_q * 2 * q_err[1:]
        alpha = - k_omega * (omega_b - omega_ref)
        tau = self.J @ alpha + np.cross(omega_b, self.J @ omega_b)
        
        # Compute the motor speeds.
        B = np.array([
            [self.constant_thrust, self.constant_thrust, self.constant_thrust, self.constant_thrust],
            [0, -self.arm_length * self.constant_thrust, 0, self.arm_length * self.constant_thrust],
            [-self.arm_length * self.constant_thrust, 0, self.arm_length * self.constant_thrust, 0],
            [self.constant_drag, -self.constant_drag, self.constant_drag, -self.constant_drag]
        ])
        B_inv = np.linalg.inv(B)
        omega_motor_square = B_inv @ np.concatenate([np.array([thrust]), tau])
        omega_motor = np.sqrt(np.clip(omega_motor_square, 0, None))
        return omega_motor

PLAYBACK_SPEED = 1
CONTROL_FREQUENCY = 200 # Hz for attitude control loop
dt = 1.0 / CONTROL_FREQUENCY
time = [0.0]

def get_reference_velocity(ref_trajectory : np.ndarray):
    return np.gradient(ref_trajectory, axis=0) / dt

def get_pos_full_quadcopter(quad : Robot):
    """ position returns a 3 x 6 matrix 
        where row is [x, y, z] column is m1 m2 m3 m4 origin h
    """
    origin = quad.state[IDX_POS_X:IDX_POS_Z+1]
    quat = quad.state[IDX_QUAT_W:IDX_QUAT_Z+1]
    rot = scipy.spatial.transform.Rotation.from_quat(quat, scalar_first=True).as_matrix()
    wHb = np.r_[np.c_[rot,origin], np.array([[0, 0, 0, 1]])]
    quadBodyFrame = quad.body_frame.T
    quadWorldFrame = wHb.dot(quadBodyFrame)
    pos_full_quad = quadWorldFrame[0:3]
    return pos_full_quad

def part_2_1():
    quad = Robot()
    sim_time = 10.0
    # def control_loop(i):
    #     t = quad.time
    #     T = 1.5
    #     r = 2*np.pi * t / T
    #     prop_thrusts = np.array([100, 0, 100, 0])
    #     quad.update(prop_thrusts, dt)
    #     return get_pos_full_quadcopter(quad)
    # plotter = QuadPlotter()
    # plotter.plot_animation(control_loop)
    for i in range(int(sim_time / dt)):
        t = quad.time
        T = 1.5
        r = 2*np.pi * t / T
        prop_thrusts = np.array([100, 0, 100, 0])
        quad.update(prop_thrusts, dt)
    
    trajectory = np.array(quad.trajectory)

    plt.plot(np.arange(0, sim_time, dt), trajectory[:,IDX_POS_Z], 'b-', label='z position')
    plt.plot(np.arange(0, sim_time, dt), trajectory[:, IDX_OMEGA_Z], 'r-', label='z angular velocity')
    plt.xlabel('Time (s)')
    plt.ylabel('Value')
    plt.title('z position and z angular velocity over time')
    plt.legend()
    plt.grid(True)
    plt.show()

def part_2_2():
    quad = Robot()
    sim_time = 10.0
    # def control_loop(i):
    #     t = quad.time
    #     T = 1.5
    #     F_wind = np.zeros(3)
    #     if t > 0.5:
    #         F_wind = np.array([2.0, 0, 0])

    #     prop_thrusts = quad.control(p_d_I = np.zeros(3), 
    #                                 v_d_I = np.zeros(3))
    #     quad.update(prop_thrusts, dt, F_wind)
    #     return get_pos_full_quadcopter(quad)
    # plotter = QuadPlotter()
    # plotter.plot_animation(control_loop)
    for i in range(int(sim_time / dt)):
        t = quad.time
        T = 1.5
        F_wind = np.zeros(3)
        if t > 0.5:
            F_wind = np.array([2.0, 0, 0])
        prop_thrusts = quad.control(p_d_I = np.zeros(3), 
                                    v_d_I = np.zeros(3))
        quad.update(prop_thrusts, dt, F_wind)
    
    trajectory = np.array(quad.trajectory)

    plt.plot(np.arange(0, sim_time, dt), trajectory[:, IDX_POS_X], 'b-', label='x position')
    # plt.plot(np.arange(0, sim_time, dt), trajectory[:, IDX_POS_Y], 'r-', label='y position')
    plt.xlabel('Time (s)')
    plt.ylabel('Value')
    plt.title('x position over time')
    plt.legend()
    plt.grid(True)
    plt.show()

def control_propellers_2_2(quad : Robot):
    t = quad.time
    T = 1.5
    r = 2*np.pi * t / T
    F_wind = np.zeros(3)
    if t > 1.0:
        F_wind = np.array([10.0, 0, 0])

    # # Part 2.2
    prop_thrusts = quad.control(p_d_I = np.array([0, 0, 0]))
    quad.update(prop_thrusts, dt, F_wind)

def run_simulation(quadcopter : Robot, ref_trajectory : np.ndarray, vel_trajectory : np.ndarray, sim_time : float, wind_direction : np.ndarray, amplitude : float, omega : float, phi : float, display_animation : bool = True):
    assert ref_trajectory.shape[1] == 3
    if display_animation:
        print("Displaying animation")
        def control_loop(i):
            print(i)
            if i > len(vel_trajectory):
                exit()
            F_wind = wind_direction * amplitude * np.cos(omega * quadcopter.time + phi)
            prop_thrusts = quadcopter.control(p_d_I = ref_trajectory[i], 
                                        # v_d_I = v_d_I[i]
                                        v_d_I = vel_trajectory[i]
                                        )
            quadcopter.update(prop_thrusts, dt, F_wind)
            return get_pos_full_quadcopter(quadcopter)
        plotter = QuadPlotter()
        plotter.plot_animation(control_loop)
    else:
        for i in range(int(sim_time / dt)):
            t = quadcopter.time
            F_wind = wind_direction * amplitude * np.cos(omega * t + phi)
            prop_thrusts = quadcopter.control(p_d_I = ref_trajectory[i], 
                                        v_d_I = vel_trajectory[i]
                                        # v_d_I = np.zeros(3)
                                        )
            quadcopter.update(prop_thrusts, dt, F_wind)

def run_simulation_NF(quadcopter : Robot, ref_trajectory : np.ndarray, vel_trajectory : np.ndarray, sim_time : float, wind_direction : np.ndarray, amplitude : float, omega : float, phi : float, display_animation : bool = True):
    assert ref_trajectory.shape[1] == 3
    if display_animation:
        print("Displaying animation")
        def control_loop(i):
            print(i)
            if i > len(vel_trajectory):
                exit()
            F_wind = wind_direction * amplitude * np.cos(omega * quadcopter.time + phi)
            prop_thrusts = quadcopter.control_adapt(p_d_I = ref_trajectory[i], 
                                        # v_d_I = v_d_I[i]
                                        v_d_I = vel_trajectory[i]
                                        )
            quadcopter.update(prop_thrusts, dt, F_wind)
            return get_pos_full_quadcopter(quadcopter)
        plotter = QuadPlotter()
        plotter.plot_animation(control_loop)
    else:
        for i in range(int(sim_time / dt)):
            t = quadcopter.time
            F_wind = wind_direction * amplitude * np.cos(omega * t + phi)
            prop_thrusts = quadcopter.control_adapt(p_d_I = ref_trajectory[i], 
                                        v_d_I = vel_trajectory[i]
                                        # v_d_I = np.zeros(3)
                                        )
            quadcopter.update(prop_thrusts, dt, F_wind)

def gather_training_data():
    data_dir = Path('/Users/yefan/Desktop/CDS245/simple-quad-sim/data')

    quadcopter = Robot()
    sim_time = 20.0
    num_directions = 5
    T = 3
    r = 2*np.pi * np.linspace(0, sim_time, int(sim_time / dt)) / T
    ref_trajectory = np.vstack([np.cos(r/2 - np.pi/2), np.sin(r), np.zeros(len(r))]).T
    vel_trajectory = get_reference_velocity(ref_trajectory)
    
    # Generate 5 random unit vectors
    wind_directions = np.random.randn(num_directions, 3)
    wind_directions /= np.linalg.norm(wind_directions, axis=1)[:, np.newaxis]
    
    amplitudes = [0, 0.5, 1, 1.5, 2]
    omegas = [0, 0.5, 1, 1.5, 2]
    phi = 0.0

    for wind_direction in tqdm(wind_directions):
        for amplitude in amplitudes:
            for omega in omegas:
                quadcopter.reset_state_and_input(np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0]))
                run_simulation(quadcopter, ref_trajectory, vel_trajectory, sim_time, wind_direction, amplitude, omega, phi, display_animation=False)
                trajectory = np.array(quadcopter.trajectory)
                omegas_motors = np.array(quadcopter.omegas_motors)
                F_winds = np.array(quadcopter.F_winds)
                np.save(data_dir / f"dir{wind_direction}_amp{amplitude}_omega{omega}_phi{phi}_trajectory.npy", trajectory)
                np.save(data_dir / f"dir{wind_direction}_amp{amplitude}_omega{omega}_phi{phi}_omegas_motors.npy", omegas_motors)
                np.save(data_dir / f"dir{wind_direction}_amp{amplitude}_omega{omega}_phi{phi}_F_winds.npy", F_winds)
                print(f"Saved data for dir{wind_direction}_amp{amplitude}_omega{omega}_phi{phi}")
    
    print("Done")
    

def main():
    '''
    everything here is mostly for testing, not used in main homework submission
    '''
    quadcopter = Robot()
    # plotter = QuadPlotter()
    # plotter.plot_animation(control_loop)
    sim_time = 10.0 
    T = 3
    r = 2*np.pi * np.linspace(0, sim_time, int(sim_time / dt)) / T
    ref_trajectory = np.vstack([np.cos(r/2 - np.pi/2), np.sin(r), np.zeros(len(r))]).T
    v_d_I = get_reference_velocity(ref_trajectory)
    # plt.figure()
    # plt.plot(ref_trajectory[:,0], ref_trajectory[:,1], 'r--', label='Reference Trajectory')
    # # plt.plot(np.diff(ref_trajectory[:,0], axis=0) / dt, np.diff(ref_trajectory[:,1], axis=0) / dt, 'b--', label='Reference Velocity')
    # plt.plot(vel_trajectory[:,0], vel_trajectory[:,1], 'b--', label='Reference Velocity')
    # # plt.plot(ref_trajectory[0,0], ref_trajectory[0,1], 'go', markersize=10, label='Start')
    # plt.plot(v_d_I[:,0], v_d_I[:,1], 'g-', label='Reference Velocity')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title('Reference Trajectory (Top View)')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # exit()
    wind_direction = np.array([1, 0, 0])
    amplitude = 0.0
    omega = 1.0
    phi = 0.0

    run_simulation(quadcopter, ref_trajectory, v_d_I, sim_time, wind_direction, amplitude, omega, phi, display_animation=False)

    trajectory = np.array(quadcopter.trajectory)
    print(trajectory.shape)
    print(trajectory[0:10,0], trajectory[0:10,1])
    plt.figure()
    plt.plot(ref_trajectory[:,0], ref_trajectory[:,1], 'r--', label='Reference Trajectory')
    plt.plot(trajectory[:,IDX_POS_X], trajectory[:,IDX_POS_Y], 'b-', label='Actual Trajectory')
    # plt.plot(ref_trajectory[0,0], ref_trajectory[0,1], 'go', markersize=10, label='Start')
    # plt.plot(ref_trajectory[-1,0], ref_trajectory[-1,1], 'ro', markersize=10, label='End')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Reference vs Actual Trajectory (Top View)')
    plt.legend()
    plt.grid(True)
    plt.show()

def part_4():
    model_path = Path('/Users/yefan/Desktop/CDS245/simple-quad-sim/models/epoch850.pth')
    phi_net = load_model(model_path)
    quadcopter1 = Robot(phi_net=phi_net)
    quadcopter2 = Robot()
    sim_time = 20.0
    T = 3
    r = 2*np.pi * np.linspace(0, sim_time, int(sim_time / dt)) / T
    ref_trajectory = np.vstack([np.cos(r/2 - np.pi/2), np.sin(r), np.zeros(len(r))]).T
    vel_trajectory = get_reference_velocity(ref_trajectory)

    wind_direction = np.array([1, 0, 0])
    amplitude = 2
    omega = 2
    phi = 0.0

    run_simulation_NF(quadcopter1, ref_trajectory, vel_trajectory, sim_time, wind_direction, amplitude, omega, phi, display_animation=False)
    trajectory1 = np.array(quadcopter1.trajectory)
    run_simulation(quadcopter2, ref_trajectory, vel_trajectory, sim_time, wind_direction, amplitude, omega, phi, display_animation=False)
    trajectory2 = np.array(quadcopter2.trajectory)
    plt.figure()
    plt.plot(ref_trajectory[:,0], ref_trajectory[:,1], 'r--', label='Reference Trajectory')
    plt.plot(trajectory2[:,IDX_POS_X], trajectory2[:,IDX_POS_Y], 'g-', label='Trajectory')
    plt.plot(trajectory1[:,IDX_POS_X], trajectory1[:,IDX_POS_Y], 'b-', label='NF Trajectory')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Reference vs Actual Trajectory (Top View), wind amp={amplitude}, omega={omega}')
    plt.legend()
    plt.grid(True)
    plt.show()



if __name__ == "__main__":
    # part_2_1()
    # part_2_2()
    part_4()
