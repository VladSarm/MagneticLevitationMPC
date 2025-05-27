import numpy as np
import casadi as ca
import threading
import time
from collections import deque
import pygame



class MPCBall:
    def __init__(self):
        # Initialize pygame
        pygame.init()
        self.width = 800
        self.height = 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Real-time MPC Bouncing Ball")
        self.clock = pygame.time.Clock()
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.GREEN = (0, 255, 0)
        self.BLACK = (0, 0, 0)
        self.GRAY = (128, 128, 128)
        
        # Ball properties (height in meters, 0 = bottom, 10 = top)
        self.ball_radius = 20
        self.ball_pos = np.array([5.0, 0.0])  # [height in meters, velocity in m/s]
        self.target_pos = 5.0  # target height in meters
        
        # Physics parameters
        self.dt = 0.01  # 20 FPS physics update
        self.g = 9.81   # gravity (m/s²)
        self.k = 50.0   # control gain
        self.eps = 0.1  # avoid division by zero
        self.max_height = 10.0  # max height in meters
        
        # MPC parameters
        self.N = 100  # horizon length
        self.Q1 = 500.0  # position weight
        self.Q2 = 80.0   # velocity weight
        self.R = 0.1     # control effort
        self.u_max = 10.0  # max control force
        
        # Setup MPC
        self.setup_mpc()
        
        # Control and simulation
        self.u_current = 0.0
        self.mpc_thread = None
        self.mpc_running = False
        self.mpc_lock = threading.Lock()
        
        # UI elements
        self.slider_rect = pygame.Rect(100, 50, 300, 20)
        self.slider_handle = pygame.Rect(100 + int(300 * self.target_pos / self.max_height), 45, 20, 30)
        self.dragging = False
        
        # History for plotting
        self.pos_history = deque(maxlen=200)
        self.target_history = deque(maxlen=200)
        self.control_history = deque(maxlen=200)
    

    def meters_to_pixels(self, height_m):
        """Convert height in meters to screen pixels (0m = bottom, 10m = top)"""
        return int(self.height - 100 - (height_m / self.max_height) * (self.height - 200))

    def pixels_to_meters(self, pixels):
        """Convert screen pixels to height in meters"""
        return (self.height - 100 - pixels) * self.max_height / (self.height - 200)
        
    def setup_mpc(self):
        """Setup CasADi MPC problem"""
        # State and control symbols
        X = ca.SX.sym('X', 2)  # [position, velocity]
        u = ca.SX.sym('u')     # control force
        x1, x2 = X[0], X[1]
        
        # Dynamics (magnetic levitation)
        xdot1 = x2
        xdot2 = -self.g + self.k * u**2 / ((x1 + self.eps)**2)
        
        # Discrete-time dynamics
        X_next = X + self.dt * ca.vertcat(xdot1, xdot2)
        
        # Dynamics function
        self.f = ca.Function('f', [X, u], [X_next])
        
        # MPC decision variables
        U = ca.SX.sym('U', self.N)
        Xh = ca.SX.sym('Xh', 2, self.N + 1)
        target = ca.SX.sym('target')  # target position parameter
        
        # Objective and constraints
        obj = 0
        g_constr = []
        
        # Initial condition constraint
        g_constr.append(Xh[:, 0] - X)
        
        # Horizon loop
        for k in range(self.N):
            # Dynamics constraint
            Xpred = self.f(Xh[:, k], U[k])
            g_constr.append(Xh[:, k + 1] - Xpred)
            
            # Stage cost
            obj += (self.Q1 * (Xh[0, k] - target)**2 + 
                   self.Q2 * Xh[1, k]**2 + 
                   self.R * U[k]**2)
        
        # Terminal cost
        obj += self.Q1 * (Xh[0, self.N] - target)**2 + self.Q2 * Xh[1, self.N]**2
        
        # Stack constraints
        g_v = ca.vertcat(*g_constr)
        
        # NLP formulation
        nlp = {
            'f': obj,
            'x': ca.vertcat(ca.reshape(Xh, -1, 1), U),
            'g': g_v,
            'p': ca.vertcat(X, target)
        }
        
        # Solver options
        opts = {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.max_iter': 20,
            'ipopt.warm_start_init_point': 'yes'
        }
        
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        self.n_vars = (self.N + 1) * 2 + self.N
        self.n_cons = (self.N + 1) * 2
        
    def solve_mpc(self, current_state, target):
        """Solve MPC problem"""
        try:
            # Initial guess
            x0_guess = np.tile(current_state.reshape(-1, 1), (1, self.N + 1)).flatten()
            u0_guess = np.ones(self.N)*0.1
            init_guess = np.concatenate([x0_guess, u0_guess])
            
            # Bounds
            lbw = np.concatenate([[-np.inf] * ((self.N + 1) * 2), [0] * self.N])
            ubw = np.concatenate([[np.inf] * ((self.N + 1) * 2), [self.u_max] * self.N])
            
            # Constraint bounds (all equality)
            lbg = np.zeros(self.n_cons)
            ubg = np.zeros(self.n_cons)
            
            # Solve
            sol = self.solver(
                x0=init_guess,
                lbx=lbw, ubx=ubw,
                lbg=lbg, ubg=ubg,
                p=np.concatenate([current_state, [target]])
            )
            
            # Extract first control action
            w_opt = sol['x'].full().flatten()
            u_opt = w_opt[(self.N + 1) * 2]  # First control
            
            return u_opt
            
        except Exception as e:
            print(f"MPC solver failed: {e}")
            return 0.0
    
    def mpc_worker(self):
        """MPC solver thread"""
        while self.mpc_running:
            with self.mpc_lock:
                current_state = self.ball_pos.copy()
                target = self.target_pos
            
            # Solve MPC
            u_new = self.solve_mpc(current_state, target)
            
            with self.mpc_lock:
                self.u_current = u_new
            
            time.sleep(self.dt)  # MPC update
    
    def update_physics(self):
        """Update ball physics"""
        with self.mpc_lock:
            u = self.u_current
        
        # Apply dynamics
        x1, x2 = self.ball_pos[0], self.ball_pos[1]
        
        # Magnetic levitation dynamics
        x1dot = x2
        x2dot = -self.g + self.k * u**2 / ((x1 + self.eps)**2)
        
        # Euler integration
        self.ball_pos[0] += self.dt * x1dot
        self.ball_pos[1] += self.dt * x2dot
        
        # Store history
        self.pos_history.append(self.ball_pos[0])
        self.target_history.append(self.target_pos)
        self.control_history.append(u)
    
    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if self.slider_rect.collidepoint(event.pos):
                    self.dragging = True
            
            elif event.type == pygame.MOUSEBUTTONUP:
                self.dragging = False
            
            elif event.type == pygame.MOUSEMOTION:
                if self.dragging:
                    # Update target position based on slider
                    rel_pos = max(0, min(1, (event.pos[0] - self.slider_rect.left) / self.slider_rect.width))
                    self.target_pos = rel_pos * self.max_height
                    self.slider_handle.centerx = self.slider_rect.left + int(rel_pos * self.slider_rect.width)
        
        return True
    
    def draw_magnet(self):
        """Draw electromagnetic magnet at the bottom"""
        magnet_width = 120
        magnet_height = 40
        magnet_x = self.width // 4 - magnet_width // 2
        magnet_y = self.height - magnet_height - 10
        
        # Magnet body (gray)
        pygame.draw.rect(self.screen, (100, 100, 100), 
                        (magnet_x, magnet_y, magnet_width, magnet_height))
        
        # Magnet poles (red and blue)
        pole_width = magnet_width // 2 - 5
        pygame.draw.rect(self.screen, (200, 0, 0), 
                        (magnet_x + 5, magnet_y + 5, pole_width, magnet_height - 10))
        pygame.draw.rect(self.screen, (0, 0, 200), 
                        (magnet_x + magnet_width // 2, magnet_y + 5, pole_width, magnet_height - 10))
        
        # Labels
        font = pygame.font.Font(None, 24)
        n_text = font.render("S", True, self.WHITE)
        s_text = font.render("N", True, self.WHITE)
        self.screen.blit(n_text, (magnet_x + 15, magnet_y + 12))
        self.screen.blit(s_text, (magnet_x + magnet_width - 25, magnet_y + 12))
        
        # Magnetic field indicators (size depends on control value)
        if abs(self.u_current) > 0.1:
            # Normalize control to 0-1 based on max control
            field_strength = min(abs(self.u_current) / self.u_max, 1.0)
            
            # Number of field lines depends on control strength (1-8 lines)
            num_lines = max(1, int(8 * field_strength))
            
            # Field extends up to ball height based on control strength
            max_field_height = field_strength * 200  # max 200 pixels up
            
            for i in range(num_lines):
                field_y = magnet_y - 10 - (i * max_field_height / num_lines)
                alpha = int(255 * field_strength * (1 - i * 0.1))
                if alpha > 20:  # minimum visibility
                    field_color = (0, 0, 0)
                    # Circle size also depends on strength
                    circle_radius = 8
                    pygame.draw.circle(self.screen, field_color, 
                                    (self.width // 4, int(field_y)), circle_radius)
    
    def draw(self):
        """Draw everything"""
        self.screen.fill(self.WHITE)
        
        # Draw magnet first
        self.draw_magnet()
        
        # Draw target line
        target_y = self.meters_to_pixels(self.target_pos)
        pygame.draw.line(self.screen, self.GREEN, 
                        (0, target_y), 
                        (self.width, target_y), 3)
        
        # Draw ball
        ball_screen_y = self.meters_to_pixels(self.ball_pos[0])
        pygame.draw.circle(self.screen, self.RED, 
                          (self.width // 4, ball_screen_y), 
                          self.ball_radius)
        
        # Draw height scale
        font_small = pygame.font.Font(None, 24)
        for h in range(0, 11, 1):
            y = self.meters_to_pixels(h)
            pygame.draw.line(self.screen, self.BLACK, (10, y), (20, y), 1)
            height_text = font_small.render(f"{h}m", True, self.BLACK)
            self.screen.blit(height_text, (25, y - 8))
        
        # Draw slider with label
        font = pygame.font.Font(None, 36)
        slider_label = font.render("Target Height (m):", True, self.BLACK)
        self.screen.blit(slider_label, (100, 20))
        
        pygame.draw.rect(self.screen, self.GRAY, self.slider_rect)
        pygame.draw.rect(self.screen, self.BLACK, self.slider_handle)
        
        # Draw text
        target_text = font.render(f"Target: {self.target_pos:.1f}m", True, self.BLACK)
        pos_text = font.render(f"Height: {self.ball_pos[0]:.1f}m", True, self.BLACK)
        vel_text = font.render(f"Velocity: {self.ball_pos[1]:.1f}m/s", True, self.BLACK)
        control_text = font.render(f"Control: {self.u_current:.1f}A", True, self.BLACK)
        
        self.screen.blit(target_text, (450, 50))
        self.screen.blit(pos_text, (450, 80))
        self.screen.blit(vel_text, (450, 110))
        self.screen.blit(control_text, (450, 140))
        
        # Draw wider history plots
        plot_start_x = self.width // 2
        plot_width = self.width // 2 - 50
        
        if len(self.pos_history) > 1:
            # Position history (blue)
            points = [(plot_start_x + int(i * plot_width / 100), 
                      self.meters_to_pixels(pos)) 
                     for i, pos in enumerate(list(self.pos_history)[-100:])]
            if len(points) > 1:
                pygame.draw.lines(self.screen, self.BLUE, False, points, 3)
            
            # Target history (green dashed line effect)
            target_points = [(plot_start_x + int(i * plot_width / 100), 
                            self.meters_to_pixels(target)) 
                           for i, target in enumerate(list(self.target_history)[-100:])]
            if len(target_points) > 1:
                for i in range(0, len(target_points) - 1, 3):
                    if i + 1 < len(target_points):
                        pygame.draw.line(self.screen, self.GREEN, 
                                       target_points[i], target_points[i + 1], 2)
        
        # Add plot labels
        plot_label = font_small.render("Position History", True, self.BLACK)
        self.screen.blit(plot_label, (plot_start_x, 200))
        
        pygame.display.flip()
    
    def run(self):
        """Main game loop"""
        # Start MPC thread
        self.mpc_running = True
        self.mpc_thread = threading.Thread(target=self.mpc_worker)
        self.mpc_thread.daemon = True
        self.mpc_thread.start()
        
        running = True
        while running:
            running = self.handle_events()
            self.update_physics()
            self.draw()
            self.clock.tick(60)  # 60 FPS display
        
        # Cleanup
        self.mpc_running = False
        if self.mpc_thread and self.mpc_thread.is_alive():
            self.mpc_thread.join(timeout=1.0)
        
        pygame.quit()

if __name__ == "__main__":
    game = MPCBall()
    game.run()