# A simple, single-file dog daycare simulation for architecture portfolios.
# This code demonstrates agent-based simulation with a live visualization feature
# and an interactive mode to draw the simulation boundary.
#
# To run this script, you will need to install the following libraries:
# pip install numpy matplotlib shapely seaborn

# === Imports ===
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from shapely.geometry import Point, Polygon
from shapely import affinity # For scaling, although not used here, it's a good import to keep
# No longer importing geopandas to keep the build simple.
import seaborn as sns 
import warnings
import sys

# === Configuration Parameters ===
# The daycare space dimensions.
DAYCARE_BOUNDS = (0, 0, 150, 100)  # [min_x, min_y, max_x, max_y]
NUM_DOGS = 25
NUM_SIMULATION_STEPS = 500

# === Simulation Physics & Behavior ===
BOUNDARY_REPULSION_STRENGTH = 0.5   
CENTER_PULL_STRENGTH = 0.1          
RANDOM_WALK_STEP_SIZE = 0.5         
MAX_ANGLE_CHANGE = math.pi / 4      

# === Visualization Settings ===
DRAW_CUSTOM_BOUNDARY = True # Set to True to draw a new boundary, False to use DAYCARE_BOUNDS
RUN_LIVE_VISUALIZATION = True
UPDATE_INTERVAL_MS = 50

# --- Global variable to store points from interactive drawing ---
drawn_points = None
daycare_shape = None # Initialize daycare_shape globally

# === Helper Functions ===
def generate_random_point_in_shape(shape, max_tries=10000):
    """Generates a random point inside a given Shapely shape."""
    minx, miny, maxx, maxy = shape.bounds
    for _ in range(max_tries):
        p = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if p.within(shape):
            return np.array([p.x, p.y], dtype=float)
    raise TimeoutError(f"Could not find a random point in shape after {max_tries} attempts.")

def initialize_dogs(num_dogs, daycare_shape):
    """Creates a list of dog agents with random initial positions inside the given shape."""
    dogs = []
    for i in range(num_dogs):
        try:
            pos = generate_random_point_in_shape(daycare_shape)
            dogs.append({
                "id": i,
                "pos": pos,
                "last_angle": random.uniform(0, 2 * math.pi),
                "history": [tuple(pos)] # History for final heatmap plotting
            })
        except TimeoutError as e:
            print(f"Warning: {e}. Initialized {len(dogs)} dogs instead of {num_dogs}.")
            break
    return dogs

def normalize(v):
    """Normalizes a vector to a unit vector."""
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v

def simulate_step(dogs, daycare_shape, params):
    """Applies forces and updates the position of each dog for one step."""
    shape_centroid = np.array([daycare_shape.centroid.x, daycare_shape.centroid.y])

    for dog in list(dogs):
        pos = dog["pos"]
        net_force = np.zeros(2)

        # 1. Boundary Repulsion Force
        distance_to_boundary = Point(pos).distance(daycare_shape.exterior)
        if distance_to_boundary < 5:
            closest_pt_on_boundary = daycare_shape.exterior.interpolate(daycare_shape.exterior.project(Point(pos)))
            vector_from_boundary = pos - np.array([closest_pt_on_boundary.x, closest_pt_on_boundary.y])
            if np.linalg.norm(vector_from_boundary) > 1e-6:
                net_force += normalize(vector_from_boundary) * params['boundary_repulsion_strength'] * (5 - distance_to_boundary)
            else: # Fallback for edge cases where dog is exactly on the boundary
                net_force -= normalize(shape_centroid - pos) * params['boundary_repulsion_strength'] * 0.5
        # 2. Center Pull Force
        vector_to_center = shape_centroid - pos
        net_force += normalize(vector_to_center) * params['center_pull_strength']

        # 3. Random Walk Movement
        angle_change_val = random.uniform(-params['max_angle_change'], params['max_angle_change'])
        new_angle = (dog["last_angle"] + angle_change_val) % (2 * math.pi)
        random_vector = np.array([math.cos(new_angle), math.sin(new_angle)]) * params['random_walk_step_size']
        dog["last_angle"] = new_angle

        move_vector = random_vector + net_force
        potential_new_pos = pos + move_vector

        potential_point = Point(potential_new_pos)
        if daycare_shape.contains(potential_point):
            dog["pos"] = potential_new_pos
        else:
            projected_point = daycare_shape.exterior.interpolate(daycare_shape.exterior.project(potential_point))
            dog["pos"] = np.array([projected_point.x, projected_point.y])
            
        dog["history"].append(tuple(dog["pos"]))

    return dogs

def plot_final_results(dogs, daycare_shape):
    """Visualizes the simulation results with a final heatmap and dog positions."""
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Use Matplotlib directly to plot the shape
    shape_x, shape_y = daycare_shape.exterior.xy
    ax.plot(shape_x, shape_y, color='black', linewidth=2, label='Daycare Boundary', zorder=0, alpha=0.5)

    all_positions = [pos for dog in dogs for pos in dog["history"]]
    if all_positions:
        x_coords, y_coords = zip(*all_positions)
        try:
            sns.kdeplot(x=x_coords, y=y_coords, cmap="magma", fill=True, bw_adjust=1.0, ax=ax, zorder=1)
        except ImportError:
            print("Seaborn not found. Skipping heatmap.")
    
    final_x = [d["pos"][0] for d in dogs]
    final_y = [d["pos"][1] for d in dogs]
    ax.scatter(final_x, final_y, s=35, alpha=0.9, color='blue', edgecolor='black', zorder=2, label='Final Dog Positions')

    ax.set_title("Dog Daycare Simulation (Final State)", fontsize=16)
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(daycare_shape.bounds[0]-5, daycare_shape.bounds[2]+5)
    ax.set_ylim(daycare_shape.bounds[1]-5, daycare_shape.bounds[3]+5)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

def draw_boundary(ax):
    """
    A function to interactively draw a polygon on a Matplotlib plot.
    Returns the list of points defining the polygon.
    """
    print("Click on the plot to define the vertices of your custom daycare boundary.")
    print("Press 'Enter' to finalize the polygon.")
    print("Press 'Escape' to cancel.")
    
    points = []
    line, = ax.plot([], [], 'r-o', markersize=5) 
    
    global drawn_points
    drawn_points = None

    def onclick(event):
        if event.button == 1:
            points.append((event.xdata, event.ydata))
            x, y = zip(*points)
            line.set_data(x, y)
            plt.draw()

    def onkey(event):
        nonlocal points
        global drawn_points
        if event.key == 'enter' and len(points) >= 3:
            drawn_points = points
            x, y = zip(*points)
            line.set_data(x + (x[0],), y + (y[0],))
            plt.draw()
            plt.close()
        elif event.key == 'escape':
            drawn_points = None
            plt.close()

    cid1 = plt.connect('button_press_event', onclick)
    cid2 = plt.connect('key_press_event', onkey)
    
    plt.show()
    return drawn_points

# === Main Execution ===
if __name__ == "__main__":
    if DRAW_CUSTOM_BOUNDARY:
        fig_draw, ax_draw = plt.subplots(figsize=(12, 9))
        ax_draw.set_title("Draw Your Custom Daycare Boundary")
        ax_draw.set_xlabel("X Coordinate")
        ax_draw.set_ylabel("Y Coordinate")
        ax_draw.grid(True, linestyle='--', alpha=0.5)
        ax_draw.set_xlim(DAYCARE_BOUNDS[0], DAYCARE_BOUNDS[2])
        ax_draw.set_ylim(DAYCARE_BOUNDS[1], DAYCARE_BOUNDS[3])
        
        custom_points = draw_boundary(ax_draw)

        if custom_points and len(custom_points) >= 3:
            # Add this line to attempt to fix invalid geometries
            daycare_shape = Polygon(custom_points).buffer(0)
            if not daycare_shape.is_valid:
                print("Warning: The drawn shape is invalid. Using default bounds.")
                daycare_shape = Polygon([
                    (DAYCARE_BOUNDS[0], DAYCARE_BOUNDS[1]),
                    (DAYCARE_BOUNDS[2], DAYCARE_BOUNDS[1]),
                    (DAYCARE_BOUNDS[2], DAYCARE_BOUNDS[3]),
                    (DAYCARE_BOUNDS[0], DAYCARE_BOUNDS[3])
                ])
        else:
            print("Not enough points drawn. Using default bounds.")
            daycare_shape = Polygon([
                (DAYCARE_BOUNDS[0], DAYCARE_BOUNDS[1]),
                (DAYCARE_BOUNDS[2], DAYCARE_BOUNDS[1]),
                (DAYCARE_BOUNDS[2], DAYCARE_BOUNDS[3]),
                (DAYCARE_BOUNDS[0], DAYCARE_BOUNDS[3])
            ])
    else:
        daycare_shape = Polygon([
            (DAYCARE_BOUNDS[0], DAYCARE_BOUNDS[1]),
            (DAYCARE_BOUNDS[2], DAYCARE_BOUNDS[1]),
            (DAYCARE_BOUNDS[2], DAYCARE_BOUNDS[3]),
            (DAYCARE_BOUNDS[0], DAYCARE_BOUNDS[3])
        ])

    sim_params = {
        "boundary_repulsion_strength": BOUNDARY_REPULSION_STRENGTH,
        "center_pull_strength": CENTER_PULL_STRENGTH,
        "random_walk_step_size": RANDOM_WALK_STEP_SIZE,
        "max_angle_change": MAX_ANGLE_CHANGE,
    }

    dogs = initialize_dogs(NUM_DOGS, daycare_shape) # Pass the correct daycare shape to initialize dogs
    print(f"Starting simulation with {len(dogs)} dogs for {NUM_SIMULATION_STEPS} steps...")
    
    if RUN_LIVE_VISUALIZATION:
        print("Live Visualization Mode: A plot window will open and update in real time.")
        fig, ax = plt.subplots(figsize=(12, 9))
        
        scatter = ax.scatter([], [], color='blue', edgecolor='black', zorder=2, label="Dogs")

        # Plot the static elements of the plot once
        shape_x, shape_y = daycare_shape.exterior.xy
        ax.plot(shape_x, shape_y, color='black', linewidth=2, zorder=0, alpha=0.5, label='Daycare Boundary')

        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(daycare_shape.bounds[0]-5, daycare_shape.bounds[2]+5)
        ax.set_ylim(daycare_shape.bounds[1]-5, daycare_shape.bounds[3]+5)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        def update(frame):
            global dogs
            dogs = simulate_step(dogs, daycare_shape, sim_params)
            
            x_coords = [d["pos"][0] for d in dogs]
            y_coords = [d["pos"][1] for d in dogs]
            
            scatter.set_offsets(np.c_[x_coords, y_coords])
            ax.set_title(f"Dog Daycare Simulation (Live) - Step {frame + 1}/{NUM_SIMULATION_STEPS}")
            return scatter,

        ani = animation.FuncAnimation(fig, update, frames=NUM_SIMULATION_STEPS, interval=UPDATE_INTERVAL_MS, repeat=False, blit=False)
        plt.show()

    else:
        for step in range(NUM_SIMULATION_STEPS):
            dogs = simulate_step(dogs, daycare_shape, sim_params)
        
        print("Batch Mode: Simulation finished. Generating final plot...")
        plot_final_results(dogs, daycare_shape)