import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.patches import Circle, Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.widgets import Button  
from datetime import datetime, timedelta
import random
from collections import defaultdict
import tkinter as tk
from tkinter import ttk, scrolledtext
import matplotlib.gridspec as gridspec
import sys
import networkx as nx

def get_simulation_parameters():
    # Default parameters
    default_params = {
        'NUM_PEOPLE': 100,
        'WIDTH': 100,
        'HEIGHT': 100,
        'INFECTION_RADIUS': 5.0,
        'INFECTION_PROB': 0.15,
        'MASK_EFFECTIVENESS': 0.7,
        'VACCINE_EFFECTIVENESS': 0.85,
        'RECOVERY_DAYS': 7,
        'SEVERITY_THRESHOLD': 0.6,
        'TIME_PER_FRAME_MINUTES': 2,
        'MASK_RATE': 0.7,
        'VACCINE_RATE': 0.6,
        'ISOLATION_RATE': 0.8,
        'LOCKDOWN_THRESHOLD': 0.5
    }

    result = {'params': None}  # Use a dictionary to store the result

    def create_parameter_window():
        root = tk.Tk()
        root.title("Epidemic Simulation Parameters")
        root.geometry("600x800")

        # Create main frame with scrollbar
        main_frame = ttk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Style
        style = ttk.Style()
        style.configure("TLabel", padding=5)
        style.configure("TEntry", padding=5)
        style.configure("TButton", padding=5)

        # Title
        title_label = ttk.Label(scrollable_frame, 
                              text="Epidemic Simulation Parameters",
                              font=("Helvetica", 16, "bold"))
        title_label.pack(pady=10)

        # Parameter entries
        entries = {}
        row = 0
        for param, value in default_params.items():
            frame = ttk.Frame(scrollable_frame)
            frame.pack(fill=tk.X, pady=2)
            
            label = ttk.Label(frame, text=param.replace('_', ' ').title(), width=30)
            label.pack(side=tk.LEFT, padx=5)
            
            entry = ttk.Entry(frame, width=20)
            entry.insert(0, str(value))
            entry.pack(side=tk.LEFT, padx=5)
            entries[param] = entry

        # Buttons frame
        button_frame = ttk.Frame(scrollable_frame)
        button_frame.pack(pady=20)

        def use_defaults():
            for param, entry in entries.items():
                entry.delete(0, tk.END)
                entry.insert(0, str(default_params[param]))

        def start_simulation():
            try:
                params = {}
                for param, entry in entries.items():
                    value = entry.get()
                    if param in ['NUM_PEOPLE', 'WIDTH', 'HEIGHT', 'RECOVERY_DAYS']:
                        params[param] = int(value)
                    else:
                        params[param] = float(value)
                result['params'] = params  # Store the result
                root.destroy()
            except ValueError as e:
                tk.messagebox.showerror("Error", f"Invalid input: {str(e)}")

        ttk.Button(button_frame, text="Use Defaults", command=use_defaults).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Start Simulation", command=start_simulation).pack(side=tk.LEFT, padx=5)

        # Pack scrollbar and canvas
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        root.mainloop()
        return result['params']  # Return the stored result

    # Get parameters from user
    params = create_parameter_window()
    if params is None:
        print("Using default parameters...")
        return default_params
    return params

# Get simulation parameters
params = get_simulation_parameters()
print("Using parameters:", params)  # Debug print to verify parameters

# Simulation parameters
NUM_PEOPLE = params['NUM_PEOPLE']
WIDTH, HEIGHT = params['WIDTH'], params['HEIGHT']
INFECTION_RADIUS = params['INFECTION_RADIUS']
INFECTION_PROB = params['INFECTION_PROB']
MASK_EFFECTIVENESS = params['MASK_EFFECTIVENESS']
VACCINE_EFFECTIVENESS = params['VACCINE_EFFECTIVENESS']
RECOVERY_DAYS = params['RECOVERY_DAYS']
SEVERITY_THRESHOLD = params['SEVERITY_THRESHOLD']
TIME_PER_FRAME = timedelta(minutes=params['TIME_PER_FRAME_MINUTES'])
MASK_RATE = params['MASK_RATE']
VACCINE_RATE = params['VACCINE_RATE']
ISOLATION_RATE = params['ISOLATION_RATE']
LOCKDOWN_THRESHOLD = 0.5  # Set to 50%

current_time = datetime(2023, 1, 1, 6, 0)
lockdown = False

HOME_AREA = (0, 50, 50, 100)
WORK_AREA = (50, 50, 100, 100)
RECREATIONAL_AREA = (0, 0, 50, 50)  # Formerly PARK_AREA
SCHOOL_AREA = (50, 0, 100, 50)

# Spatial partitioning grid for efficient collision detection
GRID_SIZE = 10  # Smaller grid size for more precise partitioning
grid = defaultdict(list)

class EventLogger:
    def __init__(self, max_events=100):
        self.events = []
        self.max_events = max_events
        self.area_counts = {
            'HOME': 0,
            'WORK': 0,
            'SCHOOL': 0,
            'RECREATIONAL': 0
        }
        self.state_counts = {
            'SUSCEPTIBLE': 0,
            'INFECTED': 0,
            'RECOVERED': 0,
            'QUARANTINED': 0
        }
        self.protection_counts = {
            'MASKED': 0,
            'VACCINATED': 0
        }
        self.susceptible_history = []
        self.infection_history = []
        self.quarantined_history = []
        self.recovery_history = []
        self.time_history = []

    def log_event(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.events.append(f"[{timestamp}] {message}")
        if len(self.events) > self.max_events:
            self.events.pop(0)

    def update_counts(self, people):
        # Reset counts
        for key in self.area_counts:
            self.area_counts[key] = 0
        for key in self.state_counts:
            self.state_counts[key] = 0
        for key in self.protection_counts:
            self.protection_counts[key] = 0

        # Update counts
        for person in people:
            if person.current_area:
                self.area_counts[person.current_area] += 1
            
            if person.state == 0:
                self.state_counts['SUSCEPTIBLE'] += 1
            elif person.state == 1:
                self.state_counts['INFECTED'] += 1
                if person.in_quarantine:
                    self.state_counts['QUARANTINED'] += 1
            else:
                self.state_counts['RECOVERED'] += 1
            
            if person.masked:
                self.protection_counts['MASKED'] += 1
            if person.vaccinated:
                self.protection_counts['VACCINATED'] += 1
        
        self.susceptible_history.append(self.state_counts['SUSCEPTIBLE'])
        self.infection_history.append(self.state_counts['INFECTED'])
        self.quarantined_history.append(self.state_counts['QUARANTINED'])
        self.recovery_history.append(self.state_counts['RECOVERED'])
        self.time_history.append(current_time)

    def get_status_text(self):
        status = "=== Current Status ===\n"
        
        # States
        status += "States:\n"
        for state, count in self.state_counts.items():
            status += f"{state}: {count}\n"
        
        # Areas
        status += "\nAreas:\n"
        for area, count in self.area_counts.items():
            status += f"{area}: {count}\n"
        
        # Protection
        status += "\nProtection:\n"
        for protection, count in self.protection_counts.items():
            status += f"{protection}: {count}\n"
        
        # Recent Events (show only last 3)
        status += "\nRecent Events:\n"
        for event in self.events[-3:]:
            status += f"{event}\n"
        
        return status

# Create the event logger
event_logger = EventLogger()

class Person:
    def __init__(self):
        self.age = random.randint(5, 80)
        self.home = self._random_point_in_area(HOME_AREA)
        self.workplace = self._random_point_in_area(WORK_AREA)
        self.school = self._random_point_in_area(SCHOOL_AREA)
        self.x, self.y = self.home  # Initialize x and y coordinates to home position
        self.state = 0  # 0=SUSCEPTIBLE, 1=INFECTED, 2=RECOVERED
        self.masked = random.random() < MASK_RATE
        self.vaccinated = random.random() < VACCINE_RATE
        self.days_infected = 0
        self.infection_severity = random.random()
        self.size = 30
        self.current_target = None
        self.time_in_area = 0
        self.max_area_time = random.randint(100, 300)
        self.speed = random.uniform(0.5, 1.2)
        self.daily_routine = self._create_daily_routine()
        self.routine_index = 0
        self.target_reached = True
        self.assign_initial_target()
        self.will_isolate = random.random() < ISOLATION_RATE
        self.in_quarantine = False
        self.grid_cell = None
        self.current_area = None

    def _random_point_in_area(self, area, max_attempts=100):
        for _ in range(max_attempts):
            x = random.uniform(area[0], area[2])
            y = random.uniform(area[1], area[3])
            if not (RECREATIONAL_AREA[0] <= x <= RECREATIONAL_AREA[2] and RECREATIONAL_AREA[1] <= y <= RECREATIONAL_AREA[3]):
                return (x, y)
        return ((area[0]+area[2])/2, (area[1]+area[3])/2)

    def in_area(self, area):
        x1, y1, x2, y2 = area
        return x1 <= self.x <= x2 and y1 <= self.y <= y2

    def _create_daily_routine(self):
        if self.age < 18:
            return [(6, 7, self.home), (7, 8, self.school), (8, 15, self.school),
                    (15, 16, self.home), (16, 18, self.home), (18, 6, self.home)]
        else:
            return [(6, 7, self.home), (7, 8, self.workplace), (8, 17, self.workplace),
                    (17, 18, self.home), (18, 20, None), (20, 6, self.home)]

    def assign_initial_target(self):
        self.current_target = self.home
        self.target_reached = True

    def update_target(self, current_time):
        # If in lockdown, force everyone to stay at home
        if lockdown:
            self.current_target = self.home
            self.target_reached = False
            return

        if self.in_quarantine:
            self.current_target = self.home
            self.target_reached = False
            return

        if self.state == 1 and self.will_isolate and self.infection_severity > SEVERITY_THRESHOLD:
            self.in_quarantine = True
            self.current_target = self.home
            self.target_reached = False
            return

        hour = current_time.hour + current_time.minute / 60

        # Prevent any school visits after 3pm for everyone
        if hour >= 15 and self.current_area == 'SCHOOL':
            self.current_target = self.home
            self.target_reached = False
            return

        for i, (start, end, target) in enumerate(self.daily_routine):
            if start <= hour < end or (start > end and (hour >= start or hour < end)):
                if self.routine_index != i or self.target_reached:
                    self.routine_index = i
                    # Additional check to prevent school visits after 3pm for everyone
                    if target == self.school and hour >= 15:
                        self.current_target = self.home
                    else:
                        self.current_target = target
                    self.target_reached = False
                    self.time_in_area = 0
                break

        if 16 <= hour < 18 and random.random() < 0.3:
            self.current_target = self._random_point_in_area(RECREATIONAL_AREA)
            self.target_reached = False
            self.time_in_area = 0

    def update_current_area(self):
        if self.in_area(HOME_AREA):
            self.current_area = 'HOME'
        elif self.in_area(WORK_AREA):
            self.current_area = 'WORK'
        elif self.in_area(SCHOOL_AREA):
            self.current_area = 'SCHOOL'
        elif self.in_area(RECREATIONAL_AREA):
            self.current_area = 'RECREATIONAL'
        else:
            self.current_area = None

    def move(self, current_time):
        # Remove from current grid cell
        if self.grid_cell is not None and self in grid[self.grid_cell]:
            grid[self.grid_cell].remove(self)
            
        # If in quarantine, force stay at home with no movement
        if self.in_quarantine:
            self.x, self.y = self.home
            self.current_target = self.home
            self.target_reached = True
            self.time_in_area = 0
            # Update grid position
            new_cell = (int(self.x / GRID_SIZE), int(self.y / GRID_SIZE))
            if new_cell != self.grid_cell:
                if self.grid_cell is not None and self in grid[self.grid_cell]:
                    grid[self.grid_cell].remove(self)
                self.grid_cell = new_cell
                grid[self.grid_cell].append(self)
            self.update_current_area()
            return

        self.update_target(current_time)

        if self.current_target is None:
            if random.random() < 0.05:
                self.current_target = (random.uniform(0, WIDTH), random.uniform(0, HEIGHT))
                self.target_reached = False
        elif not self.target_reached:
            tx, ty = self.current_target
            dx, dy = tx - self.x, ty - self.y
            distance = np.hypot(dx, dy)

            if distance < 1:
                self.target_reached = True
                self.time_in_area += 1
            else:
                speed_modifier = 0.7 if self.state == 1 else 1.0
                self.x += (dx / distance) * self.speed * speed_modifier
                self.y += (dy / distance) * self.speed * speed_modifier
        else:
            self.time_in_area += 1
            if self.time_in_area > self.max_area_time and random.random() < 0.1:
                self.target_reached = False

        # Ensure person stays within bounds
        self.x = np.clip(self.x, 0, WIDTH)
        self.y = np.clip(self.y, 0, HEIGHT)
        
        # Update grid position
        new_cell = (int(self.x / GRID_SIZE), int(self.y / GRID_SIZE))
        if new_cell != self.grid_cell:
            if self.grid_cell is not None and self in grid[self.grid_cell]:
                grid[self.grid_cell].remove(self)
            self.grid_cell = new_cell
            grid[self.grid_cell].append(self)

        self.update_current_area()

# Initialize
people = [Person() for _ in range(NUM_PEOPLE)]

# Set 10% of population as initially infected
initial_infected = int(NUM_PEOPLE * 0.1)
for p in random.sample(people, initial_infected):
    p.state = 1

# Visualization setup
fig = plt.figure(figsize=(18, 12))

# Create main grid for the entire figure
if NUM_PEOPLE <= 500:
    # Original layout with all three panels when population is small
    gs_main = fig.add_gridspec(1, 3, width_ratios=[5, 1, 3], wspace=0.05)
else:
    # Modified layout for large populations - give more space to status and disease graph
    gs_main = fig.add_gridspec(1, 3, width_ratios=[7, 3, 0], wspace=0.1)

gs_left = gridspec.GridSpecFromSubplotSpec(
    2, 1,
    subplot_spec=gs_main[0, 0],
    height_ratios=[15, 1],  # Simulation gets 15x height, legend just 1x
    hspace=0.05
)

# Simulation Panel
ax1 = fig.add_subplot(gs_left[0])
ax1.set_xlim(0, WIDTH)
ax1.set_ylim(0, HEIGHT)
ax1.axis('off')

#  Legend Panel directly below simulation
ax_legend = fig.add_subplot(gs_left[1])
ax_legend.clear()
ax_legend.set_xlim(0, 1)
ax_legend.set_ylim(0, 1)
ax_legend.set_aspect('auto') 
ax_legend.axis('off')

# Define legend items
legend_items = [
    ("Healthy", '#1f77b4', False),
    ("Vaccinated", '#17becf', False),
    ("Infected", (1.0, 0.1, 0.1), False),
    ("Quarantined", '#800080', False),
    ("Recovered", '#2ca02c', False),
    ("Masked", '#1f77b4', True),  # Base blue + purple mask overlay
]

# Coordinates for layout
x_start = 0.05
x_spacing = 0.15
y_center = 0.5
radius = 0.025

for i, (label, color, is_masked) in enumerate(legend_items):
    x = x_start + i * x_spacing

    # Main circle
    circle = plt.Circle((x, y_center), radius, color=color, zorder=1)
    ax_legend.add_patch(circle)

    if is_masked:
        # Overlay a smaller purple mask circle (just like in main simulation)
        mask_circle = plt.Circle((x, y_center), radius * 0.7, color='#800080', zorder=2)
        ax_legend.add_patch(mask_circle)

    # Label text
    ax_legend.text(x + 0.035, y_center, label, va='center', ha='left', fontsize=9)

if NUM_PEOPLE <= 500:
    # Panel B: Status/Control/Stats (middle)
    ax_status = fig.add_subplot(gs_main[0, 1])
    ax_status.axis('off')
    
    # Panel C: Right side container (network + disease graph)
    gs_right = gridspec.GridSpecFromSubplotSpec(
        2, 1,
        subplot_spec=gs_main[0, 2],
        height_ratios=[1.4, 0.6],
        hspace=0.25
    )
    
    # Network Graph (top right)
    ax_network = fig.add_subplot(gs_right[0])
    ax_network.axis('off')

        # Create NetworkX graph
    G = nx.Graph()
    for i in range(NUM_PEOPLE):
        G.add_node(i)
    pos = nx.spring_layout(G, k=0.5, iterations=50)

    def show_full_network(event):
     if G is None or pos is None:
        print("Network graph not available for large population.")
        return

     fig_full, ax_full = plt.subplots(figsize=(10, 10))
     ax_full.set_title("Full Social Network", fontsize=16)

     ax_full.clear()
     node_colors = []
     for p in people:
        if p.state == 0:
            node_colors.append('#1f77b4' if not p.vaccinated else '#17becf')
        elif p.state == 1:
            severity = min(0.8 + p.infection_severity * 0.2, 1.0)
            node_colors.append((severity, 0.1, 0.1))  # Red shade
        else:
            node_colors.append('#2ca02c')  # Green for recovered

    # Draw the updated graph with latest colors and edges
     nx.draw_networkx(
        G, pos=pos, ax=ax_full,
        node_color=node_colors,
        with_labels=False,
        node_size=75,
        edge_color='gray',
        alpha=0.7,
        width=0.6
     )
     ax_full.set_xlim([min(x for x, y in pos.values()) - 0.1, max(x for x, y in pos.values()) + 0.1])
     ax_full.set_ylim([min(y for x, y in pos.values()) - 0.1, max(y for x, y in pos.values()) + 0.1])

# Ensure the aspect ratio is equal to prevent stretching
    
     plt.draw()
     plt.show()
    
    #the button 
    button_ax = fig.add_axes([0.75, 0.92, 0.12, 0.045])  # Adjust position as needed
    full_view_button = Button(button_ax, 'Full Network', color='lightgray', hovercolor='skyblue')
    full_view_button.on_clicked(show_full_network)



    # Disease Spread Graph (bottom right)
    ax3 = fig.add_subplot(gs_right[1])
    ax3.set_xlabel("Time", fontsize=12)
    ax3.set_ylabel("Number of People", fontsize=12)
    ax3.grid(True)
    ax3.tick_params(axis='both', which='major', labelsize=10)
    

else:
    # For larger populations - combine status and disease graph in one panel
    gs_right = gridspec.GridSpecFromSubplotSpec(
        2, 1,
        subplot_spec=gs_main[0, 1],
        height_ratios=[1, 1],
        hspace=0.25
    )
    
    # Status panel (top)
    ax_status = fig.add_subplot(gs_right[0])
    ax_status.axis('off')
    
    # Disease Spread Graph (bottom)
    ax3 = fig.add_subplot(gs_right[1])
    ax3.set_xlabel("Time", fontsize=12)
    ax3.set_ylabel("Number of People", fontsize=12)
    ax3.grid(True)
    ax3.tick_params(axis='both', which='major', labelsize=10)

    
    # No network graph for large populations
    G = None
    pos = None

# Initialize disease spread lines (common for both cases)
susceptible_line, = ax3.plot([], [], 'b-', label='Susceptible')
infected_line, = ax3.plot([], [], 'r-', label='Infected')
quarantined_line, = ax3.plot([], [], 'm-', label='Quarantined') 
recovered_line, = ax3.plot([], [], 'g-', label='Recovered')
ax3.legend(loc='upper right', fontsize=10)

# Adjust subplot spacing
plt.subplots_adjust(
    left=0.05,
    right=0.95,
    top=0.95,
    bottom=0.05,
    wspace=0.1 if NUM_PEOPLE <= 500 else 0.2
)

# Update the status text position and size
status_text = ax_status.text(0.5, 0.95, "", transform=ax_status.transAxes,
                          va='top', ha='center', fontsize=9,
                          bbox=dict(facecolor='white', alpha=0.7))

# Update time and stats text positions
time_text = ax1.text(0.5, 0.98, "", transform=ax1.transAxes, ha='center', va='top', fontsize=14,
                   bbox=dict(facecolor='white', alpha=0.7))
stats_text = ax1.text(0.02, 0.98, "", transform=ax1.transAxes, va='top', fontsize=12,
                    bbox=dict(facecolor='white', alpha=0.7))

# Optimized visualization
circles = [Circle((p.x, p.y), p.size/50) for p in people]
circle_collection = PatchCollection(circles, match_original=True)
ax1.add_collection(circle_collection)

mask_circles = [Circle((p.x, p.y), p.size/70) for p in people if p.masked]
mask_collection = PatchCollection(mask_circles, facecolor='white', alpha=0.8)
ax1.add_collection(mask_collection)

# Draw zones
zones = [
    (HOME_AREA, "RESIDENTIAL", '#9467bd'),
    (WORK_AREA, "WORK", '#ff7f0e'),
    (RECREATIONAL_AREA, "RECREATIONAL", '#2ca02c'),
    (SCHOOL_AREA, "SCHOOL", '#e377c2')
]

for area, label, color in zones:
    x1, y1, x2, y2 = area
    rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1.5, edgecolor=color, facecolor=color + '20', zorder=0)
    ax1.add_patch(rect)
    ax1.text((x1+x2)/2, (y1+y2)/2, label, color=color, fontsize=10, ha='center', va='center',
            bbox=dict(facecolor='white', alpha=0.7))

def update(frame):
    global current_time, lockdown, pos
    current_time += TIME_PER_FRAME
    time_str = current_time.strftime("%A, %b %d\n%I:%M %p")
    day_night = "Day" if 6 <= current_time.hour < 18 else "Night"
    time_text.set_text(f"{time_str} ({day_night})")

    # Movement and recovery
    for p in people:
        p.move(current_time)
        if p.state == 1 and current_time.hour == 0 and current_time.minute == 0:
            p.days_infected += 1
            if p.days_infected >= RECOVERY_DAYS:
                p.state = 2
                p.in_quarantine = False
                event_logger.log_event("Person recovered from infection")

    # Infection spread using spatial partitioning
    infected_count = 0
    for p1 in people:
        if p1.state == 1:  # Count all infected, regardless of quarantine status
            infected_count += 1
            if not p1.in_quarantine:  # Only spread infection if not quarantined
                cx, cy = p1.grid_cell
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        cell = (cx + dx, cy + dy)
                        for p2 in grid.get(cell, []):
                            if p2.state == 0:
                                dist = np.hypot(p1.x - p2.x, p1.y - p2.y)
                                if dist < INFECTION_RADIUS:
                                    # Add edge to graph if interaction occurs and NUM_PEOPLE <= 500
                                    if NUM_PEOPLE <= 500:
                                        G.add_edge(people.index(p1), people.index(p2))
                                    
                                    prob = INFECTION_PROB
                                    if p1.in_area(HOME_AREA) or p2.in_area(HOME_AREA):
                                        prob *= 0.8
                                    elif p1.in_area(WORK_AREA) or p2.in_area(WORK_AREA):
                                        prob *= 0.4
                                    if p2.masked and p1.masked:
                                        prob *= (1 - MASK_EFFECTIVENESS * 1.5)
                                    elif p2.masked or p1.masked:
                                        prob *= (1 - MASK_EFFECTIVENESS)
                                    if p2.vaccinated:
                                        prob *= (1 - VACCINE_EFFECTIVENESS)
                                    if random.random() < prob:
                                        p2.state = 1
                                        p2.infection_severity = random.random()
                                        # Place in quarantine if severity is high
                                        if p2.infection_severity > SEVERITY_THRESHOLD:
                                            p2.in_quarantine = True
                                            p2.current_target = p2.home
                                            p2.target_reached = False
                                            event_logger.log_event(f"High severity case quarantined at {p2.current_area if p2.current_area else 'Unknown'} area")
                                        event_logger.log_event(
                                            f"INFECTION: {p2.current_area if p2.current_area else 'Unknown'} area")

    # Lockdown trigger at 50% infection rate (counting all infected)
    if infected_count >= NUM_PEOPLE * LOCKDOWN_THRESHOLD and not lockdown:
        lockdown = True
        event_logger.log_event(f"LOCKDOWN! {infected_count}/{NUM_PEOPLE} ({infected_count/NUM_PEOPLE:.0%}) infected")
        
        # Place 60% of infected in quarantine
        infected_people = [p for p in people if p.state == 1]
        quarantine_count = int(len(infected_people) * 0.6)
        for p in random.sample(infected_people, quarantine_count):
            p.in_quarantine = True
            p.current_target = p.home
            p.target_reached = False
            event_logger.log_event("Infected person placed in quarantine during lockdown")
        
        # During lockdown, reduce movement but don't completely stop it
        for p in people:
            if not p.in_quarantine:  # Only affect non-quarantined people
                # Reduce their speed during lockdown
                p.speed *= 0.5
                # Increase probability of staying in current area
                p.max_area_time *= 1.5
                event_logger.log_event("Movement restrictions applied during lockdown")

    # Update event logger counts
    event_logger.update_counts(people)

    # Update status text
    status_text.set_text(event_logger.get_status_text())

    # Optimized visualization update
    circles = []
    colors = []
    mask_circles = []
    for p in people:
        circles.append((p.x, p.y, p.size / 50))
        if p.state == 0:
            colors.append('#1f77b4' if not p.vaccinated else '#17becf')
        elif p.state == 1:
             if p.in_quarantine:  # Quarantined infected
              colors.append('#800080')  # Purple for quarantined infected
             else:
              severity = min(0.8 + p.infection_severity * 0.2, 1.0)
              colors.append((severity, 0.1, 0.1))  # Red shades for infected
        else:
            colors.append('#2ca02c')
        if p.masked:
            mask_circles.append((p.x, p.y, p.size / 70))

    circle_collection.set_paths([Circle((x, y), r) for x, y, r in circles])
    circle_collection.set_facecolor(colors)
    mask_collection.set_paths([Circle((x, y), r) for x, y, r in mask_circles])

    # Update NetworkX graph visualization only if NUM_PEOPLE <= 500
    if NUM_PEOPLE <= 500:
        ax_network.clear()
        ax_network.axis('off')
        
        # Update node colors based on state
        node_colors = []
        for p in people:
            if p.state == 0:
                node_colors.append('#1f77b4' if not p.vaccinated else '#17becf')
            elif p.state == 1:
                severity = min(0.8 + p.infection_severity * 0.2, 1.0)
                node_colors.append((severity, 0.1, 0.1))
            else:
                node_colors.append('#2ca02c')

        # Draw the graph with larger nodes
        nx.draw_networkx(G, pos=pos, ax=ax_network, node_color=node_colors, 
                        with_labels=False, node_size=80, alpha=0.6,
                        edge_color='gray', width=0.8)

    # Update disease spread graph
    ax3.clear()
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Number of People")
    ax3.grid(True)
    
    if len(event_logger.time_history) > 0:
        times = event_logger.time_history
        start_time = times[0]
        x_values = [(t - start_time).total_seconds() / 3600 for t in times]
        
        # Convert times to hours since start for x-axis
        susceptible = [NUM_PEOPLE - inf - rec for inf, rec in 
                  zip(event_logger.infection_history, event_logger.recovery_history)]
        
        susceptible_line.set_data(x_values, susceptible)
        infected_line.set_data(x_values, event_logger.infection_history)
        quarantined_line.set_data(x_values, event_logger.quarantined_history)
        recovered_line.set_data(x_values, event_logger.recovery_history)
        
        ax3.plot(x_values, susceptible, 'b-', label='Susceptible')
        ax3.plot(x_values, event_logger.infection_history, 'r-', label='Infected')
        ax3.plot(x_values, event_logger.quarantined_history, 'm-', label='Quarantined')
        ax3.plot(x_values, event_logger.recovery_history, 'g-', label='Recovered')
        ax3.legend(loc='upper left')
        
        # Add lockdown indicator if applicable
        if lockdown:
            lockdown_time = [(t - start_time).total_seconds() / 3600 
                            for t in times if t >= event_logger.time_history[-1]][0]
            ax3.axvline(x=lockdown_time, color='k', linestyle='--', label='Lockdown')
            ax3.text(lockdown_time, max(event_logger.infection_history)*0.9, 
                    'Lockdown', rotation=90, va='top')

    return circle_collection, mask_collection, time_text, status_text

ani = animation.FuncAnimation(fig, update, frames=1000, interval=50, blit=False)
plt.tight_layout()
plt.show()