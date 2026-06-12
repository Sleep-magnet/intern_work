import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the dataset you generated
try:
    df = pd.read_csv("ieee_optimized_dataset.csv")
except FileNotFoundError:
    print("Error: Please run paper_trial.py first to generate the dataset.")
    exit()

# We will just plot the first 50 flight simulations so the graph is easy to read
subset = df.head(50)

# 2. Set up the visual figure
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
fig.suptitle('C-LSTM Real-Time Autopilot Tuning vs. Environmental Changes', fontsize=14, fontweight='bold')

# --- TOP GRAPH: THE WEATHER (Altitude & Turbulence) ---
color_alt = 'tab:blue'
ax1.set_ylabel('Altitude (m)', color=color_alt, fontweight='bold')
ax1.plot(subset.index, subset['H_Altitude'], color=color_alt, linewidth=2, label='Flight Altitude')
ax1.tick_params(axis='y', labelcolor=color_alt)

# Create a secondary Y-axis for Turbulence
ax1_turb = ax1.twinx()  
color_turb = 'tab:red'
ax1_turb.set_ylabel('Turbulence Scale (m)', color=color_turb, fontweight='bold')
ax1_turb.plot(subset.index, subset['L_Wx_Scale'], color=color_turb, linewidth=2, linestyle='dashed', label='Turbulence')
ax1_turb.tick_params(axis='y', labelcolor=color_turb)
ax1.set_title('Phase 1: Environmental Triggers (The Inputs)')

# --- BOTTOM GRAPH: THE AI OUTPUT (Autopilot Gains) ---
ax2.set_xlabel('Simulated Flight Iterations', fontweight='bold')
ax2.set_ylabel('Multiplier Gain Value', fontweight='bold')
ax2.plot(subset.index, subset['ie_beta'], color='purple', linewidth=2, label='Aileron Gain (ie_beta)')
ax2.plot(subset.index, subset['ic_beta'], color='green', linewidth=2, label='Rudder Gain (ic_beta)')
ax2.set_title('Phase 2: Neural Network Outputs (The Tuning Knobs)')
ax2.legend(loc='upper right')
ax2.grid(True, linestyle='--', alpha=0.6)

# Display the graph
plt.tight_layout()
plt.show()