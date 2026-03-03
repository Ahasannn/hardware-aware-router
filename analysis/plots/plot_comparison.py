import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================================
# Uniform styling parameters
# ============================================
FIGURE_SIZE = (12, 3)
FONT_SIZE = 12

# Color scheme for CARROT vs Our method vs IRT-Router
CARROT_COLOR = '#1f77b4'  # Blue
OUR_COLOR = '#ff7f0e'     # Orange
IRT_COLOR = '#2ca02c'     # Green

# ============================================
# Read data
# ============================================
quality_latency_df = pd.read_csv('Quality-Latency.csv', on_bad_lines='skip')
slo_quality_df = pd.read_csv('SLO-Quality.csv')
queue_df = pd.read_csv('Queue.csv')

# ============================================
# Create figure with three subfigures
# ============================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=FIGURE_SIZE)

# ============================================
# Subfigure 1: Quality vs Latency
# ============================================
ax1.plot(quality_latency_df['Carrot Latency'], quality_latency_df['Carrot Quality'],
         marker='o', linewidth=2, label='CARROT', color=CARROT_COLOR,
         markersize=6, markeredgecolor='black', markeredgewidth=0.5)
ax1.plot(quality_latency_df['IRT-Router Latency'], quality_latency_df['IRT-Router Quality'],
         marker='^', linewidth=2, label='IRT-Router', color=IRT_COLOR,
         markersize=6, markeredgecolor='black', markeredgewidth=0.5)
ax1.plot(quality_latency_df['Our Latency'], quality_latency_df['Our Quality'],
         marker='s', linewidth=2, label='Ours', color=OUR_COLOR,
         markersize=6, markeredgecolor='black', markeredgewidth=0.5)

ax1.set_xlabel('Latency (s)', fontsize=FONT_SIZE)
ax1.set_ylabel('Quality', fontsize=FONT_SIZE)
ax1.tick_params(axis='both', labelsize=FONT_SIZE)
ax1.legend(loc='upper left', fontsize=FONT_SIZE, framealpha=0.85, edgecolor='none')
ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

# ============================================
# Subfigure 2: SLO vs Quality
# ============================================
ax2.plot(slo_quality_df['Carrot Quality'], slo_quality_df['Carrot SLO'],
         marker='o', linewidth=2, label='CARROT', color=CARROT_COLOR,
         markersize=6, markeredgecolor='black', markeredgewidth=0.5)
ax2.plot(slo_quality_df['IRT-Router Quality'], slo_quality_df['IRT-Router SLO'],
         marker='^', linewidth=2, label='IRT-Router', color=IRT_COLOR,
         markersize=6, markeredgecolor='black', markeredgewidth=0.5)
ax2.plot(slo_quality_df['Our Quality'], slo_quality_df['Our SLO'],
         marker='s', linewidth=2, label='Ours', color=OUR_COLOR,
         markersize=6, markeredgecolor='black', markeredgewidth=0.5)

ax2.set_xlabel('Quality', fontsize=FONT_SIZE)
ax2.set_ylabel('SLO Attainment', fontsize=FONT_SIZE)
ax2.tick_params(axis='both', labelsize=FONT_SIZE)
ax2.legend(loc='upper right', fontsize=FONT_SIZE, framealpha=0.85, edgecolor='none')
ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

# ============================================
# Subfigure 3: Queue Composition (Stacked Bar)
# ============================================
arrival_rates = queue_df['Arrival Rate'].values
x = np.arange(len(arrival_rates))
bar_width = 0.25

# CARROT stacked bars (Running + Waiting)
carrot_running = queue_df['Carrot Runing'].values
carrot_waiting = queue_df['Carrot Waitting'].values

# IRT-Router stacked bars (Running + Waiting)
irt_running = queue_df['IRT-Router Runing'].values
irt_waiting = queue_df['IRT-Router Waitting'].values

# Our method stacked bars (Running + Waiting)
our_running = queue_df['Our Runing'].values
our_waiting = queue_df['Our Waitting'].values

# Colors for running (darker) and waiting (lighter)
carrot_running_color = '#2E5A88'  # Darker blue
carrot_waiting_color = '#A6C8E0'  # Lighter blue
irt_running_color = '#1E7A1E'     # Darker green
irt_waiting_color = '#A6D9A6'     # Lighter green
our_running_color = '#CC6600'     # Darker orange
our_waiting_color = '#FFD1A3'     # Lighter orange

# CARROT bars
ax3.bar(x - bar_width, carrot_running, bar_width, label='CARROT - Running',
        color=carrot_running_color, edgecolor='black', linewidth=0.5)
ax3.bar(x - bar_width, carrot_waiting, bar_width, bottom=carrot_running,
        label='CARROT - Waiting', color=carrot_waiting_color, edgecolor='black', linewidth=0.5)

# IRT-Router bars
ax3.bar(x, irt_running, bar_width, label='IRT-Router - Running',
        color=irt_running_color, edgecolor='black', linewidth=0.5)
ax3.bar(x, irt_waiting, bar_width, bottom=irt_running,
        label='IRT-Router - Waiting', color=irt_waiting_color, edgecolor='black', linewidth=0.5)

# Our method bars
ax3.bar(x + bar_width, our_running, bar_width, label='Ours - Running',
        color=our_running_color, edgecolor='black', linewidth=0.5)
ax3.bar(x + bar_width, our_waiting, bar_width, bottom=our_running,
        label='Ours - Waiting', color=our_waiting_color, edgecolor='black', linewidth=0.5)

ax3.set_xlabel('Arrival Rate', fontsize=FONT_SIZE)
ax3.set_ylabel('Normalized Queue Composition', fontsize=FONT_SIZE)
ax3.set_xticks(x)
ax3.set_xticklabels(arrival_rates)
ax3.tick_params(axis='both', labelsize=FONT_SIZE)
ax3.legend(loc='lower right', fontsize=8, framealpha=0.85, edgecolor='none', ncol=1)
ax3.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
ax3.set_ylim(0, 1.0)

# ============================================
# Add subfigure labels
# ============================================
ax1.set_title('(a) Quality vs Latency', fontsize=FONT_SIZE, pad=10)
ax2.set_title('(b) SLO vs Quality', fontsize=FONT_SIZE, pad=10)
ax3.set_title('(c) Queue Composition', fontsize=FONT_SIZE, pad=10)

# Tight layout
plt.tight_layout()

# Save the figure
plt.savefig('comparison_figure.png', dpi=300, bbox_inches='tight')
plt.savefig('comparison_figure.pdf', bbox_inches='tight')
print("Figures saved as 'comparison_figure.png' and 'comparison_figure.pdf'")

plt.close()

print("Comparison figure generated successfully!")
