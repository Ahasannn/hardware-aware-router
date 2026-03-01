import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast

# ============================================
# Uniform styling parameters
# ============================================
FIGURE_SIZE = (6, 4)
FONT_SIZE = 12  # Uniform font size for all text elements

# Professional research color palette (ColorBrewer-inspired)
MODEL_COLORS = {
    'Qwen2.5-3B-Instruct': '#1f77b4',  # Blue
    'Qwen2.5-14B-Instruct': '#ff7f0e',   # Orange
    'Llama-3.1-8B-Instruct': '#2ca02c',  # Green
    'Mistral-7B-Instruct-v0.3': '#d62728',  # Red
    'Phi-3-mini-128k-instruct': '#9467bd'  # Purple
}

def get_model_name(model):
    """Return model name without -Instruct suffix"""
    return model.replace('-Instruct', '').replace('-instruct', '')

# ============================================
# Read and parse data
# ============================================
df = pd.read_csv('motivation_sweep_with_latency.csv')
arrival_rates = sorted(df['arrival_rate'].unique())

# Parse tail latency values
provided_tail_latency = [16.049221, 21.458732, 24.706466, 33.490302, 44.526749, 74.80542042255402, 129.67236471176147]
weighted_tail_latency = provided_tail_latency[:len(arrival_rates)]

# Parse model distribution for stacked bar
models = list(ast.literal_eval(df.iloc[0]['model_distribution']).keys())
req_dist = df["model_distribution"].apply(ast.literal_eval)

# Compute per-model share (normalized to percentage)
req_share = []
for rd in req_dist:
    total = sum(rd.values())
    req_share.append([rd[m] / total * 100 if total > 0 else 0 for m in models])

req_share = np.array(req_share)

# Parse waiting time data
waiting_data = []
for idx, row in df.iterrows():
    arrival_rate = row['arrival_rate']
    waiting_dict = ast.literal_eval(row['avg_waiting_per_model'])
    for model, waiting in waiting_dict.items():
        waiting_data.append({
            'arrival_rate': arrival_rate,
            'model': model,
            'avg_waiting': waiting
        })

plot_df = pd.DataFrame(waiting_data)

# ============================================
# Create figure with two subfigures stacked vertically
# ============================================
fig, (ax1, ax3) = plt.subplots(2, 1, figsize=FIGURE_SIZE)

# ============================================
# TOP SUBPLOT: Model Distribution + Tail Latency
# ============================================
x = np.arange(len(arrival_rates))
bar_width = 0.6

bottom = np.zeros(len(arrival_rates))
for i, model in enumerate(models):
    model_label = get_model_name(model)
    color = MODEL_COLORS.get(model, '#cccccc')

    bars = ax1.bar(
        x,
        req_share[:, i],
        bar_width,
        bottom=bottom,
        label=model_label,
        color=color,
        edgecolor='black',
        linewidth=0.8,
        alpha=0.8
    )
    bottom += req_share[:, i]

ax1.set_ylabel('Query\nAllocation (%)', fontsize=FONT_SIZE)
ax1.set_xticks(x)
ax1.set_xticklabels(arrival_rates, fontsize=FONT_SIZE)
ax1.tick_params(axis='y', labelsize=FONT_SIZE)
ax1.set_ylim(0, 100)
ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)

# Right y-axis for tail latency
ax2 = ax1.twinx()

ax2.plot(x, weighted_tail_latency, marker='s', linewidth=2.5,
         label='p95 Tail Latency', color='darkred',
         markersize=7, linestyle='-', alpha=0.9, markeredgecolor='black', markeredgewidth=0.5)

ax2.set_ylabel('p95 Tail Latency\n(Sec)', fontsize=FONT_SIZE, color='darkred')
ax2.tick_params(axis='y', labelcolor='darkred', labelsize=FONT_SIZE)


# ============================================
# BOTTOM SUBPLOT: Waiting Time
# ============================================
bar_width_waiting = 0.13
x_waiting = np.arange(len(arrival_rates))

for i, model in enumerate(models):
    model_data = plot_df[plot_df['model'] == model]
    waiting_times = [model_data[model_data['arrival_rate'] == rate]['avg_waiting'].values[0]
                     for rate in arrival_rates]

    model_label = get_model_name(model)
    color = MODEL_COLORS.get(model, '#cccccc')

    ax3.bar(x_waiting + i * bar_width_waiting, waiting_times, bar_width_waiting,
            label=model_label, color=color, edgecolor='black',
            linewidth=0.8, alpha=0.8)

ax3.set_ylabel('Average Waiting\nTime (Sec)', fontsize=FONT_SIZE)
ax3.set_xticks(x_waiting + bar_width_waiting * (len(models) - 1) / 2)
ax3.set_xticklabels(arrival_rates, fontsize=FONT_SIZE)
ax3.tick_params(axis='y', labelsize=FONT_SIZE)
ax3.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)

# ============================================
# Shared legend at the bottom with 2 rows
# ============================================
# Collect all handles and labels from both subplots
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

# Combine handles and labels (models + tail latency line)
all_handles = handles1 + handles2
all_labels = labels1 + labels2

# Create a single legend at the top center with 2 rows (3 columns per row)
fig.legend(all_handles, all_labels, loc='upper center', ncol=3,
           fontsize=FONT_SIZE, framealpha=0.85, edgecolor='none',
           fancybox=False, shadow=False, columnspacing=0.6,
           bbox_to_anchor=(0.5, 1.1))

# Add titles at the bottom
ax1.set_xlabel('Arrival Rate (requests/sec)',
               fontsize=FONT_SIZE)
ax3.set_xlabel('Arrival Rate (requests/sec)',
               fontsize=FONT_SIZE)

# Tight layout with space for legend at top
plt.tight_layout()
plt.subplots_adjust(hspace=0.5, top=0.92)

# Save the figure
plt.savefig('combined_motivation_figure.png', dpi=300, bbox_inches='tight')
plt.savefig('combined_motivation_figure.pdf', bbox_inches='tight')
print("Figures saved as 'combined_motivation_figure.png' and 'combined_motivation_figure.pdf'")

plt.close()

print("Combined motivation figure generated successfully!")
