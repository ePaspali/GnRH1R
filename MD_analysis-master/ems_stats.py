import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import mannwhitneyu, ttest_ind, levene
import seaborn as sns
from matplotlib.patches import Rectangle


plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 12,
    "figure.dpi": 300
})


file_path = "path/to/file"


with open(file_path, 'r') as f:
    first_row = f.readline().strip().split(',')

df = pd.read_csv(file_path, skiprows=1)


time_col = df.columns[0]
df[time_col] = df[time_col] / 1000.0


ros1_cols = [df.columns[1], df.columns[2], df.columns[3]]
apo_cols = [df.columns[8], df.columns[9]]

# reshape data for analysis
def melt_condition(cols, condition_name):
    return df[[time_col] + cols].melt(
        id_vars=time_col,
        value_vars=cols,
        var_name="replicate",
        value_name="distance"
    ).assign(condition=condition_name)

ros1_long = melt_condition(ros1_cols, "ROS-1")
apo_long = melt_condition(apo_cols, "APO-GnRH1R")
long_df = pd.concat([ros1_long, apo_long], ignore_index=True)

long_df = pd.concat([ros1_long, apo_long], ignore_index=True)


long_df["distance"] = pd.to_numeric(long_df["distance"], errors="coerce")
long_df = long_df.dropna(subset=["distance"])
long_df = long_df.rename(columns={time_col: "time_us"})

# analysis cutoff after equilibration
equilibration_cutoff = 1.0  
analysis_df = long_df[long_df["time_us"] >= equilibration_cutoff].copy()

# split data by condition
ros1_data = analysis_df[analysis_df['condition'] == 'ROS-1']['distance'].values
apo_data = analysis_df[analysis_df['condition'] == 'APO-GnRH1R']['distance'].values

print(f"Data points after {equilibration_cutoff} µs:")
print(f"ROS-1: {len(ros1_data)} points")
print(f"APO-GnRH1R: {len(apo_data)} points")
print()

# --------------------
# statistical analysis
# --------------------

# basic stats
def describe_data(data, name):
    return {
        'name': name,
        'n': len(data),
        'mean': np.mean(data),
        'std': np.std(data, ddof=1),
        'median': np.median(data),
        'q25': np.percentile(data, 25),
        'q75': np.percentile(data, 75),
        'min': np.min(data),
        'max': np.max(data)
    }

ros1_stats = describe_data(ros1_data, 'ROS-1')
apo_stats = describe_data(apo_data, 'APO-GnRH1R')


print("=== DESCRIPTIVE STATISTICS ===")
for stats_dict in [ros1_stats, apo_stats]:
    print(f"{stats_dict['name']}:")
    print(f"  n = {stats_dict['n']}")
    print(f"  Mean ± SD: {stats_dict['mean']:.3f} ± {stats_dict['std']:.3f} Å")
    print(f"  Median [IQR]: {stats_dict['median']:.3f} [{stats_dict['q25']:.3f}-{stats_dict['q75']:.3f}] Å")
    print(f"  Range: {stats_dict['min']:.3f} - {stats_dict['max']:.3f} Å")
    print()

# effect size calculation
def cohens_d(x, y):
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / dof)
    return (np.mean(x) - np.mean(y)) / pooled_std

effect_size = cohens_d(ros1_data, apo_data)
mean_diff = np.mean(ros1_data) - np.mean(apo_data)

def interpret_effect_size(d):
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"

print("=== EFFECT SIZE ANALYSIS ===")
print(f"Mean difference: {mean_diff:.3f} Å")
print(f"Cohen's d: {effect_size:.3f} ({interpret_effect_size(effect_size)} effect)")
print()

# statistical tests
print("=== STATISTICAL TESTS ===")

# test for equal variances
levene_stat, levene_p = levene(ros1_data, apo_data)
equal_var = levene_p > 0.05

print(f"Levene's test: F = {levene_stat:.3f}, p = {levene_p:.2e}")
print(f"Equal variances: {equal_var}")
print()

# non-parametric test
mw_stat, mw_p = mannwhitneyu(ros1_data, apo_data, alternative='two-sided')
print(f"Mann-Whitney U: U = {mw_stat:.0f}, p = {mw_p:.2e}")

# parametric test
welch_stat, welch_p = ttest_ind(ros1_data, apo_data, equal_var=equal_var)
print(f"Welch's t-test: t = {welch_stat:.3f}, p = {welch_p:.2e}")
print()

# bootstrap confidence interval
def bootstrap_mean_diff(x, y, n_bootstrap=10000):
    np.random.seed(42)
    bootstrap_diffs = []
    
    for _ in range(n_bootstrap):
        x_boot = np.random.choice(x, size=len(x), replace=True)
        y_boot = np.random.choice(y, size=len(y), replace=True)
        bootstrap_diffs.append(np.mean(x_boot) - np.mean(y_boot))
    
    ci_lower = np.percentile(bootstrap_diffs, 2.5)
    ci_upper = np.percentile(bootstrap_diffs, 97.5)
    
    return ci_lower, ci_upper

boot_ci_lower, boot_ci_upper = bootstrap_mean_diff(ros1_data, apo_data)
print(f"Bootstrap 95% CI for mean difference: [{boot_ci_lower:.3f}, {boot_ci_upper:.3f}] Å")
print()

# --------------------
# results summary
# --------------------
print("=== RESULTS ===")
print(f"ROS-1 vs APO-GnRH1R distances")
print(f"Mean difference: {mean_diff:.3f} Å (95% CI: [{boot_ci_lower:.3f}, {boot_ci_upper:.3f}])")
print(f"Cohen's d = {effect_size:.3f} ({interpret_effect_size(effect_size)} effect)")

# select main test
if equal_var:
    main_p = welch_p
    test_name = "Welch's t-test"
else:
    main_p = welch_p
    test_name = "Welch's t-test (unequal variances)"

print(f"{test_name}: p = {main_p:.2e}")
print(f"Mann-Whitney U: p = {mw_p:.2e}")

# significance level
if main_p < 0.001:
    sig_text = "highly significant (p < 0.001)"
elif main_p < 0.01:
    sig_text = "very significant (p < 0.01)"  
elif main_p < 0.05:
    sig_text = "significant (p < 0.05)"
else:
    sig_text = f"not significant (p = {main_p:.3f})"

print(f"Result: {sig_text}")
print()

# --------------------
# plotting
# --------------------
fig, axes = plt.subplots(2, 2, figsize=(15, 12))


colors = {'ROS-1': '#2171b5', 'APO-GnRH1R': '#636363'}

# 5a. time series
ros1_colors = ['#084594', '#2171b5', '#6baed6']  # deep to light blue
apo_colors = ['black', 'grey']

for condition in ['ROS-1', 'APO-GnRH1R']:
    cond_data = long_df[long_df['condition'] == condition]
    reps = sorted(cond_data['replicate'].unique())
    for i, rep in enumerate(reps):
        rep_data = cond_data[cond_data['replicate'] == rep]
        if condition == "ROS-1":
            color = ros1_colors[i % len(ros1_colors)]
            label = f"{condition}-{i+1}"
        else:
            color = apo_colors[i % len(apo_colors)]
            label = f"{condition}-{i+1}"
        axes[0,0].plot(rep_data['time_us'], rep_data['distance'], 
                      color=color, alpha=0.8, linewidth=1.2, label=label)

# threshold and analysis lines
axes[0,0].axhline(y=10, color='red', linestyle='--', alpha=0.7, label='Activation threshold')
axes[0,0].axvline(x=equilibration_cutoff, color='orange', linestyle=':', alpha=0.7, label='Analysis start')

axes[0,0].set_xlabel('Time (µs)')
axes[0,0].set_ylabel('TM3-TM6 Distance (Å)')
axes[0,0].set_title('TM3-TM6 Distance Time Series')
axes[0,0].legend()

# 5b. violin plot
data_for_violin = []
labels_for_violin = []
for condition in ['ROS-1', 'APO-GnRH1R']:
    data_for_violin.append(analysis_df[analysis_df['condition'] == condition]['distance'])
    labels_for_violin.append(condition)

parts = axes[0,1].violinplot(data_for_violin, positions=[1, 2], showmeans=True, showmedians=True)
for i, (pc, condition) in enumerate(zip(parts['bodies'], ['ROS-1', 'APO-GnRH1R'])):
    pc.set_facecolor(colors[condition])
    pc.set_alpha(0.7)

axes[0,1].set_xticks([1, 2])
axes[0,1].set_xticklabels(labels_for_violin)
axes[0,1].set_ylabel('TM3-TM6 Distance (Å)')

# format p-value for display
if main_p == 0.0 or main_p < 1e-15:
    p_text = "p < 1×10⁻¹⁵"
else:
    p_text = f"p = {main_p:.2e}"

axes[0,1].set_title(f'Distribution Comparison\n{test_name}: {p_text}')

# statistical significance annotation
y_max = max(np.max(ros1_data), np.max(apo_data))
if main_p < 0.001:
    sig_symbol = '***'
elif main_p < 0.01:
    sig_symbol = '**'
elif main_p < 0.05:
    sig_symbol = '*'
else:
    sig_symbol = 'ns'

axes[0,1].plot([1, 2], [y_max + 0.5, y_max + 0.5], 'k-', linewidth=1)
axes[0,1].text(1.5, y_max + 0.7, sig_symbol, ha='center', va='bottom', fontsize=14)

# 5c. effect size visualisation
effect_data = [ros1_stats['mean'], apo_stats['mean']]
effect_errors = [ros1_stats['std']/np.sqrt(ros1_stats['n']), 
                apo_stats['std']/np.sqrt(apo_stats['n'])]

bars = axes[1,0].bar(['ROS-1', 'APO-GnRH1R'], effect_data, 
                    yerr=effect_errors, capsize=5, 
                    color=[colors['ROS-1'], colors['APO-GnRH1R']], 
                    alpha=0.8, edgecolor='black', linewidth=1)

axes[1,0].set_ylabel('Mean TM3-TM6 Distance (Å)')
axes[1,0].set_title(f'Mean Distance Comparison\nCohen\'s d = {effect_size:.3f} ({interpret_effect_size(effect_size)} effect)')


for bar, value, error in zip(bars, effect_data, effect_errors):
    axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + error + 0.1,
                  f'{value:.2f}', ha='center', va='bottom', fontweight='bold')

# 5d. bootstrap distribution of mean differences
bootstrap_diffs = []
np.random.seed(42)
for _ in range(1000):
    ros1_boot = np.random.choice(ros1_data, size=len(ros1_data), replace=True)
    apo_boot = np.random.choice(apo_data, size=len(apo_data), replace=True)
    bootstrap_diffs.append(np.mean(ros1_boot) - np.mean(apo_boot))

axes[1,1].hist(bootstrap_diffs, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
axes[1,1].axvline(x=mean_diff, color='red', linestyle='-', linewidth=2, label=f'Observed: {mean_diff:.3f} Å')
axes[1,1].axvline(x=boot_ci_lower, color='red', linestyle='--', alpha=0.7)
axes[1,1].axvline(x=boot_ci_upper, color='red', linestyle='--', alpha=0.7)
axes[1,1].axvline(x=0, color='black', linestyle=':', alpha=0.5, label='No difference')

axes[1,1].set_xlabel('Mean Difference (Å)')
axes[1,1].set_ylabel('Frequency')
axes[1,1].set_title(f'Bootstrap Distribution\n95% CI: [{boot_ci_lower:.3f}, {boot_ci_upper:.3f}] Å')
axes[1,1].legend()

plt.tight_layout()
plt.savefig("path", 
           format="pdf", dpi=300, bbox_inches='tight')
plt.show()

# --------------------
# 6. export summary statistics
# --------------------
summary_stats = pd.DataFrame([ros1_stats, apo_stats])
summary_stats['effect_size'] = [effect_size, np.nan]
summary_stats['p_value_welch'] = [welch_p, np.nan]
summary_stats['p_value_mw'] = [mw_p, np.nan]

summary_stats.to_csv("/Users/nikipaspali/Desktop/tm3_tm6_summary_stats.csv", index=False)
print(f"Summary statistics exported to: /Users/nikipaspali/Desktop/tm3_tm6_summary_stats.csv")
