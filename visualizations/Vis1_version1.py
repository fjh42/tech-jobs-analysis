import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('archive\global_ai_jobs.csv')

# Step 1: Calculate composite opportunity score for each job role/industry


# Calculate adjusted compensation
df['adjusted_compensation'] = (df['salary_usd'] + df['bonus_usd']) *\
    (1 - df['tax_rate_percent']/100) /df['cost_of_living_index']

# Aggregate by job role and industry
agg_df = df.groupby(['job_role', 'industry']).agg({
    'adjusted_compensation': 'median',
    'job_security_score': 'median',
    'career_growth_score': 'median',
    'work_life_balance_score': 'median',
    'hiring_difficulty_score': 'median',
    'interview_rounds': 'median',
    'salary_usd': 'median',
    'bonus_usd': 'median',
    'company_size': lambda x: x.mode()[0] if not x.mode().empty else 'Unknown',  # Most common
    'work_mode': lambda x: x.mode()[0] if not x.mode().empty else 'Unknown',    # Most common
    'experience_level': lambda x: x.mode()[0] if not x.mode().empty else 'Unknown',
    'country': lambda x: x.mode()[0] if not x.mode().empty else 'Unknown'
}).reset_index()

# Normalize the components
scaler = StandardScaler()
components = ['adjusted_compensation', 'job_security_score', 
              'career_growth_score', 'work_life_balance_score']
scaled_components = scaler.fit_transform(agg_df[components])

# Create composite score
weights = [0.4, 0.2, 0.2, 0.2] 
composite_score = np.dot(scaled_components, weights)

# Convert to 0-100 scale
composite_score = 100 * (composite_score - composite_score.min()) / \
(composite_score.max() - composite_score.min())
agg_df['opportunity_score'] = composite_score


# Step 2:Calculate combined hiring difficulty


difficulty_scaler = StandardScaler()
difficulty_components = ['hiring_difficulty_score', 'interview_rounds']
scaled_difficulty = difficulty_scaler.fit_transform(agg_df[difficulty_components])

# Combine (with equal weighting)
combined_difficulty = scaled_difficulty.mean(axis=1)

# Convert to 0-100 scale
combined_difficulty = 100 * (combined_difficulty - combined_difficulty.min()) / \
(combined_difficulty.max() - combined_difficulty.min())
agg_df['hiring_difficulty_combined'] = combined_difficulty


# Step 3: Plotting

# Sort by opportunity score
agg_df = agg_df.sort_values('opportunity_score', ascending=False).reset_index(drop=True)
agg_df['rank'] = range(1, len(agg_df) + 1)

agg_df['short_label'] = agg_df['job_role'] + ' | ' + agg_df['industry']

fig = plt.figure(figsize=(24, 14))
ax1 = plt.axes([0.08, 0.1, 0.55, 0.8]) 
ax2 = plt.axes([0.68, 0.1, 0.3, 0.8]) 

scatter = ax1.scatter(agg_df['hiring_difficulty_combined'], 
                     agg_df['opportunity_score'],
                     c=agg_df['opportunity_score'], 
                     cmap='viridis',
                     alpha=0.8,
                     s=250,
                     edgecolors='black', 
                     linewidth=1)

# Add rank numbers to each dot
for idx, row in agg_df.iterrows():
    ax1.annotate(str(row['rank']), 
                (row['hiring_difficulty_combined'], row['opportunity_score']),
                xytext=(0, 0), textcoords='offset points',
                ha='center', va='center',
                fontsize=8, fontweight='bold',
                color='white',
                bbox=dict(boxstyle='circle', facecolor='black', alpha=0.8, edgecolor='none', pad=0.3))

ax1.set_xlabel('Hiring Difficulty Score', fontsize=14)
ax1.set_ylabel('Composite Opportunity Score', fontsize=14)
ax1.set_title('Tech Job Opportunities by Role and Industry', 
              fontsize=16, fontweight='bold')

# Caption area on the right 
ax2.axis('off')
ax2.text(0, 0.98, 'JOB ROLE | INDUSTRY KEY', 
         fontsize=16, fontweight='bold', transform=ax2.transAxes)
ax2.text(0, 0.92, '─' * 35, transform=ax2.transAxes, color='gray')

# Create numbered captions (single line each)
y_position = 0.88
line_height = 0.028 

# Determine how many entries we can fit
max_entries = int((y_position - 0.05) / line_height)

for i in range(min(max_entries, len(agg_df))):
    row = agg_df.iloc[i]
    
    # Truncate caption if too long
    label = row['short_label']
    if len(label) > 35:
        label = label[:32] + '...'
    
    caption = f"{row['rank']:2d}. {label}"
    
    # Add to caption list
    ax2.text(0.02, y_position, caption, 
            fontsize=8, transform=ax2.transAxes,
            verticalalignment='top')
    
    y_position -= line_height

# If couldn't fit all, show count of remaining
if max_entries < len(agg_df):
    remaining = len(agg_df) - max_entries
    ax2.text(0.02, y_position - 0.01, f'... and {remaining} more roles', 
            fontsize=9, style='italic', transform=ax2.transAxes)

plt.tight_layout()
plt.show()
