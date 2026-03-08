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


# Step 3: Plot

agg_df['point_label'] = agg_df['job_role'] + '\n(' + agg_df['industry'] + ')'


plt.figure(figsize=(16, 10))
scatter = plt.scatter(agg_df['hiring_difficulty_combined'], 
                      agg_df['opportunity_score'], 
                      c=pd.Categorical(agg_df['job_role']).codes, cmap='tab20',
                      alpha=0.7,s=200) 
for idx, row in agg_df.iterrows():
    plt.annotate(row['point_label'], 
                (row['hiring_difficulty_combined'], row['opportunity_score']),
                xytext=(5, 5), textcoords='offset points', 
                fontsize=8, alpha=0.8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))
plt.xlabel('Hiring Difficulty Score', fontsize=14)
plt.ylabel('Composite Opportunity Score', fontsize=14)
plt.title('Tech Job Opportunities by Role and Industry: Quality vs. Accessibility (2020-2026)', 
          fontsize=16, fontweight='bold')

plt.tight_layout()
plt.show()