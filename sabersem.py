import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd 
import numpy as np 
from scipy.stats import gamma 

std_summaries = pd.read_csv('data/std_summaries.csv') 

righty_ff_3_2 =  (
    std_summaries
    .loc[std_summaries['pitch_group'] == 'FF']
    .loc[std_summaries['pitcher_hand'] == 'R']
    .loc[std_summaries['balls'] == 3]
    .loc[std_summaries['strikes'] == 2]
    [['plate_x_std','plate_z_std']]
)

alpha_x, loc_x, scale_x = gamma.fit(righty_ff_3_2['plate_x_std'], floc=0) 

x = np.linspace(righty_ff_3_2['plate_x_std'].min(), righty_ff_3_2['plate_x_std'].max(), 500)
fig, ax = plt.subplots(figsize=(8,8))
sns.kdeplot(righty_ff_3_2['plate_x_std'], label='KDE of Standard Deviations') 
ax.plot(x, gamma.pdf(x, a=alpha_x, loc=loc_x, scale=scale_x), label='Best-fitted Gamma Dist.') 

ax.set_xlabel("X-Location Standard Deviation")
ax.legend() 
plt.title("3-2 Count Four-Seam Fastballs From RHP")
plt.show() 
