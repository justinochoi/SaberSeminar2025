
import matplotlib.pyplot as plt 

almonte = data[(data['pitcher_name'] == 'Almonte, Yency') & (data['pitch_group'] == 'ST')] 

plt.hist(x=almonte['plate_x']) 

matrix = almonte[['plate_x','plate_z']].cov() 

matrix.iloc[1,1] 
