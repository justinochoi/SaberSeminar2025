
import pandas as pd 
import arviz as az 
import bambi as bmb 

cards_data = pd.read_csv("data/cards_data.csv") 

swing_mod_formula = '''swing_status['swing'] ~ 1 + (1 | batter_id) + 
   balls + strikes + I(plate_x**2)*plat_adv + plate_z + arm_angle*plat_adv ''' 
   
pitch_groups = ['FF','SI','FC','SL','CU','CH','ST'] 

def save_swing_mods(data): 
    
    for i, pitch in enumerate(pitch_groups): 
        subset = data[data['pitch_group'] == pitch_groups[i]]
        row_count = len(subset)
        print(f"Subset size: {row_count}")
        model = bmb.Model(
            formula = swing_mod_formula, 
            data = subset, family = 'bernoulli', dropna = True,
        ) 
        print(f"Fitting swing model for {pitch}")
        idata = model.fit(chains=4, cores=4, draws=1000, nuts_sampler='blackjax')
        idata.to_netcdf(f"models/{pitch}/swing_mod_idata.nc") 

save_swing_mods(cards_data)
