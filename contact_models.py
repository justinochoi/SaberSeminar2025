
import pandas as pd
import arviz as az 
import bambi as bmb 

cards_data = pd.read_csv("data/cards_data.csv") 
cards_swings = cards_data[cards_data['swing_status'] == 'swing']
 
contact_mod_formula = '''swing_outcome ~ 1 + (1 | batter_id) + release_speed + 
    pfx_z + I(pfx_x**2)*plat_adv + plate_z + I(plate_x**2)*plat_adv'''
     
pitch_groups = ['FF','SI','FC','SL','CU','CH','ST'] 
     
def save_contact_mods(data): 
    
    for i, pitch in enumerate(pitch_groups): 
        subset = data[data['pitch_group'] == pitch_groups[i]]
        row_count = len(subset)
        print(f"Subset size: {row_count}")
        model = bmb.Model(
            formula = contact_mod_formula, 
            data = subset, family = 'categorical', dropna = True,
        ) 
        print(f"Fitting contact model for {pitch}")
        idata = model.fit(chains=4, cores=4, draws=1000, nuts_sampler="blackjax")
        idata.to_netcdf(f"models/{pitch}/contact_mod_idata.nc")     

# for four-seam fastballs, rhat > 1.00 for some parameters 
save_contact_mods(cards_swings) 


