
import arviz as az 
import bambi as bmb 

cards_swings = cards_data[cards_data['swing_status'] == 'swing']
 
contact_mod_formula = '''swing_outcome ~ 1 + (1 | batter_id) + plat_adv + 
     release_speed + pfx_z + pfx_x + plate_z + plate_x'''
     
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
     
save_contact_mods(cards_swings) 
     

