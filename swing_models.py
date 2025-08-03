
import arviz as az 
import bambi as bmb 

swing_mod_formula = '''swing_status['swing'] ~ 1 + (1 | batter_id) + 
   balls + strikes + plate_x + plate_z''' 
   
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
        idata = model.fit(chains=4, cores=4, draws=1000, nuts_sampler="blackjax")
        idata.to_netcdf(f"models/{pitch}/swing_mod_idata.nc") 

save_swing_mods(cards_data)

swing_prob = az.extract(swing_idata, group='posterior', num_samples=500,
                            combined=True)['p'].mean(dim='sample').values 


with open('model_dict.json') as model_json: 
    model_dict = json.load(model_json)

swing_idata = az.from_netcdf(model_dict["ST"]["swing_mod_path"])
swing_idata.observed_data

