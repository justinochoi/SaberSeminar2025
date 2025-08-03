
import arviz as az 
import bambi as bmb 

cards_bip = cards_data[cards_data['swing_outcome'] == 'bip'] 

bip_mod_formula = '''bbe_outcome ~ 1 + (1 | batter_id) + (1 | pitch_group) + plat_adv + 
  release_speed + pfx_z + pfx_x + plate_z + plate_x''' 
  
bip_mod = bmb.Model(
    formula = bip_mod_formula, 
    data = cards_bip, family = 'categorical', dropna=True
    )

bip_mod_idata = bip_mod.fit(
    chains=4, cores=4, draws=1000, nuts_sampler='blackjax', 
    idata_kwargs=dict(log_likelihood=True)
    )

bip_mod_idata.to_netcdf("models/bip_mod_idata.nc")



