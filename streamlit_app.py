
import joblib 
import json 
import catboost 
import arviz as az 
import bambi as bmb 
import numpy as np 
import pandas as pd 
import pymc as pm 
import streamlit as st 
import matplotlib.pyplot as plt 
import matplotlib.patches as patches 
import seaborn as sns 
from collections import defaultdict 
from scipy import stats 


# shoutout Max Bay 
def reverse_name(name):
    parts = name.split(', ')
    if len(parts) == 1:
        return name
    last = parts[0]
    first_and_suffix = parts[1].split()
    if len(first_and_suffix) > 1 and first_and_suffix[-1] in ['Jr.', 'Sr.', 'II', 'III', 'IV', '(L)', '(R)']:
        first = ' '.join(first_and_suffix[:-1])
        suffix = first_and_suffix[-1]
        return f"{first} {last} {suffix}"
    else:
        first = ' '.join(first_and_suffix)
        return f"{first} {last}"

# covariance estimating function
def estimate_sigma(pitcher_locs, prior_sds): 
    
    alpha_x, loc_x, scale_x = stats.gamma.fit(prior_sds['plate_x_std'], floc=0) 
    alpha_z, loc_z, scale_z = stats.gamma.fit(prior_sds['plate_z_std'], floc=0)
    
    with pm.Model() as cov_model: 
        sd_dist = pm.Gamma.dist(alpha=[alpha_x, alpha_z], beta=[1/scale_x, 1/scale_z])
        
        chol, corr, stds = pm.LKJCholeskyCov(
            "chol", n=2, eta=1, sd_dist=sd_dist 
            )
        
        cov = pm.Deterministic("cov", chol.dot(chol.T))
        mu = pm.Normal("mu", mu=0.0, sigma=1.5, shape=2) 
        obs = pm.MvNormal("obs", mu, chol=chol, observed=pitcher_locs) 
    
    with cov_model: 
        trace = pm.sample(cores=4, random_seed=76)  
        
    sigma = trace.posterior["cov"].mean(("chain", "draw")).values
    
    return sigma

def nested_defaultdict():
    return defaultdict(dict)

@st.cache_data 
def load_dicts(): 
    with open('count_dict.json') as count_json: 
        count_dict = json.load(count_json)
        
    with open('model_dict.json') as model_json: 
        model_dict = json.load(model_json)
        
    run_values = pd.read_csv('data/run_values.csv')
    rv_dict = defaultdict(nested_defaultdict)  

    for _, row in run_values.iterrows():
        event = row['event']
        balls = row['balls']
        strikes = row['strikes']
        delta_run_exp = row['delta_run_exp']
        
        rv_dict[event][balls][strikes] = delta_run_exp
    
    return count_dict, model_dict, rv_dict 

count_dict, model_dict, rv_dict = load_dicts() 

@st.cache_data 
def load_data(): 
    cards_data = pd.read_csv('data/cards_data.csv')
    cards_info = pd.read_csv('data/cards_info.csv') 
    pitch_summaries = pd.read_csv('data/pitch_summaries.csv') 
    pitch_locs = pd.read_csv('data/pitch_locations.csv')
    std_summaries = pd.read_csv('data/std_summaries.csv') 
    
    return cards_data, cards_info, pitch_summaries, pitch_locs, std_summaries 

cards_data, cards_info, pitch_summaries, pitch_locs, std_summaries = load_data() 
    
@st.cache_data 
def load_model(): 
    called_strike_mod = joblib.load("models/called_strike_mod.pkl")
    
    return called_strike_mod 

called_strike_mod = load_model() 

st.title("Pitch Target Optimizer") 
st.write("Given a batter-pitcher matchup, count, and pitch type associated with the selected pitcher, this app estimates where that pitcher should aim to maximize his success. The underlying logic is that there shouldn't be one ideal target for every pitcher - your ideal target must take into account how good your command and stuff is, as well as who is currently at the plate.")
st.text('') 
st.write("Because the targets are calculated using a Bayesian framework, note that repeated simulations may yield slightly different results. However, the model parameters have been set to ensure that the targets or predicted run values don't wary wildly. It takes approximately 30 seconds to run the app. For more information on how everything is executed, please refer to the section at the bottom.")

col1, col2 = st.columns(2) 

with col1: 
    batters = sorted(cards_info['batter_name'].unique()) 
    batter = st.selectbox("Select a batter", batters)
    counts = count_dict.keys() 
    count = st.selectbox("Select a count", counts) 
    
with col2: 
    pitchers = sorted(pitch_summaries['pitcher_name'].unique()) 
    pitcher = st.selectbox("Select a pitcher", pitchers)
    pitch_groups = pitch_summaries.loc[pitch_summaries['pitcher_name'] == pitcher]['pitch_group'].unique() 
    pitch_group = st.selectbox("Select a pitch type", pitch_groups)
    
balls = count_dict[count]['balls'] 
strikes = count_dict[count]['strikes']
    
@st.cache_resource
def rebuild_bambi(pitch_group): 
    
    swing_mod_formula = '''swing_status['swing'] ~ 1 + (1 | batter_id) + 
        balls + strikes + I(plate_x**2)*plat_adv + plate_z + arm_angle*plat_adv ''' 
       
    contact_mod_formula = '''swing_outcome ~ 1 + (1 | batter_id) + release_speed + 
        pfx_z + I(pfx_x**2)*plat_adv + plate_z + I(plate_x**2)*plat_adv'''
          
    bip_mod_formula = '''bbe_outcome ~ 1 + (1 | batter_id) + (1 | pitch_group) + 
        release_speed + pfx_z + I(pfx_x**2)*plat_adv + plate_z + I(plate_x**2)*plat_adv''' 
    
    swing_mod = bmb.Model(
        formula = swing_mod_formula, 
        data = cards_data[cards_data['pitch_group'] == pitch_group], 
        family = 'bernoulli', dropna=True
    ) 
        
    contact_mod = bmb.Model(
        formula = contact_mod_formula, 
        data = cards_data[(cards_data['pitch_group'] == pitch_group) & (cards_data['swing_status'] == 'swing')], 
        family = 'categorical', dropna=True
    ) 
        
    bip_mod = bmb.Model(
        formula = bip_mod_formula, 
        data = cards_data[cards_data['swing_outcome'] == 'bip'], 
        family = 'categorical', dropna=True
    )
    
    return swing_mod, contact_mod, bip_mod 

swing_mod, contact_mod, bip_mod = rebuild_bambi(pitch_group)
    
@st.cache_data 
def load_idata(pitch_group): 
    swing_idata = az.from_netcdf(model_dict[pitch_group]["swing_mod_path"]) 
    contact_idata = az.from_netcdf(model_dict[pitch_group]["contact_mod_path"]) 
    bip_idata = az.from_netcdf("models/bip_mod_idata.nc")
    
    return swing_idata, contact_idata, bip_idata

swing_idata, contact_idata, bip_idata = load_idata(pitch_group) 

@st.cache_data 
def load_info(pitcher, batter, pitch_group, balls, strikes): 
    
    pitcher_info = (
        pitch_summaries 
        .loc[pitch_summaries['pitcher_name'] == pitcher]
        .loc[pitch_summaries['pitch_group'] == pitch_group]
        [['pitcher_hand','mean_velo','mean_pfx_x','mean_pfx_z','mean_arm_angle']]
    )
    
    batter_info = (
        cards_info
        .loc[cards_info['batter_name'] == batter]
        [['batter_id','batter_hand','sz_top','sz_bot']]
    ) 
    
    loc_info = (
        pitch_locs
        .loc[pitch_locs['pitcher_name'] == pitcher]
        .loc[pitch_locs['pitch_group'] == pitch_group]
        .loc[pitch_locs['balls'] == balls]
        .loc[pitch_locs['strikes'] == strikes]
        [['plate_x', 'plate_z']]
    ) 
    
    return pitcher_info, batter_info, loc_info 

pitcher_info, batter_info, loc_info = load_info(pitcher, batter, pitch_group, balls, strikes)

@st.cache_data
def get_handedness(pitcher_info, batter_info): 
    
    pitcher_hand = pitcher_info['pitcher_hand'].unique()[0]
    batter_hand = batter_info['batter_hand'].unique()[0]
    
    if batter_hand == pitcher_hand: 
        plat_adv = 1 
    else: 
        plat_adv = 0 
        
    return pitcher_hand, batter_hand, plat_adv 

pitcher_hand, batter_hand, plat_adv = get_handedness(pitcher_info, batter_info)

@st.cache_data 
def load_prior(pitch_group, pitcher_hand, balls, strikes): 
    
    prior_std = (
        std_summaries
        .loc[std_summaries['pitch_group'] == pitch_group]
        .loc[std_summaries['pitcher_hand'] == pitcher_hand]
        .loc[std_summaries['balls'] == balls]
        .loc[std_summaries['strikes'] == strikes]
        [['plate_x_std','plate_z_std']]
    )
    
    return prior_std 

prior_std = load_prior(pitch_group, pitcher_hand, balls, strikes)

if st.button("Create and Plot Optimal Pitch Target"): 
    
    # if not enough location data for pitch type + count combo, use league data 
    if len(loc_info) < 10: 
        league = (
            pitch_locs
            .loc[pitch_locs['pitch_group'] == pitch_group]
            .loc[pitch_locs['balls'] == balls]
            .loc[pitch_locs['strikes'] == strikes]
            [['plate_x', 'plate_z']]
        )
        sigma = league.cov().to_numpy() 
    else: 
        sigma = estimate_sigma(loc_info, prior_std)

    # based on step size, generate candidate targets 
    x_seq = np.linspace(-1, 1, num=5)
    z_seq = np.linspace(1.5, 3.5, num=5) 
    targets = [(x, z) for x in x_seq for z in z_seq]
    results = pd.DataFrame() 
    
    for i in range(len(targets)):
        
        n = 500 
        target = np.array(targets[i]) 
        sims = stats.multivariate_normal.rvs(
            mean = target, cov = sigma, 
            size = n, random_state=76
        )
        
        sim_data = pd.DataFrame({
            'plate_x': sims[:,0],  
            'plate_z': sims[:,1],  
            'pitch_group': [pitch_group] * n,
            'batter_id': [batter_info['batter_id'].values[0]] * n,
            'pitcher_hand': [pitcher_hand] * n,
            'batter_hand': [batter_hand] * n,
            'plat_adv': [plat_adv] * n, 
            'balls': [balls] * n, 
            'strikes': [strikes] * n, 
            'release_speed': [pitcher_info['mean_velo'].values[0]] * n, 
            'arm_angle': [pitcher_info['mean_arm_angle'].values[0]] * n, 
            'pfx_x': [pitcher_info['mean_pfx_x'].values[0]] * n, 
            'pfx_z': [pitcher_info['mean_pfx_z'].values[0]] * n, 
            'sz_bot': [batter_info['sz_bot'].values[0]] * n, 
            'sz_top': [batter_info['sz_top'].values[0]] * n 
            })
        
        cb_feats = ['plate_x','plate_z','sz_bot','sz_top','pitcher_hand','batter_hand']
        cb_probs = called_strike_mod.predict_proba(sim_data[cb_feats]) 
        
        # predict probabilities 
        ball_prob = cb_probs[:,0]
        cs_prob = cb_probs[:,1]
        hbp_prob = cb_probs[:,2]
        
        new_swing = swing_mod.predict(swing_idata, kind="response_params", data=sim_data, inplace=False)
        swing_prob = az.extract(new_swing, group='posterior', num_samples=500,
                                combined=True)['p'].mean(dim='sample').values 
        
        new_contact = contact_mod.predict(contact_idata, kind="response_params", data=sim_data, inplace=False)
        contact_probs = az.extract(new_contact, group='posterior', num_samples=500,
                                   combined=True)['p'].mean(dim='sample').values 
        
        new_bip = bip_mod.predict(bip_idata, kind="response_params", data=sim_data, inplace=False)
        bip_probs = az.extract(new_bip, group='posterior', num_samples=500,
                               combined=True)['p'].mean(dim='sample').values 
        
        foul_prob = contact_probs[:,0]
        in_play_prob = contact_probs[:,1]
        whiff_prob = contact_probs[:,2]
     
        double_prob = bip_probs[:,0]
        hr_prob = bip_probs[:,1]
        out_prob = bip_probs[:,2]
        single_prob = bip_probs[:,3]
        triple_prob = bip_probs[:,4]
        
        # event values 
        if strikes == 2: 
            cs_val = rv_dict['strikeout'][balls][strikes] 
            whiff_val = rv_dict['strikeout'][balls][strikes]
        else: 
            cs_val = rv_dict['called_strike'][balls][strikes] 
            whiff_val = rv_dict['swinging_strike'][balls][strikes]
            
        if balls == 3: 
            ball_val = rv_dict['walk'][balls][strikes]
        else: 
            ball_val = rv_dict['ball'][balls][strikes]
        
        hbp_val = rv_dict['hit_by_pitch'][balls][strikes]
        foul_val = rv_dict['foul'][balls][strikes]
        double_val = rv_dict['double'][balls][strikes]
        hr_val = rv_dict['home_run'][balls][strikes]
        out_val = rv_dict['field_out'][balls][strikes]
        single_val = rv_dict['single'][balls][strikes]
        triple_val = rv_dict['triple'][balls][strikes]
        
        # probabilities times values 
        take_rv = (1 - swing_prob) * (ball_prob * ball_val + cs_prob * cs_val + hbp_prob * hbp_val)
        
        whiff_rv = swing_prob * (whiff_prob * whiff_val) 
        foul_rv = swing_prob * (foul_prob * foul_val) 
        double_rv = swing_prob * in_play_prob * (double_prob * double_val) 
        hr_rv = swing_prob * in_play_prob * (hr_prob * hr_val) 
        out_rv = swing_prob * in_play_prob * (out_prob * out_val) 
        single_rv = swing_prob * in_play_prob * (single_prob * single_val)
        triple_rv = swing_prob * in_play_prob * (triple_prob * triple_val)
        
        swing_rv = whiff_rv + foul_rv + double_rv + hr_rv + out_rv + single_rv + triple_rv 
        
        total_rv = swing_rv + take_rv
        result = pd.DataFrame({
            'plate_x': [target[0]], 
            'plate_z': [target[1]], 
            'run_value': [np.mean(total_rv)]
        })
        results = pd.concat([results, result], axis=0, ignore_index=True) 
        
    best_idx = results['run_value'].idxmin() 
    best_x = results.iloc[best_idx]['plate_x']
    best_z = results.iloc[best_idx]['plate_z']
    best_rv = round(results.iloc[best_idx]['run_value'] * 100, 2) 
    
    # simulate pitches to use in plot 
    # for plotting purposes, x-coordinates need to be flipped (pitcher's POV)
    sigma[0,1] = sigma[0,1] * (-1) 
    sigma[1,0] = sigma[1,0] * (-1)
    best_x = best_x * (-1)
    plot_data = stats.multivariate_normal.rvs(mean=[best_x,best_z], cov=sigma, 
                                              size=500, random_state=76) 
    
    # set up the plot 
    fig, ax = plt.subplots(figsize=(8,8)) 
    
    # limits and labels 
    pitcher = reverse_name(pitcher)
    batter = reverse_name(batter) 

    ax.set_xlim(-2, 2)
    ax.set_ylim(1, 4)
    ax.set_aspect("equal") 
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_xlabel("Horizontal Plate Location ($ft$)", fontsize=10) 
    ax.set_ylabel("Vertical Plate Location ($ft$)", fontsize=10)
    ax.set_title(f"{pitcher} vs. {batter} (2024) | Count: {count}\nPredicted RV per 100 (lower is better): {best_rv}")
    
    # kde of simulated pitch locations 
    sns.kdeplot(x=plot_data[:,0], y=plot_data[:,1], fill=True, 
                thresh=0, levels=100, cmap="mako", bw_adjust=2)
    
    # add strike zone rectangle 
    rect = patches.Rectangle(xy=(-0.83,1.5), width=1.6, height=2.1, linewidth=1, edgecolor="white", fill=False)
    ax.add_patch(rect)
    
    # add 'x' marker indicating optimal target 
    plt.plot(best_x, best_z, marker='x', linestyle='None', markersize=20, color='red')
    
    # main title 
    plt.suptitle(f"Distribution of Simulated {pitch_group} Locations Given Optimal Target", 
                 fontsize=15, y = 0.90) 
    
    st.pyplot(fig)

st.markdown("---") 

st.markdown('<h2 style="font-size: 24px;">How does it all work?</h2>', unsafe_allow_html=True) 
with st.expander("Pull down to expand"): 
    st.write('''We first estimate the swing probability, foul/contact/whiff probabilities, and single/double/triple/home run/out
    probabilities of the hitter, using variables such as release speed, pitch movement, and arm angle. The hitter himself 
    is included as a random effect to account for individual tendencies; moreover, for hitters with small samples, 
    the random effect component effecitvely "shrinks" them towards the global mean. These estimates are obtained in advance.''')
    st.text('') 
    st.write('''When the app is run, we estimate the covariance matrix of the pitcher's pitch locations for the 
    specified count and pitch type. This covariance matrix represents the pitcher's command: Pitchers with poor command will have 
    relatively large variance in their locations, whereas pitchers with good command will have relatively small variance.
    We work under the assumption that for each count and pitch type, pitch locations follow a multivariate normal distribution.
    We further assume that the prior for a pitch location covariance matrix is a gamma distribution, whose shape is determined by the 
    curve that emerges when you plot all the league's pitchers' pitch location standard deviations. Essentially, we're saying 
    that absent any other information, a given pitcher has league-average command for a specific pitch type, thrown in a 
    specific count.''') 
    st.text('') 
    st.write('''To determine the optimal target, we iterate through a list of candidate targets. For each candidate, we simulate
    pitches using the estimated covariance matrix. Each pitch is assigned a run value via the swing/contact/in-play probabilities 
    from earlier, and we compute the average. The target with the lowest average run value is deemed optimal.''')

