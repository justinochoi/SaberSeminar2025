
import numpy as np 
import pandas as pd 
import os

# data that i need 
# - pitch locations (player_id, player_name, plate_x, plate_z)
# - pitch summaries (player_id, player_name, mean_velo, mean_pfx_x, 
# mean_pfx_z, mean_arm_angle, plate_x_std, plate_z_std) 
# - run values (from Thomas J. Nestico)
# - cars batter info 
# - dict connecting count to no. of balls and strikes 

os.chdir('/Users/justinchoi/SaberSem 2025 Python') 
os.environ["PATH"] += os.pathsep + '/opt/local/bin'
pd.set_option('display.max_columns', 500)

data = pd.read_csv("/Users/justinchoi/statcast_2024.csv") 
data = data.rename(columns = {'pitcher': 'pitcher_id', 'batter': 'batter_id', 
                              'player_name': 'pitcher_name', 
                              'stand': 'batter_hand', 'p_throws': 'pitcher_hand'})

# need to clean data / add vars before doing any joins or aggregation 

def clean_vars(df): 
    
    cols = ['pitcher_id','pitcher_name','batter_id','pitcher_hand','batter_hand',
             'release_speed','pitch_type','plate_x','plate_z','pfx_x','pfx_z',
             'balls','strikes','events','description','sz_top','sz_bot','arm_angle']
    
    df = df[(df['balls'] != 4) | (df['strikes'] != 3)]
    df = df[df['pitch_type'].isin(['FF','SL','CU','FC','SI','ST','CH','FS',
                                   'KC','SV','FO'])]
    df = df[cols]
    df = df.dropna(subset=['release_speed','balls','strikes','plate_x','plate_z',
                           'pfx_x','pfx_z','arm_angle'])
    
    return(df)


def add_vars(df): 
    
    df['swing_status'] = np.where(
        df['description'].isin(['hit_into_play','foul',
                                'swinging_strike','swinging_strike_blocked']), 
        'swing', 'take'
        ) 
    
    df['swing_outcome'] = np.where(
        df['description'] == 'hit_into_play', 'bip', 
        np.where(
            df['description'] == 'foul', 'foul', 'whiff'
            )
        )
    
    df['bbe_outcome'] = np.where(
        df['events'] == 'single', 'single', 
        np.where(
            df['events'] == 'double', 'double', 
            np.where(
                df['events'] == 'triple', 'triple', 
                np.where(
                    df['events'] == 'home_run', 'home_run', 'out' 
                    )
                )
            )
        )
    
    df['plat_adv'] = np.where(
        df['batter_hand'] == df['pitcher_hand'], 1, 0
        )
    
    df['pitch_group'] = np.where(
       df["pitch_type"].isin(["CH","FS","FO"]),
       "CH", np.where(
           df["pitch_type"].isin(["KC","CU","SV"]),
           "CU", df["pitch_type"]
           )
       )
        
    for col in ['plat_adv','pitcher_id','batter_id','pitch_type','pitch_group']: 
        df[col] = pd.Categorical(df[col]) 
        
    return df


data = clean_vars(data) 
data = add_vars(data)
   
cards = pd.read_csv("/Users/justinchoi/Downloads/cardinals_players.csv")[['player_id', 'player_name']]
cards = cards.rename(columns = {'player_id': 'batter_id', 'player_name': 'batter_name'})
cards_data = pd.merge(data, cards, how='inner', on='batter_id') 

pitch_locations = data[['pitcher_name','pitch_group','balls','strikes','plate_x','plate_z']]

pitch_summaries = (
    data
    .groupby(['pitcher_id','pitcher_name','pitcher_hand','pitch_group'])
    .agg(
        pitches = ('pitcher_id','size'), 
        mean_velo = ('release_speed', 'mean'), 
        mean_pfx_x = ('pfx_x', 'mean'), 
        mean_pfx_z = ('pfx_z', 'mean'), 
        mean_arm_angle = ('arm_angle', 'mean')
    )
    .query('pitches >= 10')
    .reset_index() 
  )

std_summaries = (
    data
    .groupby(['pitcher_id','pitcher_name','pitcher_hand','pitch_group','balls','strikes'])
    .agg(
        pitches = ('pitcher_id','size'), 
        plate_x_std = ('plate_x', 'std'), 
        plate_z_std = ('plate_z', 'std')
    )
    .query('pitches >= 10')
    .reset_index() 
  )

cards_info = (
    cards_data 
    .groupby(['batter_id','batter_name','batter_hand']) 
    .agg(
        sz_top = ('sz_top', 'mean'), 
        sz_bot = ('sz_bot', 'mean') 
    )
    .reset_index() 
  )

# dylan carlson is a switch hitter, so we need to distinguish between L and R 
cards_info.loc[(cards_info['batter_id'] == 666185) & 
               (cards_info['batter_hand'] == 'L'),'batter_name'] = "Carlson, Dylan (L)"

cards_info.loc[(cards_info['batter_id'] == 666185) & 
               (cards_info['batter_hand'] == 'R'),'batter_name'] = "Carlson, Dylan (R)"


pitch_locations.to_csv('data/pitch_locations.csv', index=False)
pitch_summaries.to_csv('data/pitch_summaries.csv', index=False)
std_summaries.to_csv('data/std_summaries.csv', index=False) 
cards_info.to_csv('data/cards_info.csv', index=False) 
cards_data.to_csv('data/cards_data.csv', index=False)
