import json 

model_dict = {
  "FF": {
    "swing_mod_path": "models/FF/swing_mod_idata.nc",
    "contact_mod_path": "models/FF/contact_mod_idata.nc"
  }, 
  "SI": {
    "swing_mod_path": "models/SI/swing_mod_idata.nc",
    "contact_mod_path": "models/SI/contact_mod_idata.nc"
  },
  "FC": {
    "swing_mod_path": "models/FC/swing_mod_idata.nc",
    "contact_mod_path": "models/FC/contact_mod_idata.nc"
  },
  "SL": {
    "swing_mod_path": "models/SL/swing_mod_idata.nc",
    "contact_mod_path": "models/SL/contact_mod_idata.nc"
  },
  "ST": {
    "swing_mod_path": "models/ST/swing_mod_idata.nc",
    "contact_mod_path": "models/ST/contact_mod_idata.nc"
  },
  "CU": {
    "swing_mod_path": "models/CU/swing_mod_idata.nc",
    "contact_mod_path": "models/CU/contact_mod_idata.nc"
  },
  "CH": {
    "swing_mod_path": "models/CH/swing_mod_idata.nc",
    "contact_mod_path": "models/CH/contact_mod_idata.nc"
  }
}

count_dict = {
    "0-0": {"balls": 0, "strikes": 0}, 
    "0-1": {"balls": 0, "strikes": 1}, 
    "0-2": {"balls": 0, "strikes": 2}, 
    "1-0": {"balls": 1, "strikes": 0}, 
    "2-0": {"balls": 2, "strikes": 0}, 
    "3-0": {"balls": 3, "strikes": 0}, 
    "1-1": {"balls": 1, "strikes": 1}, 
    "2-2": {"balls": 2, "strikes": 2}, 
    "1-2": {"balls": 1, "strikes": 2}, 
    "2-1": {"balls": 2, "strikes": 1}, 
    "3-1": {"balls": 3, "strikes": 1}, 
    "3-2": {"balls": 3, "strikes": 2}
}

with open("count_dict.json", "w") as outfile: 
    json.dump(count_dict, outfile, indent=4)

with open("model_dict.json", "w") as outfile: 
    json.dump(model_dict, outfile, indent=4)
    