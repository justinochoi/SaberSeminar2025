# SaberSeminar2025

This is the repository for a presentation given at the 2025 Saberseminar in Chicago entitled "A Method for Optimizing Pitch Targets." 

To optimize pitch targets, we first create a measure of pitch value. We do this by calculating the probabilities of events that can happen after a pitch is thrown (e.g. whiff, foul, ball-in-play) using variables such as the count, pitch movement, location, and most importantly, a batter random effect. 

Next, we create an estimate of pitcher command via the posterior covariance matrix of a pitcher's pitch locations. We place a Gamma distribution prior, as determined by the league-wide data, on the standard deviation components (x- and z-locations of the pitches). The priors are unique up to the pitch type and number of balls and strikes. We place a uniform prior on the covariance between the x- and z-locations, and let the pitches themselves determine this covariance.  

Finally, given the batter, pitcher, pitch type, and count, we iterate through a list of candidate pitch targets. For each target, we simulate 500 pitches by assuming they follow a bivariate normal distribution, with the target as the mean vector and the estimated posterior covariance matrix. Using the event probability models, we calculate the run value of these simulate pitches, then take their average. The target that returns the best average run value is the optimal target. 

The repository is organized as follows:

- The getting_data.py file shows how the necessary data was obtained and cleaned. 
- The model_dict.json and count_dict.json files are dictionaries used to map pitch types and counts to models and run values, respectively. 
- The called_strike_model.py file shows how the model used to calculate the strike probability of a pitch was constructed. 
- The swing_models.py, contact_models.py, and in-play_models.py files show the formulas of the event probability models. Note that these do not include important checks such as model convergence or posterior predictiveness, but rest assured that all the models passed them with flying colors. 
- The streamlit_app.py shows the process of estimating the posterior covariance matrix, simulating pitches, and calculating their run values. The app is still a work in progress, as only Cardinals hitters are available to try out for now. It can be accessed using this link: https://pitch-target-optimizer.streamlit.app/

