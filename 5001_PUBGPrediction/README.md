# MSBD5001 Project - Battle Royale-style games Ranks Prediction

Project Source: https://www.kaggle.com/c/pubg-finish-placement-prediction

Data Source: https://developer.pubg.com/

## Proposal

### Project Description: 

#### 1. Problem exploring with your dataset

Battle Royale-style video games have taken the world by storm. 100 players are dropped onto an island empty-handed and must explore, scavenge, and eliminate other players until only one is left standing, all while the play zone continues to shrink.

The team at [PUBG](https://www.pubg.com/) has made official game data available for the public to explore and scavenge outside of "The Blue Circle." This competition is not an official or affiliated PUBG site .

Over 65,000 games' worth of anonymized player data, we try to predict final placement from final in-game stats and initial player ratings.

#### 2. Expected outcome of your project

- Win place prediction

- Analysis of different models' performance


### Data Set

#### 1. Data Description:

In a PUBG game, up to 100 players start in each match (matchId). Players can be on teams (groupId) which get ranked at the end of the game (winPlacePerc) based on how many other teams are still alive when they are eliminated. In game, players can pick up different munitions, revive downed-but-not-out (knocked) teammates, drive vehicles, swim, run, shoot, and experience all of the consequences -- such as falling too far or running themselves over and eliminating themselves.

There are a large number of anonymized PUBG game stats, formatted so that each row contains one player's post-game stats. The data comes from matches of all types: solos, duos, squads, and custom.

#### 2. Some Important Data infos

- **id** - players id

- **groupId** - Integer ID to identify a group within a match. If the same group of players plays in different matches, they will have a different groupId each time.

- **matchId** - Integer ID to identify match. There are no matches that are in both the training and testing set.

- **damageDealt** - Total damage dealt. Note: Self inflicted damage is subtracted.

- **Items_Used** - Number of healing & boost items used.

- **killPlace** - Ranking in match of number of enemy players killed.

- **Kills_info** - longest kill, headshot, kill streaks, assists, kill place, kill points

- **Supports** - Number of  reviving. 

- **weaponsAcquired** - Number of weapons picked up.

- **Distance** -  Total distance traveled in walking, swimming, riding measured in meters.

- **<u>winPlacePerc</u>** - The target of prediction. This is a percentile winning placement, where 1 corresponds to 1st place, and 0 corresponds to last place in the match. It is calculated off of maxPlace, not numGroups, so it is possible to have missing chunks in a match.



### Blueprint

#### EDA

- Inner-Group aggregation
- game modes 
  - Groups mode
    - Squad: < 20
    - Double:  <50
    - Solo:  <100
    - Zombie mode detection
- outliers
  - cheaters
  - Empty values
- Correlation analysis
- Data-distribution analysis

#### Feature Engeeing

- Feature selection
- Continuous features discretization

- Extra-Group-features aggregation stats
  - max
  - min
  - Mean
  - ...

#### Modeling

1. Approach1: Group by GroupID/MatchID - Aggreation Data - Train - Test
   1. Linear Regression ( base line )
   2. Decision tree
   3. Ensemble learning
   4. Neural network

2. Approach 2 : Add new features from grouped/matchid - Train - Test

#### 

##### Referrence links:

* Linear Regression: https://www.kaggle.com/sachinjchorge/pubg-linear-regression
* Simple NN: https://www.kaggle.com/anycode/simple-nn-baseline
* Keras Simple NN: https://www.kaggle.com/amoeba3215/keras-simple-nn-baseline
* Overall EDA: https://www.kaggle.com/datark1/pugb-overall-eda-top-10-players/data
* EDA: https://www.kaggle.com/deffro/eda-is-fun
* Cheaters: https://www.kaggle.com/rejasupotaro/cheaters-and-zombies

## EDA / Feature Engineering Talk Big Meeting!
	
1. Outliers (according to distribution)

    a. Cheat_1: Total distance (riding + swimming + walk) < 0 & kills > 0<p>
    b. Cheat_2: Road Kills > 10<p>
    c. Cheat_3: Head shot rate = 1 & kills > 9<p>
    d. Cheat_4: longest kill > 1000<p>
    e. Cheat_5: weapons > 80<p>
    f. Cheat_6: heal > 40<p>
    g. **Cheat_7: avg_walk_speed > 6 (= walkDistance / matchDuration)**<p>
    h. **FPP / TPP : 1 - cheater rate = 0.9999 / 0.999**<p>

2. Feature Engineer<p>
    a. Head shot rate. = Head shoot / kills. ( linear drop, tree keep)<p>
    b. Heals + Boost<p>
    c. One hot coding Match Type<p>
    d. **Damage per enemy = Damage / (kills + assists)**<p>
    e. **solo / squad / duo / fpp-tpp - one hot encode**<p>
    f. **vehicleDestroys: 0 / 1**<p>
3. **Remove WTF mode**<p>
    ( as outliers.  len(not_wtf_train) / len(df_train) = 0.992) <p>
    a. Normal-* modes<p>
        i. Custom mode.<p>
        ii. have totally different method to calculate winPlacePerc.<p>
    b. crashfpp / crashtpp. 肉搏模式<p>
    c. flarefpp / flaretpp. 信号枪模式<p>
4. Pre-process:<p>
    1. Outliers (cheaters detection)<p>
    2. Feature Engineer<p>
    3. Group by group-Id<p>
    4. Standardization by match.<p>
    5. Train model by different strategy<p>
        i. Tree & ensemble && NN : one hot encoding match type & train single model.<p>
        ii. Linear & SVR : train multiple model for different match type.<p>
5. [Damage Dealt by 0 killers] compare with cheated features' distribution.<p>
6. 躺赢玩家展示<p>
