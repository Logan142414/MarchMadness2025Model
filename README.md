# March Madness 2025: Predicting Tournament Games with Machine Learning

### *Machine Learning Project by Logan Laszewski*

**Project Goal:**  
March Madness is one of the most unpredictable events in sports. Each year, fans hope to fill out the perfect bracket, only to see upsets shatter predictions. Combining my passion for basketball with my growing expertise in data analytics and machine learning, I set out to build a model to predict tournament outcomes — focusing on uncovering which team statistics truly drive success.

---

## Project Overview

This project uses historical NCAA data from Kaggle’s [March Machine Learning Mania](https://www.kaggle.com/c/march-machine-learning-mania-2025) challenge. While I did not submit competition entries, the datasets provided a solid foundation to explore predictive modeling for tournament games.

**Data Split:**

Training Data | 2003–2020 tournaments
Test Data | 2021–2024 tournaments

This split evaluates the model’s ability to generalize to recent tournaments.

---

## Data Cleaning & Transformation
Raw game data required extensive cleaning to be usable for modeling.

**Key Steps:**
* Handled missing values and inconsistencies
* Restructured game-level stats into season-level summaries per team
* Created new features to capture team performance dynamics
* Normalized and scaled variables for optimal model performance

**Aggregating Team-Level Stats:**  
Winner/loser stats were transformed into a consistent, team specific perspective. Each team’s season level totals were computed for points, rebounds, assists, turnovers, and other key metrics. Additional information, such as tournament seed and conference, was merged to enrich the dataset.

---

## Feature Engineering

To capture deeper performance insights:

* Calculated advanced metrics like effective field goal percentage (eFG%), free throw rate, and estimated possessions
* Created **statistical margins**: differences between team performance and what opponents allowed, normalized by competition level
* Built matchup-level features: For each game, team margin differences were compared head-to-head, forming the core inputs for modeling

**Final Dataset:**  
After feature engineering, the predictor set was distilled to key variables, focusing purely on performance signals, excluding obvious indicators like seed, win-loss record, or conference.

---

## Modeling

### XGBoost

* Gradient boosting algorithm selected for capturing non-linear relationships between performance metrics
* Training set included tournament games (2003–2020) with engineered features
* **Evaluation:**
  * Training Accuracy: ~69%
  * Test Accuracy: ~60%
  * Log Loss: 0.66, AUC: 0.64

Key predictive metrics included field goals made margin, turnover margin, and free throw attempts margin.

### H2O AutoML

* Explored multiple models with automated cross-validation
* Surprisingly, a Generalized Linear Model (GLM) outperformed XGBoost
* **GLM Performance:**
  * Training Accuracy: ~71%
  * Test Accuracy: ~69%
  * Log Loss: 0.55, AUC: 0.78

This highlighted the power of well-engineered features over model complexity.

---

## Key Takeaways

1. **Data Quality is Foundational:** Transforming game-level stats into team-season summaries was essential
2. **Feature Engineering Drives Value:** Margin-based and head-to-head variables significantly improved predictive power
3. **Simplicity Sometimes Wins:** A simple GLM outperformed complex models like XGBoost
4. **Iterative Modeling:** Continuous evaluation and tuning of predictors was critical
5. **March Madness is Unpredictable:** Even with strong models, upsets make perfect predictions impossible — reinforcing the value of identifying patterns rather than achieving perfection

---

## Datasets & Code

**Datasets Used:**

* `MNCAATourneySeeds.csv` – Tournament seed information  
* `MTeams.csv` – Team IDs and names  
* `MRegularSeasonDetailedResults.csv` – Game-level stats (points, rebounds, turnovers, etc.)  
* `MTeamConferences.csv` – Team conference affiliation  
* `MNCAATourneyCompactResults.csv` – Tournament outcomes for training/testing  
* `MNCAATourneySlots.csv` – 2025 tournament matchups  
* `MMasseyOrdinals.csv` – End-of-season rankings (KenPom ratings)

**Core Libraries & Frameworks:**

* pandas, numpy, matplotlib  
* scikit-learn, xgboost, h2o  
* scipy, h2o.automl  

---

## Next Steps

* Explore additional feature engineering, such as player-level contributions and injury impacts
* Incorporate team momentum and advanced efficiency metrics
* Test ensemble methods and neural networks for potential performance gains
* Evaluate models with tournament simulation approaches for probabilistic bracket predictions
* Continuously update models with new seasons to improve predictive accuracy over time

---

## Python Code
- [Script](https://colab.research.google.com/drive/1mq2hmqPqFY6gWli_lPyTiJCV-bKac8Kr?usp=sharing)

---

## Medium Article
- [Modeling March Madness: A Machine Learning Approach to Predicting Tournament Games](https://medium.com/@logan.laszewski14/modeling-march-madness-a-machine-learning-approach-to-predicting-tournament-games-db8bc7b74a1d)


