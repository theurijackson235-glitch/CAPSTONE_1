# CAPSTONE_1
US Accidents
# US Accidents Capstone: Understanding and Predicting Crash Severity

**Author:** Jackson Theuri Mururi
**Methodology:** CRISP-DM
**Dataset:** A 500,000-record sample of the US Accidents (2016–2023) dataset

---

## What this project is about

Every day, thousands of road accidents happen across the United States, and not all of them are equal — some are minor fender-benders, others shut down highways and require emergency response. This project asks a simple but important question: **can we understand, and eventually predict, how severe an accident is likely to be, based on the conditions surrounding it?**

Think about it from the perspective of a state Department of Transportation or an emergency response center. If you knew that icy conditions at 7 a.m. on a particular highway tend to produce more severe crashes than average, you could position resources differently. That's the practical motivation behind this work — not just building a model for the sake of it, but trying to surface patterns that could genuinely help someone make a better decision.

The project is split into two phases that build on each other, following the CRISP-DM methodology (a fairly standard way of structuring data science projects: understand the business problem, understand the data, prepare it, explore it, model it, evaluate it).

---

## Phase 1 — Exploratory Analysis and Insights

**File:** `US_Accidents_Phase1_Capstone_Notebook.ipynb`

This is where the groundwork gets laid. Before jumping into any modeling, it's important to actually sit with the data and understand its shape, quirks, and story. This notebook:

- Loads and inspects the raw dataset (structure, missing values, duplicates)
- Engineers useful features from the raw timestamp — hour of day, month, season, day of week, and whether the accident happened at night
- Explores how severity is distributed (spoiler: it's heavily imbalanced — most accidents fall into the "moderate" Severity 2 bucket)
- Looks at how accidents vary by time of day, weather condition, and state
- Runs statistical tests (Chi-square and ANOVA) to check whether the patterns we see in the charts are likely to be real relationships, not just noise

By the end of Phase 1, four business questions have been explored:

1. Which weather conditions are associated with severe accidents?
2. Does accident severity change during the day?
3. Which states experience the largest number of severe crashes?
4. Which road features appear most frequently in serious accidents?

It closes with a plan for a Tableau dashboard (coming next) and lays out exactly which features make sense to carry into Phase 2's predictive model — reasoning carefully about which variables are known *before or during* an accident, so the model doesn't accidentally "cheat" by using information it wouldn't have in a real, live scenario.

---

## Phase 2 — Predictive Modelling

**File:** `US_Accidents_Phase2_Predictive_Modelling.ipynb`

This is where the project tries to turn insight into prediction. Given weather, time, location, and road features, can we predict how severe a new accident will be?

A few decisions worth calling out, because they matter more than they might seem:

- **The metric chosen is Weighted F1-score, not accuracy.** With four severity classes and a dataset skewed heavily toward "Severity 2," a model could get high accuracy just by guessing that category every time — which would be useless in practice. Weighted F1 forces the model to actually do reasonably well across all classes.
- **Two models are compared:** a Logistic Regression baseline (simple, interpretable, a sanity check) and a Random Forest (better at capturing nonlinear interactions between weather, location, and road features). The Random Forest is expected to justify its added complexity by outperforming the baseline.
- **Everything runs through a scikit-learn Pipeline**, meaning preprocessing (imputing missing values, scaling numeric features, one-hot encoding categories) is learned only from the training data and never leaks information from the test set.
- **Cross-validation** checks whether the Random Forest's performance is stable, rather than trusting a single lucky (or unlucky) train/test split.
- **Feature importance and a confusion matrix** are used to go beyond "the model got X% right" and actually understand *what* it's learning and *where* it struggles — for instance, confusing a Severity 3 with a Severity 4 accident is a much bigger operational problem than confusing a 1 with a 2.

The notebook is honest about the model's limits too: class imbalance, missing weather data, and the fact that a sampled dataset can't perfectly represent every accident in the country. This isn't a model meant to replace human judgment — it's a decision-support tool meant to sit alongside it.

---

## What ties the two phases together

Phase 1 doesn't just describe the data — it actively sets up Phase 2. Every feature used in the predictive model (temperature, visibility, humidity, wind speed, weather condition, hour, month, state, traffic signal presence, junction presence) was chosen in Phase 1 specifically because it's information available *before or during* the accident, not after. That distinction is what keeps the model honest and useful in a real-world setting, rather than just performing well on paper.

---

## What's next

- **A Tableau dashboard**, so that the insights from Phase 1 aren't locked inside a notebook but can actually be explored interactively by non-technical stakeholders (state DOTs, emergency response planners) — filtering by state, weather, or time of day themselves.
- **Phase 3**, which will take the `Description` field — a free-text narrative accompanying nearly every accident record — and explore whether a text classification model can predict severity from the description alone, evaluated with Accuracy, Precision, Recall, and F1-score.

---

## Getting set up

If you want to run these notebooks yourself:

1. Place the source dataset (`US_Accidents_March23_sampled_500k.zip`) in your data folder.
2. Update the file path at the top of each notebook to point to that folder on your machine.
3. Run Phase 1 first — it produces the cleaned CSV (`us_accidents_clean.csv`) used both for the Tableau dashboard and, indirectly, as the basis for Phase 2's feature choices.
4. Run Phase 2 to train and evaluate the models.

**Libraries used:** pandas, numpy, matplotlib, scipy (`chi2_contingency`, `f_oneway`), and scikit-learn (`train_test_split`, `Pipeline`, `ColumnTransformer`, `LogisticRegression`, `RandomForestClassifier`, and related metrics).

---

## A closing thought

Data on car accidents is, in the end, data about real people having some of the worst moments of their day. It's easy to get lost in F1-scores and p-values and forget that. The goal of this project has always been practical: give the people responsible for keeping roads safe a clearer picture of where and when things go wrong, so they can act on it — not to build the fanciest model, but the most useful one.
