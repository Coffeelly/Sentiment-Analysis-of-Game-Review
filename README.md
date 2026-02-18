# Steam Review Intelligence: Mining Business Insights from User Feedback

![Project Banner Placeholder](https://via.placeholder.com/1000x300?text=Steam+Review+Intelligence+Project)

## Executive Summary

In the competitive gaming industry, user feedback is abundant but often unstructured. While standard sentiment analysis classifies reviews as Positive or Negative, this project goes deeper to understand the **"Why"** behind the ratings.

By processing the full dataset of over **6.4 million Steam reviews**, this project utilizes Natural Language Processing (NLP) to extract actionable business insights. It identifies specific drivers of player churn (e.g., predatory monetization, technical instability) and retention (e.g., social value, immersion), providing a data-driven strategy for game development and community management.

---

This study was conducted using [Google Colaboratory](https://colab.research.google.com/)

## Data & Methodology

### Data Source

- **Dataset:** Steam Reviews Dataset (Kaggle : https://www.kaggle.com/datasets/andrewmvd/steam-reviews/data).
- **Volume:** ~6.4 Million Rows (Full dataset used for N-Gram analysis to capture the total volume of user sentiment).

### Advanced Preprocessing Pipeline

To extract meaningful keywords from millions of rows of noisy text, a rigorous cleaning pipeline was implemented:

1.  **Noise Reduction:** Removal of HTML tags, URLs, Emojis, and special symbols.
2.  **Normalization:** Lowercasing and removing punctuation/numbers to standardize text.
3.  **Domain-Specific Filtering:**
    - **Custom Stopwords:** Removed generic domain words (e.g., `'game'`, `'play'`) to prevent them from obscuring specific insights.
4.  **Linguistic Standardization (Lemmatization):**
    - Used `WordNetLemmatizer` instead of Stemming.
    - _Why?_ To ensure the output remains readable for business stakeholders (e.g., transforming "Optimized" -> "Optimize" rather than cutting it to "Optim").

## Key Business Insights
* **Bigram**

<img width="1989" height="790" alt="Imae" src="https://github.com/user-attachments/assets/689d4f3f-fb83-43e0-85cc-777d70d48523" />

* **Trigram**

<img width="1990" height="790" alt="image" src="https://github.com/user-attachments/assets/67b640e3-3948-4af9-9c65-cd8470104bbc" />


Based on **Bigram & Trigram Analysis** of the full dataset, the following patterns emerged as the primary drivers of user sentiment.

1. **The "Churn Drivers" (Why Players Leave)**
   Analysis of the top negative N-Grams reveals that users rarely quit due to "bad gameplay" alone. The major deal-breakers are:

- **Predatory Monetization (The #1 Complaint):**
  - Top Keywords: `pay win`, `waste money`, `dlc dlc dlc`, `want money back`.
  - Insight: Financial frustration is the strongest negative driver. Players react aggressively to "Pay-to-Win" mechanics or content that feels like a cash grab.

- **Technical Instability:**
  - Top Keywords: `crash crash crash`, `every single time`.
  - Insight: Technical failures lead to immediate refund requests. Stability is a prerequisite for retention.

- **The "Disappointed Fan":**
  - Top Keywords: `really wanted like`, `seems like`.
  - Insight: Captures the sentiment of players who were hyped by marketing but let down by the actual product experience.

2. **The "Retention Drivers" (Why Players Stay)**

- **Social Value is King:**
  - Top Keywords: `pay win`, `waste money`, `dlc dlc dlc`, `want money back`.
  - Insight: Multiplayer experiences are the stickiest feature. "Fun with friends" often overrides minor gameplay flaws.

- **High Perceived Value:**
  - Top Keywords: `worth every penny`, `one best ever`.
  - Insight: When content volume matches the price tag, sentiment soars.

- **Immersion:**
  - Top Keywords: `voice acting`, `open world`.
  - Insight: Immersion is vital in video games because it fosters a, deep, emotional, and psychological "suspension of disbelief," allowing players to feel as though they are genuinely experiencing the virtual world rather than just observing it.

## Comparative Case Study: PAYDAY 2 vs. Left 4 Dead 2

A head-to-head analysis of two major **Co-op Shooters** reveals why one remains beloved while the other faces backlash, despite sharing the same genre.

- **PAYDAY 2: The Trust Crisis**
  - **Top Negative Trigrams:** `shame thought otherwise`, `made clear payday`, `micro transaction whatsoever`, `completely overkill pack`.
  - **Analysis:** The negative sentiment is highly specific and political. It points directly to a scandal where the developers broke a promise regarding microtransactions.
  - **Business Impact:** The data shows that users feel betrayed. The sentiment isn't about the game being "unfun," but about the developer being "untrustworthy."

- **Left 4 Dead 2: Operational Noise**
  - **Top Negative Trigrams:** `shame thought otherwise`, `made clear payday`, `micro transaction whatsoever`, `completely overkill pack`.
  - **Analysis:** Complaints are either regulatory (censored versions in specific countries), social (toxic teammates), or non-specific noise.
  - **Business Impact:** There are no fundamental complaints about the game design or business model. As a result, the game maintains a massive positive reputation compared to PAYDAY 2.

## Predictive Modeling Preparation

To transition from exploratory analysis to machine learning, the dataset underwent a strict transformation pipeline to ensure model integrity and prevent bias.

* **Trigram**
<img width="1990" height="790" alt="image" src="https://github.com/user-attachments/assets/3952c235-03d9-41a9-bb27-54b0cee0f59e" />


1. **The "High-Quality" Filter**

- **Logic:** `df[df['review_votes'] == 1]`
- **Rationale:** Strictly selected reviews that were explicitly voted as "Helpful" by the community. This eliminates low-effort spam and ensures the model learns from high-quality, human-validated feedback.

2. **Handling Class Imbalance (Undersampling)**

- **Problem:** Real-world steam reviews are heavily skewed towards Positive sentiment, which causes models to be biased towards predicting "Positive."
- **Solution:** Strictly selected reviews that were explicitly voted as "Helpful" by the community. This eliminates low-effort spam and ensures the model learns from high-quality, human-validated feedback.
  - We retained all valid Negative reviews (Minority Class).
  - We randomly sampled the Positive reviews (Majority Class) to match the exact count of the negative reviews.
  - Result: A perfectly balanced 50/50 dataset. This forces the model to learn the distinct features of negative reviews rather than just guessing "Positive" based on probability.

3. **Data Integrity Checks**

- **Deduplication:** Removed duplicate text entries to prevent Data Leakage (where the same review appears in both Training and Testing sets).
- **Shuffling:** The balanced dataset was shuffled (`random_state=42`) to destroy any inherent ordering bias before splitting.

4. **Final Feature Selection**
   The final dataset for training retains four critical columns:

- `text_clean`: The preprocessed text (Lemmatized) for the model features (X).
- `label`: The target variable (0 for Negative, 1 for Positive).
- `review_text`: The raw text (kept for manual error analysis and debugging).
- `app_name`: To analyze model performance across specific games.

## 🤖 Predictive Modeling Phase

To automate sentiment classification, I compare multiple models. Instead of relying on a single algorithm, I benchmarked three distinct architectures to find the best balance between accuracy and efficiency.

### Experimental Setup

- **Feature Engineering:** TF-IDF Vectorization (Top 10,000 Features, N-Gram 1-2).
- **Data Split:** 80% Training / 20% Testing (Stratified).
- **Metric:** Accuracy (on the unseen Test Set).

### Model Candidates

1.  **Logistic Regression (Baseline):** Chosen for its speed and interpretability in high-dimensional sparse data.
2.  **LightGBM (The Challenger):** A Gradient Boosting Decision Tree (GBDT) model, typically state-of-the-art for tabular data.
3.  **Simple Neural Network (Deep Learning):** A Multi-Layer Perceptron (MLP) with Dropout layers to capture non-linear relationships.

### 🏆 Results & Evaluation

| Model Architecture       | Test Accuracy | Performance Notes                                                                                    |
| :----------------------- | :------------ | :--------------------------------------------------------------------------------------------------- |
| **Neural Network (MLP)** | **83.14%**    | 🥇 **Champion.** Successfully captured complex non-linear patterns.                                  |
| **Logistic Regression**  | 82.77%        | 🥈 **Runner-up.** Extremely efficient and fast, only 0.37% behind the deep learning model.           |
| LightGBM                 | 78.55%        | Struggled with the high-dimensional sparse nature of TF-IDF vectors compared to linear/dense models. |

### 🧠 Model Analysis

- **The "Deep Learning" Edge:** The Neural Network slightly outperformed the baseline, likely due to its ability to understand complex negations (e.g., _"not bad at all"_) better than a linear model.
- **The Efficiency of Linearity:** Logistic Regression proved to be a robust production candidate. If deployment resources are limited, the 0.37% drop in accuracy is a worthy trade-off for 10x faster inference speed.
- **Why LightGBM Failed:** Tree-based models generally struggle with very wide, sparse datasets (10,000+ features) where information is spread thin across dimensions, unlike Linear models or NNs which handle high-dimensionality natively.
