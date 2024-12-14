# **Social Media Sentiment Analysis**

## **Table of Contents**
1. [**Project Overview**](#project-overview)
2. [**Importance**](#importance)
3. [**Process and Methodology**](#process-and-methodology)
4. [**Results and Findings**](#results-and-findings)
5. [**Example Prediction**](#example-prediction)
6. [**Limitations**](#limitations)
7. [**Future Work**](#future-work)
8. [**Contributing**](#contributing)
9. [**License**](#license)
10. [**Acknowledgments**](#acknowledgments)

## **Project Overview**

This project applies **Natural Language Processing (NLP)** techniques to analyze sentiment in text data. It includes:
- Data preprocessing
- Exploratory Data Analysis (EDA)
- Model building
- Evaluation

The focus is on understanding sentiment trends and evaluating the performance of various machine learning models.

---## **Introduction**

Social media sentiment analysis is a powerful tool to understand public opinion by analyzing the emotional tone behind text data. In this project, social media posts were classified into three sentiment categories: **Positive**, **Negative**, and **Neutral**. Using a labeled dataset sourced from Kaggle, the text data was preprocessed and key features were extracted using techniques like **TF-IDF**. Machine learning models, including **Logistic Regression** and advanced **Transformer-based models**, were applied to classify the sentiments effectively. 

Exploratory Data Analysis (EDA) provided insights into the sentiment distribution, platform trends, and user engagement metrics like likes and retweets. This project highlights how sentiment influences social media activity and demonstrates the application of Natural Language Processing (NLP) techniques to uncover actionable insights. The results offer valuable information for businesses, marketers, and researchers to understand audience sentiment in real time.


---

## **Dataset Description**

This dataset was sourced from **Kaggle** and contains **732 rows and 15 columns** with attributes such as text content, sentiment, timestamps, user details, and engagement metrics (likes and retweets).

Key columns include:
- **Text**: The social media post content.
- **Sentiment**: Classified as Positive, Negative, or Neutral.
- **Platform**: The social media platform of origin (e.g., Twitter, Instagram).
- **Retweets & Likes**: Engagement metrics.

- ### **Columns**
- **Text**: The content of the social media post.
- **Sentiment**: The sentiment classification of the post (Positive, Negative, or Neutral).
- **Timestamp**: The date and time of the post.
- **User**: The username of the individual who made the post.
- **Platform**: The platform where the post was made (e.g., Twitter, Instagram, Facebook).
- **Hashtags**: Hashtags associated with the post.
- **Retweets**: The number of retweets (if applicable).
- **Likes**: The number of likes the post received.
- **Country**: The country where the post originated.
- **Year, Month, Day, Hour**: Additional temporal attributes derived from the timestamp.

### **Dataset Details**
- **Total Rows**: 732
- **Total Columns**: 15
- **Missing Values**: None

The dataset is complete with **no missing values**.

### **Sample Data**
| **Text**                                      | **Sentiment** | **Timestamp**         | **User**       | **Platform** | **Hashtags**         | **Retweets** | **Likes** | **Country**   | **Year** | **Month** | **Day** | **Hour** |
|-----------------------------------------------|---------------|-----------------------|----------------|--------------|-----------------------|--------------|-----------|--------------|----------|-----------|---------|----------|
| Enjoying a beautiful day at the park!         | Positive      | 2023-01-15 12:30:00  | User123        | Twitter      | #Nature #Park         | 15           | 30        | USA          | 2023     | 1         | 15      | 12       |
| Traffic was terrible this morning.            | Negative      | 2023-01-15 08:45:00  | CommuterX      | Twitter      | #Traffic #Morning     | 5            | 10        | Canada       | 2023     | 1         | 15      | 8        |
| Just finished an amazing workout! 💪          | Positive      | 2023-01-15 15:45:00  | FitnessFan     | Instagram    | #Fitness #Workout     | 20           | 40        | USA          | 2023     | 1         | 15      | 15       |
| Excited about the upcoming weekend getaway!   | Positive      | 2023-01-15 18:20:00  | AdventureX     | Facebook     | #Travel #Adventure    | 8            | 15        | UK           | 2023     | 1         | 15      | 18       |
| Trying out a new recipe for dinner tonight.   | Neutral       | 2023-01-15 19:55:00  | ChefCook       | Instagram    | #Cooking #Food        | 12           | 25        | Australia    | 2023     | 1         | 15      | 19       |




## **Project Overview**

This project applies **Natural Language Processing (NLP)** techniques to analyze sentiment in text data. It includes:
- Data preprocessing
- Exploratory Data Analysis (EDA)
- Model building
- Evaluation

The focus is on understanding sentiment trends and evaluating the performance of various machine learning models.

---

## **Structure of the Notebook**

### **1. Data Loading**
- The dataset is uploaded and loaded using **Pandas**.
- Initial inspections include:
  - Dataset information
  - Null values
  - A preview of the data

### **2. Exploratory Data Analysis (EDA)**
- **Visualizations** to understand the distribution of sentiments.
- Analysis of:
  - Sentiment proportions
  - Platform-based sentiment distributions
- Engagement analysis based on metrics like retweets and likes.

### **3. Data Preprocessing**
- Text cleaning and tokenization.
- Feature extraction using `TfidfVectorizer`.

### **4. Model Building**
- Logistic Regression and Transformer-based models (using the `transformers` library) for sentiment classification.
- Dataset splitting using `train_test_split`.

### **5. Evaluation**
- Visualizations and metrics to evaluate model performance.

---

## **Key Features**

### **Visualization**
- Bar charts for sentiment distribution.
- Stacked bar charts for platform-based sentiment analysis.
- Boxplots to show engagement metrics (likes and retweets) by sentiment.

### **Libraries Used**
- **Data manipulation**: `pandas`, `numpy`
- **Visualization**: `matplotlib`, `seaborn`
- **Machine Learning**: `sklearn` (Logistic Regression, scaling, train-test split)
- **NLP**: `TfidfVectorizer` and Transformer models (`transformers`)

---

## **Requirements**

To run the notebook, install the following libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn transformers
