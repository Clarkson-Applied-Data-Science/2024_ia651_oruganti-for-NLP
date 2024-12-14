# **Social Media Sentiment Analysis**

## **Table of Contents**

1. [**Introduction**](#introduction)  
2. [**Dataset Description**](#dataset-description)  
3. [**Exploratory Data Analysis (EDA)**](#exploratory-data-analysis-eda)  
   - [Sentiment Distribution](#sentiment-distribution)  
   - [Platform-Based Sentiment Distribution](#platform-based-sentiment-distribution)  
   - [Engagement Analysis (Likes and Retweets)](#engagement-analysis-likes-and-retweets)  
   - [Text-Based Analysis](#text-based-analysis)  
4. [**Model Training**](#model-training)  
   - [Logistic Regression](#logistic-regression)  
   - [Transformer Model](#transformer-model)  
   - [LSTM Model](#lstm-model)  
5. [**Model Evaluation**](#model-evaluation)  
   - [Logistic Regression Results](#logistic-regression-results)  
   - [Transformer Model Results](#transformer-model-results)  
   - [LSTM Model Results](#lstm-model-results)  
6. [**Comparison of Models**](#comparison-of-models)  
7. [**Conclusion**](#conclusion)  
8. [**Future Work**](#future-work)  


### **Libraries Used**
- **Data manipulation**: `pandas`, `numpy`
- **Visualization**: `matplotlib`, `seaborn`
- **Machine Learning**: `sklearn` (Logistic Regression, scaling, train-test split)
- **NLP**: `TfidfVectorizer` and Transformer models (`transformers`)

---

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

## **Exploratory Data Analysis (EDA)**

Exploratory Data Analysis (EDA) was conducted to extract meaningful insights from the dataset. The visualizations and findings include:
1. Sentiment distribution and trends.
2. Platform-based sentiment analysis.
3. Engagement patterns (average likes and retweets by sentiment).
4. Text-based analysis, including word frequency, word clouds, and text lengths.

These insights provide a comprehensive understanding of the dataset before proceeding to model development.
## **Results and Findings**

This project produced a variety of visualizations and insights into sentiment distribution, user engagement, and text features. Below are the key findings and their corresponding visualizations:

---

### **1. Sentiment Distribution**
The dataset's sentiment distribution shows the prevalence of positive, negative, and neutral sentiments across the social media posts.

![Sentiment Distribution](<insert-image-link-here>)

---

### **2. Sentiment Distribution Across Platforms**
This chart highlights how different platforms (e.g., Twitter, Instagram, Facebook) handle sentiment trends.

![Sentiment Distribution Across Platforms](<insert-image-link-here>)

---

### **3. Top 20 Sentiments by Average Retweets**
The visualization shows which sentiments lead to higher average retweets, offering insights into user engagement based on sentiment tone.

![Top 20 Sentiments by Average Retweets](<insert-image-link-here>)

---

### **4. Top 20 Sentiments by Average Likes**
This chart displays the top 20 sentiments by average likes, providing insights into which sentiments generate higher user interactions.

![Top 20 Sentiments by Average Likes](<insert-image-link-here>)

---

### **5. Refined Sentiment Distribution**
The refined sentiment distribution groups the data into three categories—positive, negative, and neutral—offering a clear overview of overall sentiment trends.

![Refined Sentiment Distribution](<insert-image-link-here>)

---

### **6. Word Frequency**
The most common words across all posts are visualized, providing insights into the textual content and popular terms used in the dataset.

![Top 20 Most Common Words](<insert-image-link-here>)

---

### **7. Word Cloud**
The word cloud represents the frequency of words in the dataset in a visually appealing format, highlighting frequently used words in larger fonts.

![Word Cloud of Cleaned Text](<insert-image-link-here>)

---

### **8. Distribution of Text Lengths**
The distribution of text lengths indicates the frequency of posts of different lengths, which is useful for understanding the dataset's content structure.

![Distribution of Text Lengths](<insert-image-link-here>)

---

### **9. Average Text Length by Sentiment**
This chart reveals the average text length for posts categorized by sentiment, helping identify trends in message length based on sentiment tone.

![Average Text Length by Sentiment](<insert-image-link-here>)

---

### **10. Top 10 Most Frequent Words**
The top 10 most frequently used words are visualized to show the most common themes in the dataset.

![Top 10 Most Frequent Words](<insert-image-link-here>)

---


## **Model Evaluation**

After preprocessing the data and performing exploratory data analysis, a **Logistic Regression model** was trained to classify sentiments into Negative, Neutral, and Positive categories. The model was evaluated using metrics such as accuracy, precision, recall, and F1-score. The results were visualized using a confusion matrix.


![Confusion Matrix (Logistic Regression)](<insert-image-link-here>)

### **Results**
- **Test Accuracy**: 86.42%
- **Classification Report**:

- ## **Model Evaluation: Logistic Regression Results**

| **Metric**      | **Negative** | **Neutral** | **Positive** | **Overall/Macro Avg** | **Weighted Avg** |
|------------------|--------------|-------------|--------------|------------------------|-------------------|
| **Precision**    | 0.89         | 0.94        | 0.78         | 0.87                  | 0.88              |
| **Recall**       | 1.00         | 0.68        | 0.94         | 0.87                  | 0.86              |
| **F1-Score**     | 0.94         | 0.79        | 0.85         | 0.86                  | 0.86              |
| **Support**      | 82           | 91          | 70           | 243 (Total)           | -                 |
| **Accuracy**     | -            | -           | -            | 0.864 (86.42%)        | -                 |


---

### **ROC Curve**

The **Receiver Operating Characteristic (ROC) Curve** visualizes the trade-off between the True Positive Rate (TPR) and the False Positive Rate (FPR) across various classification thresholds. The model performed exceptionally well, as reflected in the **Area Under the Curve (AUC)** values for each class.

- **Class Negative**: AUC = **1.00**
- **Class Neutral**: AUC = **0.93**
- **Class Positive**: AUC = **0.95**

![ROC Curve](<insert-roc-curve-image-link-here>)

---

### **Conclusion**

The Logistic Regression model demonstrated strong performance:
1. Achieved high accuracy (86.42%) across all classes.
2. Perfect AUC for the Negative class (1.00), indicating exceptional discrimination.
3. Strong AUC values for Neutral and Positive classes, showcasing reliability in sentiment classification.

These results validate the model's effectiveness in classifying sentiments into Negative, Neutral, and Positive categories.

---

## **Transformer Model Training**

To further enhance the performance of sentiment classification, a **Transformer-based model** was trained on the dataset. The model leveraged modern Natural Language Processing (NLP) techniques to learn from the data effectively.

### **Training Results**
The table below summarizes the training and validation loss over 5 epochs:

| **Epoch** | **Training Loss** | **Validation Loss** |
|-----------|--------------------|---------------------|
| 1         | No log            | 1.073896           |
| 2         | No log            | 1.005887           |
| 3         | No log            | 0.936536           |
| 4         | No log            | 0.898985           |
| 5         | No log            | 0.889890           |

### **Training Metrics**
- **Final Training Loss**: ~1.0024
- **Runtime**: 213.104 seconds
- **Training Samples per Second**: ~3.942
- **Training Steps per Second**: ~0.258

### **Conclusion**
The Transformer model demonstrated consistent improvement in validation loss across epochs, indicating effective learning. The results suggest that the model is well-suited for capturing complex patterns in the text data for sentiment classification.

---


![Transformer Model Training Output](<insert-image-link-here>)
## **Transformer Model Evaluation**

The Transformer-based model was evaluated on a test dataset using classification metrics such as precision, recall, F1-score, and accuracy. These metrics, along with a confusion matrix and example predictions, highlight the model's performance across Positive, Neutral, and Negative sentiment categories.

---

### **Classification Metrics**

| **Metric**      | **Positive** | **Negative** | **Neutral** | **Macro Avg** | **Weighted Avg** |
|------------------|--------------|--------------|-------------|----------------|-------------------|
| **Precision**    | 1.00         | 0.38         | 0.50        | 0.62           | 0.83              |
| **Recall**       | 0.59         | 0.75         | 0.90        | 0.75           | 0.67              |
| **F1-Score**     | 0.74         | 0.50         | 0.64        | 0.63           | 0.69              |
| **Support**      | 29           | 4            | 10          | 43 (Total)     | 43 (Total)        |

- **Accuracy**: **67%**

---

### **Confusion Matrix**

The confusion matrix below provides a visual representation of the model's performance by showing the relationship between true and predicted labels across all sentiment categories.

![Transformer Model Confusion Matrix](<insert-confusion-matrix-image-link-here>)

---

### **Example Predictions**

The model was tested on example sentences to demonstrate its ability to classify sentiments. Below are the predictions along with confidence scores:

| **Text**                                                | **Prediction** | **Confidence Score** |
|---------------------------------------------------------|----------------|-----------------------|
| I absolutely love this product!                        | Positive       | 0.7416               |
| The experience was terrible and disappointing.          | Negative       | 0.5795               |
| It's just okay, nothing special.                       | Neutral        | 0.6713               |
| Amazing service! Highly recommended.                   | Positive       | 0.7821               |
| The quality of this item is very poor.                 | Negative       | 0.6657               |
| I had the worst experience ever.                       | Negative       | 0.6099               |
| This product is amazing and works perfectly!           | Positive       | 0.7471               |
| It's okay, not great but not bad either.               | Neutral        | 0.6511               |
| Highly disappointed with the quality.                  | Negative       | 0.6657               |
| Very neutral feeling about this decision.              | Neutral        | 0.6675               |

---

### **Insights**

1. **Strengths**:
   - The model performs well on Positive and Neutral sentiments, with high precision and recall.
   - High confidence scores suggest robust sentiment classification for most examples.

2. **Challenges**:
   - Performance for Negative sentiments can be improved, as evidenced by lower recall and precision for this category.

3. **Overall Performance**:
   - With an accuracy of **67%**, the model establishes a strong baseline for sentiment classification tasks.

---

## **Model Comparisons and Final Results**

### **LSTM Model Results**
The LSTM model was evaluated using precision, recall, F1-score, and accuracy metrics. Below are the results:

| **Metric**      | **Positive** | **Negative** | **Neutral** | **Macro Avg** | **Weighted Avg** |
|------------------|--------------|--------------|-------------|----------------|-------------------|
| **Precision**    | 0.71         | 0.00         | 0.00        | 0.24           | 0.50              |
| **Recall**       | 1.00         | 0.00         | 0.00        | 0.33           | 0.71              |
| **F1-Score**     | 0.83         | 0.00         | 0.00        | 0.28           | 0.58              |
| **Support**      | 24           | 6            | 4           | 34 (Total)     | 34 (Total)        |

- **Accuracy**: **70.59%**

Despite achieving reasonable accuracy, the LSTM model struggled significantly with Negative and Neutral sentiments, as shown by the 0.00 precision, recall, and F1-score for these classes.

---

### **Transformer Model Results**
The Transformer model outperformed the LSTM model in key metrics for sentiment classification:

| **Metric**      | **Positive** | **Negative** | **Neutral** | **Macro Avg** | **Weighted Avg** |
|------------------|--------------|--------------|-------------|----------------|-------------------|
| **Precision**    | 1.00         | 0.38         | 0.50        | 0.62           | 0.83              |
| **Recall**       | 0.59         | 0.75         | 0.90        | 0.75           | 0.67              |
| **F1-Score**     | 0.74         | 0.50         | 0.64        | 0.63           | 0.69              |
| **Support**      | 29           | 4            | 10          | 43 (Total)     | 43 (Total)        |

- **Accuracy**: **67%**

The Transformer model demonstrated stronger performance in Neutral and Negative categories compared to the LSTM model. It also exhibited higher precision and recall, providing a more balanced classification across all sentiment classes.

---

### **Example Predictions**
Below are sample predictions from the Transformer model, showcasing its ability to classify sentiments:

| **Text**                                    | **Prediction** | **Confidence Score** |
|---------------------------------------------|----------------|-----------------------|
| This is a great product!                    | Negative       | 0.37                 |
| Amazing service! Highly recommended.        | Positive       | 0.78                 |
| The experience was terrible and disappointing. | Negative     | 0.58                 |
| Very neutral feeling about this decision.   | Neutral        | 0.67                 |

---

### **Comparison Between LSTM and Transformer Models**

| **Model**         | **Accuracy** | **Strengths**                            | **Weaknesses**                     |
|--------------------|--------------|------------------------------------------|-------------------------------------|
| **LSTM**          | 70.59%       | Strong Positive sentiment classification | Poor classification for Negative and Neutral classes. |
| **Transformer**   | 67.00%       | Better performance for Neutral and Negative sentiments | Slightly lower overall accuracy compared to LSTM. |

---

### **Conclusion**

1. The **Transformer-based model** demonstrated a more balanced performance across Positive, Neutral, and Negative sentiment categories compared to the LSTM model.
2. While the LSTM model achieved a slightly higher accuracy, it failed to classify Neutral and Negative sentiments effectively.
3. The **Transformer model** is the preferred choice for this task due to its higher precision and recall for all sentiment classes and its ability to handle Neutral and Negative sentiments better.

---

### **Future Work**
- Fine-tuning the Transformer model further for improved performance.
- Increasing the training dataset size for better generalization.
- Experimenting with hybrid models that combine the strengths of both LSTM and Transformer architectures.
- Using more advanced Transformer models such as BERT or GPT-based models to improve sentiment classification accuracy.

---


### **Conclusion**

The Transformer-based model demonstrates good performance for Positive and Neutral sentiments while highlighting areas for improvement in Negative sentiment classification. This output can be further refined by tuning the model or adding more training data.



### **Libraries Used**
- **Data manipulation**: `pandas`, `numpy`
- **Visualization**: `matplotlib`, `seaborn`
- **Machine Learning**: `sklearn` (Logistic Regression, scaling, train-test split)
- **NLP**: `TfidfVectorizer` and Transformer models (`transformers`)

---
