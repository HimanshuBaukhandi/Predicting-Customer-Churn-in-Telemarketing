# Predicting-Customer-Churn-in-Telemarketing

## Abstract 
The banking industry is one of the financial sectors which heavily depend upon marketing. There are several types of marketing methodologies, be it direct marketing or indirect marketing. One mode of direct marketing which is in adverse use since the 1970’s is telemarketing. To evaluate the success of a telemarketing campaign, this project will create and contrast three prediction models: logistic regression, decision tree classifiers, and neural networks. The models are trained to predict the likelihood of a successful telemarketing contact based on customer demographics and interaction history using a publicly available dataset from a Portuguese bank. In this study, evaluation criteria like accuracy, precision, recall, F1-score, and ROC AUC score were used.
The research showed that the logistic regression model had a precision of 92.0%, recall of 76.8%, F1-score of 83.8%, and ROC AUC score of 88.0%, resulting in an accuracy of almost 96.4%. The Decision Tree Classifier achieved ROC AUC score of 71.5%, accuracy of 88.5%, precision of 50.7%, recall of 49.3%, and F1-score of 50.0%. In contrast, the accuracy, precision, recall, F1-score, and ROC AUC scores for the Neural Network model were 90.7%, 61.8%, 53.3%, and 91.6%, respectively. 
The Neural Network model outperformed the other two models in most evaluation measures, showing the best predictive performance, according to the results. While the Decision Tree Classifier displayed significantly weaker prediction abilities, the Logistic regression model demonstrated a well-balanced trade-off between precision and recall.
The conclusions drawn from this study can help marketing campaign managers make data-supported decisions, thereby enhancing the efficacy of telemarketing campaigns and boosting the number of successful bank long-term deposit subscriptions. However, when using them in real-world applications, additional adjustments and performance factors should be considered.
## 1.	Introduction
Several businesses rely upon telemarketing calls to increase their customer base. Telemarketing calls are a great way of advertising a new product or service to thousands of potential new customers, existing customers, and old customers. Banks and financial institutions are the organizations most benefitted from telemarketing. In this project, I have developed three machine learning models to predict which customer segment is most likely to buy a long-term deposit service offered by a Portuguese bank. The dataset used in this project is a public dataset obtained from the UC Irvine Machine Learning Repository. These models will help us predict the effectiveness of telemarketing approach on a particular demography.
## 1.1	Dataset
The dataset obtained for this project is a public dataset available on the UC Irvine Machine Learning Repository. The dataset has over 45,000 customers and 17 features. With a dataset of this magnitude, I am confident that the ML model which I will develop will be accurate and of significant use to various organizations, like banks, financial services, insurance providers, etc. The features of the dataset are:
1. age: This column represents the age of everyone in the dataset. It indicates the age demographic of the people included. 
2. job: This column captures the occupation or job title of everyone. It provides information about the type of work they are engaged in. 
3. marital: This column indicates the marital status of everyone. It may include categories such as "married," "single," "divorced," or others. 
4. education: This column represents the educational background or level of education attained by everyone. It may include categories like "primary," "secondary," "tertiary," or specific degrees. 
5. default: This column indicates whether the individual has defaulted on any previous financial obligations. It may have values such as "yes" or "no."
6. balance: This column represents the account balance of everyone. It provides information about their financial status or wealth. 
7. housing: This column indicates whether an individual has a housing loan or not. It may have values such as "yes" or "no." 
8. loan: This column indicates whether an individual has a personal loan or not. It may have values such as "yes" or "no." 
9. contact: This column specifies the communication method used to contact the individual. It may include categories like "telephone," "cellular," or others. 
10. day: This column represents the day of the month when contact was made with the individual. It provides temporal information.
11. month: This column represents the month of the year when the contact was made with the individual. It provides temporal information. 
12. duration: This column represents the duration of the last contact with the individual, typically in seconds. It provides information about the length of the interaction. 
13. campaign: This column represents the number of contacts made to the individual during the current marketing campaign. It indicates how many times the person has been contacted. 
14. pdays: This column represents the number of days that have passed since the last contact with the individual from a previous marketing campaign. It provides temporal information. 
15. previous: This column represents the number of contacts made to the individual before the current marketing campaign. It indicates how many times the person has been contacted in the past. 
16. poutcome: This column indicates the outcome of the previous marketing campaign for the individual. It may include categories like "success," "failure," or "unknown." 
17. y: This column represents the target variable, or the outcome being predicted. It may indicate whether the individual subscribed to a particular product or service, often coded as "yes" or "no."
## 1.2	Project Objective
The objective of this project is to develop a data mining (DM) approach that predicts the success of telemarketing calls for selling bank long-term deposits. The study focuses on data collected from a Portuguese retail bank between 2008 and 2013, which includes the impact of the recent financial crisis. The project involves the analysis of a large set of 150 features related to bank clients, product attributes, and socio-economic factors. The project aims to achieve the following goals: 
1. Feature Selection: Utilize a semi-automatic feature selection process to identify the most relevant features for modeling.  
2. Model Comparison: Compare three DM models - logistic regression, decision trees (DTs), and neural network (NN) - to determine the most effective model for predicting the success of telemarketing calls. 
3. Model Evaluation: Test the selected models on an evaluation set consisting of the most recent data (after July 2012) using a rolling window scheme. 
By achieving these objectives, the project aims to provide telemarketing campaign managers with a credible and valuable predictive model that can guide their decision- making process. The ultimate goal is to improve the effectiveness of telemarketing efforts and increase the number of successful subscriptions to bank long-term deposits.
## 1.3	Scope and Limitation
The scope of this project is to create and evaluate prediction models to evaluate the effectiveness of telemarketing calls in the banking industry. The research attempts to develop precise churn prediction models using three different methods: Logistic regression, Decision Tree Classifier, and Neural Network. The study will be supported by a sizable dataset of historical customer information, which includes a range of variables pertaining to banking behaviors, demographics, and customer history. Data preprocessing, exploratory data analysis (EDA), model selection, training, and evaluation will all be part of the project.
Based on the input information provided during the prediction phase, the success of telemarketing calls will be forecasted. The models will offer insightful information regarding the possibility of a successful telemarketing call, which will assist the bank in making data backed business decisions.
Limitations:
Despite the project's all-encompassing approach, some limitations must be acknowledged:

•	Generalization: Although the models were developed using historical data, their performance may differ when applied to fresh, untested data. The distribution of the data and any prospective changes in consumer behavior will determine how well the models generalize to upcoming telemarketing efforts.

•	Model Complexity: Depending on the size and complexity of the dataset, some models may need a lot of processing time and resources. This might be a challenge for real-time prediction algorithms using massive amounts of data.

•	Interpretability: Some predictive models, like neural networks, may not be very interpretable. In certain situations, it could be difficult to comprehend the precise aspects that go into a forecast.

•	Class Imbalance: The dataset might experience class imbalance, in which there are a disproportionately large number of successful telemarketing calls compared to unsuccessful ones. This mismatch may have an impact on model training and produce skewed predictions.

•	External Factors: External factors that are not included in the dataset may have an impact on the effectiveness of telemarketing calls. The predictions could be impacted by monetary situations, market movements, or shifts in consumer preferences.

•	Ethical Considerations: The project should take potential ethical issues into account, such as privacy issues, data usage, and making sure that sensitive client information is handled properly.
## 2.	Methodology
## 2.1	Data Collection and Understanding
The data for this project was obtained from a public repository. The dataset is obtained from the UC Irvine Machine Learning Repository and is the data for a Portuguese bank. The original dataset has 17 features and 45,211 rows of data. The features and their interpretations are explained in 1.1 Dataset subtopic of this report. 
On doing preliminary Exploratory Data Analysis (EDA), I have found the data type of each feature, the null values, duplicate values, and other statistics. The preliminary EDA is mentioned below:
•	Number of null values:  0

•	Number of duplicate values: 0

•	Data Type of features: 
age           int64
job          object
marital      object
education    object
default      object
balance       int64
housing      object
loan         object
contact      object
day           int64
month        object
duration      int64
campaign      int64
pdays         int64
previous      int64
poutcome     object
y            object
•	Minimum age of customers in the dataset: 18 years

•	Maximum age of customers in the dataset: 95 years

•	Average age of customers in the dataset: 40.93 years

•	Average bank balance of customers in the dataset: $1362.272

•	Average duration of calls of the customers in the dataset: 258.163 seconds

Below are some other statistics on the dataset which are displayed in form of graphs. The librarie used for these graphs is SNS Seaborne and Matplotlib. 
•	Distribution of Demographics by education
 ![image](https://github.com/user-attachments/assets/44bb07a4-d730-4937-b151-ec83da3471a2)

•	Distribution of demographics by loan default
 ![image](https://github.com/user-attachments/assets/16cc232f-73de-4cef-a393-19db46bbe709)

•	Distribution of demographics by housing loan
 ![image](https://github.com/user-attachments/assets/7631dff9-6dd3-4cd7-8305-09126a2a3820)

•	Distribution of demographics by occupation  
![image](https://github.com/user-attachments/assets/2987856c-409d-44ae-9074-92ff9b78a173)

•	Duration of Unsuccessful Calls
 ![image](https://github.com/user-attachments/assets/1adaa257-bca1-4a3e-8747-891abc37d0e7)

•	Duration of successful calls
 ![image](https://github.com/user-attachments/assets/d7334e4e-b9d5-4da3-bb1e-4cbdf947edde)

After performing the preliminary EDA, there is no feature which solely determines the success of a telemarketing call. Hence, this project uses all the features available in the dataset. The next step in the course of the project is feature engineering, wherein the features are tailored for the ML models which are made.
## 2.2	Feature Engineering 
The dataset's categorical variables are essential components and are crucial for predicting client attrition. However, to train and make predictions, machine learning algorithms need numerical data. The categorical variables in the dataset were encoded as part of the feature engineering process to close this gap.
For categorical variables with numerous clearly defined categories, such as "job," "marital," "education," "default," "housing," "loan," "contact," and "outcome," One-Hot Encoding was used. By converting each category into a binary column, this technique made sure that a category's presence or absence was appropriately represented by a 1 or a 0, respectively. Without establishing any ordinal relationship between categories, the resulting one-hot encoded features successfully represented the categorical data. The procedure preserved the data's integrity, enabling machine learning models to effectively comprehend and utilize this data.
An Ordinal Encoding technique was chosen for categorical variables with an intrinsic order, such as "education" or "outcome." Based on the categories' predetermined order, this approach assigned integer values to each one. The ordinal link between categories was preserved throughout encoding, enabling the model to comprehend the data's underlying hierarchy. Variables with inappropriate ordinal relationships were carefully considered, and one-hot encoding was selected to prevent any misunderstandings during model training. The customer churn prediction model performed better overall as a result of the encoding process' improved compatibility of categorical variables with the selected machine learning techniques.
## 2.3	Encoded Dataset Features
After feature engineering was applied to the original dataset, a changed dataset with 42 features was produced. These elements stand in for various consumer traits and interactions with the bank. The dataset contains important demographic data, including "age," "job," and "marital_status," the latter of which is encoded as "marital_divorced," "marital_married," and "marital_single" to represent the categories of marital status. Financial characteristics of the consumers, such as "balance" and "default_status," have also been encoded as "default_no" and "default_yes." 
The dataset also contains details on how clients interacted with the bank's marketing effort, such as the variables "contact_type" encoded as "contact_cellular," "contact_telephone," and "contact_unknown." Additionally, to account for the various months of the year, the'month' of the most recent interaction has been encoded as'month_apr,''month_aug,''month_dec,' and so on. Last but not least, the binary classification task has encoded the target variable 'y,' which indicates whether a consumer subscribed to the term deposit, as 'y_no' and 'y_yes'. These encoded features have undergone careful preparation in order to simplify the following machine learning modeling, enabling us to more accurately anticipate client attrition.
## 3.	Model Creation and Evaluation
In this part, we outline the development of three machine learning models for anticipating client attrition and provide metrics for measuring their effectiveness.

## 3.1	Logistic regression Model
A logistic regression model is employed in issues involving binary classification, in which the objective is to predict one of two potential outcomes, typically denoted by the numbers 0 and 1, or True and False. The input values are often mapped into the range [0, 1] using a "S"-shaped logistic function (sigmoid), which models the chance that an instance belongs to a specific class.
A probability number that indicates the possibility that the input belongs to one of the two groups is the output. Predictions can be transformed into discrete class labels by selecting a threshold (typically 0.5).
The equation for logistic regression typically involves the logistic (sigmoid) function and is represented as: p = 1 / (1 + exp(-z)), where p is the probability of the positive class, exp is the exponential function, and z is the linear combination of input features.
Evaluation metrics like accuracy, precision, recall, F1-score, or the area under the Receiver Operating Characteristic (ROC) curve are frequently used to assess logistic regression models. In this paper, all these metrics are used to evaluate the logistic regression model.

## 3.2	Decision Tree Classifier
Decision tree classifier is a well-liked supervised machine learning approach for classification and regression problems. Its goal is to build a tree-like model that splits the data recursively according to the values of the input features and eventually predicts the target variable. The leaf nodes of the tree carry the class labels or numerical values, but each inside node indicates a judgment based on a particular trait. A series of if-then conditions govern the decision-making process, with each path from the root to a leaf standing in for a different decision rule.  
Decision trees are preferred because of how easily they can be understood and because they can handle both categorical and numerical data. However, they can be prone to overfitting, which can be mitigated using techniques like pruning and using ensemble methods like random forests or gradient boosting.
The evaluation metrics used for decision tree classifiers are the same as of logistic regression model. These are accuracy, precision, recall, F1-score, or the area under the Receiver Operating Characteristic (ROC) curve. The evaluation of the models is further explained in 4.2.
## 3.3	Neural Network
A potent and adaptable machine learning technique used for a variety of tasks, including classification, is the neural network classifier. Like other classifiers, evaluation measures are vital in determining how well a model performs. Accuracy, precision, recall, F1 score, and ROC curve analysis are frequently used metrics for neural networks. Precision and recall offer information on the model's capacity to handle true positive cases and avoid false positives and false negatives, respectively. Accuracy assesses the overall accuracy of predictions. The F1 score strikes a compromise between recall and precision, making it appropriate in situations where class imbalance is a problem.
Additionally, the ROC curve and the area under the ROC curve (AUC-ROC) provide a thorough assessment of the classifier's performance for binary classification at various threshold values. These assessment measures are crucial for enhancing the model's architecture, hyperparameters, and ensuring its efficacy in diverse real-world applications due to the neural network's complexity and ability to handle high-dimensional input.


## 4.	Results and Analysis 
## 4.1	Performance of Logistic regression Model
The results of the Logistic regression model for predicting the success of telemarketing calls are as follows:

•	Accuracy: 0.964 (approx. 96.4%)

•	Precision: 0.920 (approx. 92.0%)

•	Recall: 0.768 (approx. 76.8%)

•	F1-Score: 0.838 (approx. 83.8%)

•	ROC AUC Score: 0.880 (approx. 88.0%)

![image](https://github.com/user-attachments/assets/04b14791-4402-45d7-8244-ded0ec8065ad)

 
## Interpretation:
•	Accuracy: The logistic regression model has a 96.4% accuracy rate, which means that in 96.4% of cases, it is able to forecast how telemarketing calls would turn out. The model appears to perform well in terms of overall correctness given its high accuracy.

•	Precision: The precision is roughly 92.0%, which implies that when a successful telemarketing contact is predicted using the Logistic regression model, it is usually accurate to within a factor of 92.0%. A low false positive rate is a sign of a model with high precision, which makes it suited for decreasing inaccurate success predictions when the call is unsuccessful.

•	Recall: The model can recognize roughly 76.8% of the actual successful telemarketing calls in the dataset, or a recall of around 76.8%. High recall indicates that the model can catch a sizable proportion of successful cases, making it useful for identifying successful calls.

•	F1-Score: The harmonic mean of recall and precision is the F1-Score, which is roughly 83.8%. The model can efficiently balance minimizing false positives and false negatives because to the high F1-Score, which indicates a solid balance between precision and recall.

•	ROC AUC Score: The model's ability to identify between positive and negative examples is shown by the ROC AUC Score, which is around 88.0%. A higher ROC AUC Score denotes that the model can more successfully rate positive instances higher than negative instances, hence enhancing its discriminative power. 

The success of telemarketing calls has been predicted remarkably well overall by the Logistic regression model. High F1-Score, recall, accuracy, and precision are all important criteria for assessing binary classification models. The model's ability to distinguish between successful and failed telemarketing calls is reasonably well, according to the ROC AUC Score.
As a result, it appears that the logistic regression model is a good candidate to forecast the success of telemarketing calls in the banking industry. When selecting the best model for deployment, it is crucial to take the individual business objectives and requirements into account. Further testing on a different validation set or via cross-validation would additionally help guarantee the model's applicability to unknown data and its robustness for practical applications.
## 4.2	Performance of Decision Tree Classifier
The results of the Decision Tree Classifier for predicting the success of telemarketing calls are as follows:

•	Accuracy: 0.885 (approx. 88.5%)

•	Precision: 0.507 (approx. 50.7%)

•	Recall: 0.493 (approx. 49.3%)

•	F1-Score: 0.500

•	ROC AUC Score: 0.715 (approx. 71.5%)

![image](https://github.com/user-attachments/assets/463eabac-f376-44d1-b674-2909836bc563)
 
## Interpretation: 
•	Accuracy: The Decision Tree Classifier has an accuracy of about 88.5%, meaning that in about 88.5% of cases, it is able to predict the outcome of telemarketing calls. Although accuracy is a gauge of general correctness, it may be deceptive when working with unbalanced datasets since it ignores the distribution of classes.

•	Precision: The precision is roughly 50.7%, which means that when the Decision Tree Classifier forecasts the outcome of a successful telemarketing call, it is usually accurate to some extent. Low accuracy indicates a high probability of false positives in the model, making it more likely to forecast success when the call is unsuccessful.

•	Recall: The recall is roughly 49.3%, meaning that only about 49.3% of the dataset's successful telemarketing calls can be recognized by the model. Low recall indicates that the model has trouble catching a sizable portion of positive cases, which results in a sizable number of false negatives.

•	F1-Score: The harmonic mean of recall and precision is 0.500, which is the F1-Score. A low F1-Score indicates that the model does not adequately balance minimizing false positives and false negatives with precision and recall, resulting in an imbalance between the two.

•	ROC AUC Score: The model can distinguish between positive and negative occurrences with a ROC AUC Score of roughly 71.5%. The ROC AUC value suggests a moderate amount of discriminative capability even if it is higher than random chance (0.5).

## Interpretation:
Performance of the Decision Tree Classifier for telemarketing call success prediction appears to be constrained, especially in terms of precision and recall. The model has a high false positive rate and struggles to accurately identify positive cases (successful calls).
The type of data, feature choice, and potential overfitting are a few reasons why this can be the case. A decision tree's ability to generalize to new data might suffer from overfitting, especially when it is deep and extremely complicated.
It is crucial to investigate methods to overcome class imbalance, such as employing alternate sampling procedures (like SMOTE) or modifying class weights during model training, given the relatively low F1-Score and the uneven structure of the dataset.
To choose the best model for the specific telemarketing call prediction task in the banking industry, it is critical to carefully assess the decision tree's performance, take into account its limitations, and compare it to other models (such as the Logistic regression model and the Neural Network model). Additionally, to enhance the performance and overall efficacy of the Decision Tree Classifier for this specific use case, additional analysis, hyperparameter tuning, and feature engineering may be required.
## 4.3	Performance of Neural Network Model
The following are the outcomes of the Neural Network model for telemarketing call success prediction:

•	Precision: 0.907 (about 90.7%)

•	Accuracy: 0.618 (about 61.8%)

•	Recall: 0.533, or almost 53.3%

•	F1-Score: 0.573%, or roughly 57.3%

•	ROC AUC Score: 0.916 (about 91.6%).

![image](https://github.com/user-attachments/assets/80aedfdd-f215-4fea-b600-7e2321ba4c9f)

## Interpretation:
•	Accuracy: The accuracy of the neural network model is roughly 90.7%, meaning that in about 90.7% of cases, it predicts properly the outcome of telemarketing calls. The model appears to perform well in terms of overall correctness given its high accuracy.

•	Precision: The precision is roughly 61.8%, which means that when a successful telemarketing call is predicted by the Neural Network model, it is usually accurate to within 61.8% of the mark. This precision number shows that the model has a reasonable capacity to prevent false positives, which might be useful for organizations to avoid expending resources on unsuccessful calls despite not being particularly high.

•	Recall: With a recall of about 53.3%, the model is able to recognize roughly 53.3% of the dataset's real successful telemarketing calls. This recall number, however not particularly high, indicates that the model has some ability to catch successful occurrences, making it useful for identifying successful calls.

•	F1-Score: The harmonic mean of recall and precision is the F1-Score, which is roughly 57.3%. A statistic that strikes a balance between recall and precision is the F1-Score. In this instance, it suggests a fair compromise between reducing false positives and false negatives.

•	ROC AUC Score: The model's ability to identify between positive and negative examples is shown by the ROC AUC Score, which is approximately 91.6%. A higher ROC AUC Score shows that the model has good discriminative power because it can successfully rank positive instances higher than negative ones.

The performance of the neural network model in forecasting the success of telemarketing calls is normally very good. It achieves a high accuracy, showing that it is successful in classifying the majority of instances properly. Although not particularly high, the model's precision and recall numbers demonstrate that it strikes a decent balance between preventing false positives and identifying positive examples.
The model's capacity to distinguish between successful and failed telemarketing calls is further supported by the ROC AUC Score, suggesting high overall performance.
Even though the model performed admirably, there is always space for development, particularly in recall. The model's performance may be improved further using techniques including hyperparameter tuning, architecture modifications, and data augmentation. It may also be advantageous to experiment with various neural network topologies and strategies for addressing class imbalance.
The model's projections should be compared to the business goals and the bank's telemarketing techniques when determining how the model will be used in real-world circumstances. The model's ability to generalize to new, unexplored data and its suitability for practical applications will be ensured by evaluating it on a different validation set or by utilizing cross-validation.
## 4.4	Discussion
## Model 1: Logistic regression
With an accuracy of almost 96.4%, the Logistic regression model displayed remarkable predictive ability. This high accuracy shows that a sizable proportion of the test dataset's instances were correctly identified by the model. The model's ability to correctly forecast occurrences of positive churn is demonstrated by its precision, which is roughly 92.0%. Furthermore, the recall of roughly 76.8% indicates that the model successfully captured a sizeable amount of real positive churn cases. The F1-Score of roughly 83.8% confirms the model's efficacy in finding a compromise between recall and precision. Furthermore, the model's capability to distinguish between positive and negative occurrences is confirmed by the ROC AUC Score of roughly 88.0%. The Logistic regression model often showed strong predictive ability, making it a feasible option for anticipating client attrition.
## Model 2: Decision Tree Classifier
The Decision Tree Classifier properly classified a sizable part of the test dataset instances, achieving an accuracy of about 88.5%. However, the model's ability to correctly forecast positive churn cases is only somewhat good, as evidenced by its precision of about 50.7%. Similar to this, the recall of roughly 49.3% indicates that the model would have trouble capturing a sizable fraction of real positive churn cases. The F1-Score of roughly 50.0% emphasizes the model's difficulty in striking a balance between recall and precision. Although the Decision Tree Classifier had a ROC AUC Score of roughly 71.5%, it had a moderate capacity to discriminate, and this suggests potential limitations for predicting customer churn. As a result, while the Decision Tree Classifier provides valuable insights, further model refinement or alternative algorithms might be necessary to enhance its predictive performance.
## Model 3: Neural Network
With an accuracy of almost 90.7%, the neural network model displayed excellent predicting ability. This high accuracy shows that the model can classify instances more precisely than competing models. The model's predictions of positive churn situations have greatly improved compared to the Decision Tree Classifier, as seen by the precision of about 61.8%. Additionally, the recall of roughly 53.3% implies that the Neural Network model outperforms the Decision Tree Classifier in terms of its ability to identify real positive churn cases. The Neural Network exhibits a better mix between recall and precision, as evidenced by the F1-Score of roughly 57.3%. Additionally, the Neural Network's exceptional capacity to discern between positive and negative instances is indicated by the ROC AUC Score of roughly 91.6%, which makes it a great choice for forecasting customer churn. 
## 4.5	Conclusion
In conclusion, the models for forecasting customer churn from logistic regression, decision tree classification, and neural networks all offered insightful information. A well-balanced trade-off between precision and recall was accomplished by the logistic regression model, which displayed good accuracy. Even though the Decision Tree Classifier's predictive performance was relatively poor, it provided findings that were easy to understand. The Neural Network model emerged as the most reliable and successful model for forecasting customer turnover in the provided dataset due to its high accuracy, precision, recall, and higher ROC AUC Score.

![table_result](https://github.com/user-attachments/assets/25e2105c-f325-438b-992d-ea6a88c84a55)


## 4.6	Prediction on New Data Point Using Logistic Regression Model and Neural Network Model
From the three ML models developed, it is clearly evident that Logistic Regression and Neural Network models are the best ML models for predicting the success of a telemarketing call. The last part of the project is to input a data point with the features and predict whether a telemarketing call for that datapoint will be successful or not. The first prediction is made on the logistic regression model. The data point represents a potential client with the following characteristics: age 30, administrative professional job, married status, secondary education level, no history of financial defaults, $1000 in the bank, with a housing loan but no personal loans, contacted via cellular communication, and most recent contact on August 10th. This consumer has been contacted twice during the current marketing campaign, with the most recent encounter lasting 200 seconds. The customer was additionally reached three times in earlier efforts, with the most recent contact being 50 days ago and producing a successful outcome.

The trained neural network model generated the following forecast after receiving these features:
 
Based on the model's analysis of the customer's characteristics and previous interactions, it predicts a positive outcome, indicating a higher likelihood of the customer subscribing to the bank's long-term deposit service.
Similarly, a different datapoint was used for the neural network model. The data point represents a potential client with the following characteristics: age 65, administrative professional job, married status, secondary education level, history of financial defaults, $9870 in the bank, with a housing loan but no personal loans, contacted via cellular communication, and most recent contact on May 25th. This consumer has been contacted twice during the current marketing campaign, with the most recent encounter lasting 67 seconds. The customer was additionally reached three times in earlier efforts, with the most recent contact being 80 days ago and producing a successful outcome.

The trained neural network model generated the following forecast after receiving these features:
 
Based on the model's analysis of the customer's characteristics and previous interactions, it predicts a negative outcome, indicating a higher likelihood of the customer subscribing to the bank's long-term deposit service.
## 5.	Conclusion
In conclusion, this study used three different machine learning models—logistic regression, decision tree classifiers, and neural networks—to predict the success of telemarketing calls for selling bank long-term deposits. These models' evaluation provided insightful information on how well they predicted outcomes. With a precision of about 96.4%, the logistic regression model successfully balanced the trade-off between recall and precision. The decision tree classifier has a precision of about 50.7%, however it had trouble correctly detecting positive churn cases. The neural network model, on the other hand, was found to be the most trustworthy, with an accuracy of about 90.7%, exhibiting greater precision and recall in addition to a remarkable ROC AUC score of 91.6%. Therefore, it is advised to use the neural network model to predict client attrition in telemarketing efforts. The findings of this study can help marketing campaign managers make data-supported decisions, which will ultimately improve the efficacy of telemarketing campaigns and boost the number of successful bank long-term deposit subscriptions. Further model improvement and cautious real-world implementations are necessary to maximize their performance and applicability, though, given the potential drawbacks and ethical issues.

