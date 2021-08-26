# Predicting UEFA Champions League Winner 


#### The purpose of this project is to accurately predict Champions League matches by using machine learning. 

# Introduction:
The Champions League is a year-round tournament with 32 teams broken up into 8 groups. This leauge takes the best teams in Europe and has them compete against one another to be considered the best club team in the world. The tournament began in 1955 and have evolved to be one of the biggest leauges as well as one of the most prestigious. 
# Research Question:
The Champions League is currently the most competetive league and has taken place since 1955. With all the changes and growth of this league, can machine learning be used to predict outcomes for this competition?
# The Data
The first steps were to obtain data on the Champions League that would be used to make predictions. I mainly used 2 dataset for this project, one consisted of all teams that had been in the final of the leauge. They were either the winner or the runner-up, it also had other information like coach, mvp, country, and season. The other dataset was used in the predictive model and contained the entire history of the champions league from the 1955 season to 2015 season. This dataset had the home team, away team, coutry of each team, the full-time score, half-time score, and aggregtae scores.

# Cleaning
The largest dataset required a lot of data cleaning due to how large it was as well as changes that have occurred since 1955. The competition has grown and the structure of the league has evolved to fit a large amount of teams. There were copious missing values which were replaced with character values like "no info" or "no penalties" in order to prevent errors while running the models. There were some foreign characters that were in the names of the teams since some of them were in different languages, but I  got rid of them and made sure each team was readable. 

# Data Exploratation
My EDA communicated that Real Madrid has won the most titles as well as been in the final the most out of all teams. Spain has been represented the most in this competition as the winner or runner-up, which is where Real Madrid is located. This league has had 53 countries represented all over Europe and some parts of Africa and Asia. Although, most of those countries do not compete in this competition anymore, it is interesting to see how diverse this league has been. The main countries that have proven to be the most successful as the winner or runner-up have only been from 13 countries and are all in Europe. Some other interesting visualizations showed how the most common scores from every game has been 1-0 and second most being 1-1. This shows how competitive these games are and how it is not easy to score on the opposing team. 

<p align="center">
<img width="400" alt="Screen Shot 2021-08-19 at 2 10 26 PM" src="https://user-images.githubusercontent.com/60277706/130360577-58199bb2-1f88-4edc-837f-e185757a6838.png"><img width="400" alt="Screen Shot 2021-08-19 at 2 33 22 PM" src="https://user-images.githubusercontent.com/60277706/130360580-f1c0fc2a-aadd-4a8e-a4f2-c635d0ce766a.png">
<img width="800" height="400" alt="Screen Shot 2021-08-19 at 10 11 48 PM" src="https://user-images.githubusercontent.com/60277706/130360590-51a7f440-6319-496b-8c52-a4724da14367.png">
<img width="800" height="400" alt="Screen Shot 2021-08-19 at 10 11 29 PM" src="https://user-images.githubusercontent.com/60277706/130360582-0fcedb11-1c35-40b4-bd99-ae92682f852f.png">
<img width="500" height="500" alt="Screen Shot 2021-08-19 at 11 10 35 AM" src="https://user-images.githubusercontent.com/60277706/130360596-31b98d86-01e9-41b7-9a75-157ccc5dff8b.png">
<p/>
  
# Model Building
The model building process began with deciding to use 3 classification models, which were a logistic regression, naive bayes classification, and a decision tree. The first step was making a feature set and a target variable for the full-time score to produce score outcomes.  All the numeric columns in the dataset needed to be standardized which included 6 in total. The next step was to convert the remaining object columns with characters into dummy variables that way the model would run and not stop due to errors. I was able to display the new dataset and see that all the input was numeric and scaled appropriately. I used sklearn to split the data into training and testing sets as well as shuffle it as an extra randomization. I made the test size 100 due to how large the dataset was and made the random state equal to 2. Now that the data was split, the process of training it began with training it for a certain amount of time and then stopping and moving on to the next set. The training portion also included code to produce the F1 score as well as the accuracy values for each model. The final step was to set up each model with their parameters needed, and then run it with the training and testing data. 

# Results
From the results, the training set size was 6454 and ran for different times depending on the model. The logistic regression model ran the longest for training the data at 20.18 seconds, while the prediction only took 0.003 seconds. The F1 and accuracy was 94% in total and proved to be a solid model for predicting this dataset. The Naive Bayes classification trained the model much faster at 0.0199 seconds and predicted in 0.0193 seconds. The scores for the F1 and accuracy were much higher at 100% for both, which is another great model to predict. The last model was the decision tree using the XBG classifier that trained for 6.05 seconds and predicted in 0.05 seconds. This model had the highest rate of accuracy and F1 score of 99.8%, which meant it predicted the outcome almost exactly as the actual values. Although these models were sufficient as is, I went ahead and changed the parameters of the decision tree to assess the differences. The first parameter adjustment was slightly lower than the initial model at 99% and 98%. The second adjustment proved to be less successful as the first 2 with an F1 score and accuracy score of 96%, like the SVC model. Once I determined the best model was the initial decision tree XGBoost classifier, I saved the model for future use. 
<p align="center">
<img width="500: height="550" alt="Screen Shot 2021-08-25 at 7 44 34 PM" src="https://user-images.githubusercontent.com/60277706/130886879-ae9d85c3-b692-4703-bbe0-e1d64d30f053.png">


The last step for the models was validating the results in order to assess how accurate the model is predicting. I used cross validation in order to produced a report that shows how accurate the model is in case of overfitting or any causes that might effect the model. I plotted each cross validation to visualize how well the model performed and to also compare each model to one another. The Logistic Regression model produced a 92% accuracy score, the Naive Bayes model produced a 99.6%, and the Decison Tree produced a 99.5% as well. The results were still the same, with the Naive Bayes classifier as the best model and the Logistic Regression as at the bottom. 
### Logistic Regression Classifier
<p align="center">
<img width="550" height="400" alt="Screen Shot 2021-08-25 at 7 34 16 PM" src="https://user-images.githubusercontent.com/60277706/130886808-3d75b94c-debe-4fe2-bbb5-8fd299185cf6.png">

### Naive Bayes Classifier
<p align="center">
<img width="550" height="400" alt="Screen Shot 2021-08-25 at 7 38 49 PM" src="https://user-images.githubusercontent.com/60277706/130886211-dbe71a6b-bd81-46e1-8b8e-e5bb8b11eb22.png">

### Decision Tree Classifier
<p align="center">
<img width="550" height="400" alt="Screen Shot 2021-08-25 at 7 41 19 PM" src="https://user-images.githubusercontent.com/60277706/130886407-7e8f0cb5-3868-460f-b46c-051483a37043.png">

              
# Conclusion
Overall, this project was successful in the ability to predict the outcome of The Champions League regardless of the inconsistencies and changes overtime. The models that I was able to create had very high accuracy rates at predicting the outcomes of games. This information can be very useful for sports betting, while being specific enough to predict the end score. The models could also be utilized by the organizations and teams that participate in this tournament to strategize their formations for each game. Companies like Nike and other sponsorships would benefit from the prediction of the winner by making merchandise to sell before the final has even taken place. These models can be used for various purposes for predicting the outcomes of The Champions League. 


### Links
#### Presentation:
https://youtu.be/AK7l8V2MBu4
#### Tableau:
https://public.tableau.com/app/profile/andrea.fernandez6869/viz/ChampionsLeagueMap/Dashboard1?publish=yes
https://public.tableau.com/app/profile/andrea.fernandez6869/viz/ChampionsLeagueEDA/Dashboard2?publish=yes
