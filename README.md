# MyFitnessPal

# <span style="color:#FF7B07"><div align="center">**File description**
**original_data_discuss.ipynb** - This file is designed to look into the original data, which is in json format.
we want to transfer information to pandas dataframe
so we need to understand in which form is given in json in order to convert it correctly

**parse_data.ipynb** - In this file we will parse json file and convert it to pandas dataframe
Depending on which algorithm we want to do, It is possible not to use all variables.
For example, in our case, we do not think it is necessary to know
the quantity of nutrients the customer got after each meal.

**data_preprocessing.ipynb** - In this notebook you can see data cleaning, visualization, hypothesis testing,
adding new features and preparing data so that we can run a model.

**predict_reach_goal.ipynb** - implement model which predicts if user reaches goal

**predict_180days_Diet.ipynb** - implement model which predicts if user is on 180 days diet

**cluster_Customers.ipynb** - implement customer segmentation model

**/images** - images for readme file

# <span style="color:#FF7B07"><div align="center">**To run project**
1. Clone this repository
2. Download data from https://drive.google.com/drive/folders/1-ANBuqAEf4WVFoqLzu0tqXY4DZcPKge9?usp=sharing
3. Create folder 'data' next to MyFitnessPal folder and move downloaded files here
4. run parse_data.ipynb
5. run data_preprocessing.ipynb
6. Based on your interest run: 
    - predict_reach_goal.ipynb
    - predict_180days_Diet.ipynb
    - Cluster Customers.ipynb

[<span style="color:#FF7B07">**1. Motivation**](#1)<br>
[<span style="color:#FF7B07">**2. About Dataset**](#2)<br>
[<span style="color:#FF7B07">**3. Ideas**](#3)<br>
[<span style="color:#FF7B07">**4. Data Analysis**](#4)<br> 
[<span style="color:#FF7B07">**5. Create Features**](#5)<br>

# <span style="color:#FF7B07"><div align="center">**Motivation** <a  name="1"></a>
  
Nowadays, a healthy lifestyle is becoming a valuable characteristic of modern society. More and more people try to enhance their health by doing regularly different sports and put emphasis on their food habits. In order to satisfy the specific needs of every individual, conclusions gained out of the users’ data are of high importance.
The healthcare industry is booming, especially when it comes to the analysis of health-related data. 39% of adults worldwide and over 65% of US adults are clinically overweight or diagnosed with obesity. Overweight and obesity increase the risk for health issues including diabetes, hypertension, and osteoarthritis. Half of US adults try to lose weight each year. Most often, they attempt to do so by exercising more and eating less, since negative net calorie intake is associated with weight loss. To aid in this process, many individuals use calorie-tracking apps. MyFitnessPal, for example, is an online calorie counter used to track and work toward weight loss goals. Users can track calories through diet and workout logs, and they can provide weight information over time.

# <span style="color:#FF7B07"><div align="center">**About Dataset** <a  name="2"></a>

Source https://www.kaggle.com/vetrirah/customer?select=Train.csv

MyFitnessPal provides a dataset uploaded on the Kaggle. It contains 587,187 days of food diary
records logged by 9.9K MyFitnessPal users from September 2014 through
April 2015. Each line is a tab-separated list of:
- Anonymized user ID
- Diary date
- List of food entries and nutrients (as JSON objects) <br>
![nutritient](/images/nutritient.PNG) <br>
- The daily aggregate of nutrient intake and goal (as JSON objects). <br> 
![agregate](/images/agregate.PNG) <br>


# <span style="color:#FF7B07"><div align="center">**Ideas** <a  name="3"></a>
  
Our goal is to solve some real-life problems with a machine learning approach to help people easily reach their goals. <br>
After brainstorming we realized that there are 3 of the ideas which worth to be tested with machine learning. The questions to be answered are: <br>
- Based on daily nutrient features, will the person reach a goal or not?
- Is person on 180 days diet based on nutrients she/he takes?
- How are app customers segmented?
We are going to answer above mentioned questions in this article using different algorithms.

# <span style="color:#FF7B07"><div align="center">**Data analysis** <a  name="4"></a>
  
While processing the data, the idea arose to test several hypotheses with the given data. We have all heard the phrase: "I will start on Monday" and most often it refers to the above-mentioned diet and healthy lifestyle. So we wondered if the data from Myfitnesspal would confirm the hypothesis that the diet starts mostly on Mondays and as we can see the hypothesis was really confirmed and we also noticed a very interesting fact: on Wednesdays, the least number of people start which is also due to the reason that in the middle of the week everyone is lazy about the start changing. <br> 
![start_date](/images/start_day.PNG) <br>
It is also quite interesting when people break their healthy diets with a lot of burgers? We think it has more to do with weekends and holidays. The data tells us about the same thing, on Friday people are swimming in the junk food and this is probably due to the fact that many countries have a Friday night fun culture. <br>
![cheat_date](/images/cheat_days.PNG) <br>
According to one well-known hypothesis, a large portion of users do not use the app after 72 hours. We wondered how true all this was, we tested it and found that we do not have such a decrease as is generally considered. We also found that a certain number of users follow a 6-month health program. <br> 
![logs](/images/logs.PNG) <br>
According to the data, we found people who are at risk for diabetes and other diseases due to the food they eat. It is important to take care of their health and warn them to follow the rules of a healthy lifestyle. The group on the left side are the people who are on 180 days diet which we will discuss later. <br> 
![risk](/images/risk_group.PNG) <br>
Let’s take a look at some of the exciting trends in food intake by month. As it turned out, the season is really important for consumers. If we observe, with the approach of summer, the popularity of strawberries is increasing while coffee is always a favorite product. <br> 
![strawebery](/images/strawbery.PNG) <br>
Based on the data, we can see the most frequent combos and It may be really interesting for the markets or analog companies. Here you can see 3 meals combo, It becomes clear from the illustration above that our customers eat Saba’s protein milk with peanut butter most frequently . This is logical because people who want to manage their body weight often use milk and similar product in their diet. Following, the report takes a closer look and continue with group of fours. Similarly, we can also see top 4 of this kind of food groups. However, those groups are slightly different from the previous trios. We can see that people who use Cantaloupe often eat Dark Chocolate with Peanut Butter which is a rather surprising combo and interesting finding <br>
![combo3](/images/combo_3.png) <br>
![combo](/images/combo_4.png) <br>

# <span style="color:#FF7B07"><div align="center">**Creating features** <a  name="5"></a>
  
1. FoodLen - The amount of food received during the day. We used it because it was impossible to use specific food names in the models and we wanted to use this data somewhere
2. Logged_frequency - The number of days logged by the user in a given range <br> 
![log_feature](/images/log_feature.PNG) <br>
3. Start_date - The day of the week on which the user used the application for the first time
4. Days_missed - The number of days thrown in a given range
5. healthyDistributed- How healthy the nutrients in the food taken during the day are distributed
6. Column diff-s - This refers to the difference between the goal and total of nutrients, which shows how much they missed the goal set for each day. <br>
![diff_feature](/images/diff_feature.PNG) <br>
7. Food group - according to nutrients, assume which food group belongs to (fast food, soups, and souces…) <br>
![foodGroup](/images/FoodGroup.PNG) <br>

# <span style="color:#FF7B07"><div align="center">**Predictions** <a  name="6"></a>
  
- Based on daily nutrient features, will the person reach the goal or not? 
The algorithm, as you can see, is supervised learning and is a classification. To do this, we need to label the data or have a single feature that will tell us whether the user has achieved the goal. For this, we take the record of the last 5 days for all the customers and add the deviation between the goal and total nutrient of each day. After that, I believe that those who have less than 30 percent of the misses have achieved the goal. As for the input, for this, I take the first 7 days record for all users and using that I make a prediction. Because the algorithm uses the algorithm for the first 7 and the last 5 days, we should leave users who have at least 12 days of records, but we leave users whose number of records is at least 30 days, one month.

- Is person on a 180 days diet or not?
As for the app, the Myfitnesspal, as well as for the users, it is significant and interesting, to know how many days are they going to use this app.Firstly, because if that number is quite small, they can realize that something is changing in everyday life, and secondly because the app encourages easily surrendered individuals . In the data preprocessing part, we saw that there were too much people who logged days around 180. As google said, there are too much 6 months diets and courses which are really effective and many people are doing this. After guessing that, we thought about doing the prediction to determine if the person is on the 180 days diet .So, It be quite interesting if they are on a 180 days diet because if so, then we know that they’re gonna use it at least 180 days. Something similar can be done for excellent users. We use decision tree classifier , svm and logistic regression as an algorithms and as an output, we created a new feature called is_180days_diet which is boolean feature. To be concerned, 
we need guarantee that the persons for which we tell that are not on the 180 days diet, really are not on this. So if we take people who started log in first 30 days, we have guarantee that we choose correct ones

- How are app customers segmented?
TODO


