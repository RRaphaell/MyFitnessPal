# MyFitnessPal

food_entris არის ლისტი dict ების და თითო დიქშენერი არის თითო საჭმელი იმ დღეს 

თითო საჭმლის დიქშენერი შედგება (mel dishes sequence)

meal - კარგად ვერ გავიგე რაარი

dishes არის ყველა ჩამონათალი რაც ჭამა მაგალითად ყველი და პური და თითოეული კომპონენტისგან რამდენი კალორია და სხვა კომპონენტი მიიღო

sequence არის მიმდევრობა ანუ თუ არის 1 ესეიგი ეს საჭმელი იმ დღეს პირველი ჭამა


parse_data.ipynb- In this file we will parse json file and convert it to pandas dataframe
Depending on which algorithm we want to do, It is possible not to use all variables.
For example, in our case, we do not think it is necessary to know
the quantity of nutrients the customer got after each meal.

data_preprocessing.ipynb- In this notebook you can see data cleaning, visualization, hypothesis testing,
adding new features and preparing data so that we can run a model.

original_data_discuss.ipynb - This file is designed to look into the original data, which is in json format.
we want to transfer information to pandas dataframe
so we need to understand in which form is given in json in order to convert it correctly

predict_reach_goal.ipynb - implement reach goal model

cluster_Customers.ipynb - implement customer segmentation model


