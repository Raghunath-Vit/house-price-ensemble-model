#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df=pd.read_csv('House Price India.csv')


# In[3]:


df.head(5)


# In[4]:


pd.isnull(df).head()


# In[5]:


df.isna().sum()


# In[6]:


df.info()


# In[7]:


# returns a dataframe with column names of the dataset

pd.DataFrame(list(df.columns), columns=['Column Name'])


# In[8]:


from matplotlib import pyplot as plt
import seaborn as sns


# In[9]:


# used to return a countplot with null title and figsize to be (15, 8) as default

def countplot(dataframe, x_val, plot_title='', figsize=(15,8)):
    plt.figure(figsize=figsize)
    plt.title(plot_title)
    sns.countplot(data=df, x=x_val)
    plt.show()


# In[10]:


# used to return a barplot

def barplot(dataframe, x_val, y_val):
    sns.barplot(data=dataframe, x=x_val, y=y_val)
    plt.title(x_val.title() + ' vs ' + y_val.title())
    plt.show()


# In[11]:


df.shape


# In[12]:


#Univariate Analysis
# returns the countplot of bedrooms

countplot(dataframe=df, x_val='number of bedrooms', plot_title='Count of Number of Bedrooms')


# In[13]:


# returns the countplot of bathrooms

countplot(dataframe=df, x_val='number of bathrooms', plot_title='Count of Number of Bathrooms')


# In[14]:


# returns the countplot of floors

countplot(dataframe=df, x_val='number of floors', figsize=(8,5), plot_title='Count of Number of Floors')


# In[15]:


# returns the countplot of number water present or not

countplot(dataframe=df, x_val='waterfront present', figsize=(8,6), plot_title='Water present (1) or not (0)')


# In[16]:


# returns the countplot of views

countplot(dataframe=df, x_val='number of views', figsize=(8,5), plot_title='Number of Views')


# In[17]:


plt.figure(figsize=(10,10))
sns.jointplot(x=df.Lattitude.values, y=df.Longitude.values,size=10)
plt.ylabel('Longitude',fontsize=12)
plt.xlabel('Latitude',fontsize=12)
plt.show()
sns.despine


# In[18]:


# returns the countplot of House Condition

countplot(dataframe=df, x_val='condition of the house', figsize=(8,5), plot_title='House Condition')


# In[19]:


# returns the countplot of each grade of house

countplot(dataframe=df, x_val='grade of the house', figsize=(8,5), plot_title='Grade of the House')


# In[20]:


# returns the countplot of Scools nearby

countplot(dataframe=df, x_val='Number of schools nearby', figsize=(8,5), plot_title='Number of Schools Nearby')


# In[21]:


sns.displot(df['Price'])
plt.title('Price Distribution')
plt.show()


# In[22]:


# returns the histplot of distance from airport

sns.histplot(data=df, x='Distance from the airport')
plt.show()


# In[23]:


#Bi - variate Analysis
sns.barplot(data=df, x='Number of schools nearby', y='Price')
plt.title('Number of schools vs Price')
plt.show()


# In[24]:


sns.barplot(data=df, x='number of bathrooms', y='Price')
plt.suptitle('Number of bathrooms vs Price')
plt.tight_layout()
plt.show()
# barplot(dataframe=df, x_val='number of bathrooms', y_val='Price')


# In[25]:


barplot(dataframe=df, x_val='number of bedrooms', y_val='Price')


# In[26]:


barplot(dataframe=df, x_val='grade of the house', y_val='Price')


# In[27]:


# Visualize the distribution of each numerical variable
df.hist(bins=20, figsize=(15, 10))
plt.suptitle('Histograms of Numerical Variables', y=1.02)  # Adjust the y-coordinate to prevent overlapping
plt.tight_layout()  # Automatically adjusts subplot parameters for better layout
plt.show()


# In[28]:


# Visualize correlation matrix using a heatmap
correlation_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
# plt.title('Correlation Matrix')
plt.suptitle('Correlation Matrix', y=1.02) 
plt.tight_layout()
plt.show()


# In[29]:


# Pairplot to visualize relationships between numerical variables
sns.pairplot(df[['number of bedrooms', 'living area', 'lot area', 'number of floors', 'Price']])
plt.suptitle('Pairplot of Numerical Variables', y=1.02)  # Adjust the y-coordinate to prevent overlapping
plt.tight_layout()  # Automatically adjusts subplot parameters for better layout
plt.show()


# In[30]:


# Boxplot to identify patterns and outliers
plt.figure(figsize=(15, 8))
sns.boxplot(x='number of bedrooms', y='Price', data=df)
plt.title('Boxplot of Price by Number of Bedrooms')
plt.show()


# In[31]:


from datetime import datetime
# from math import radians, sin, cos, sqrt, atan2


# In[32]:


# Feature engineering

# Current year
current_year = datetime.now().year

# Age of Property
df['age_of_property'] = current_year - df['Built Year']

# Total Square Footage
df['total_sqft'] = df['living area'] + df['lot area']

# Renovation Age
# Calculate age of renovation, substituting Built Year if Renovation Year is 0
df['age_of_renovation'] = current_year - df.apply(lambda row: row['Renovation Year'] if row['Renovation Year'] != 0 else row['Built Year'], axis=1)

# Quality of Location
df['quality_of_location'] = df['grade of the house'] + df['waterfront present']

# Basement Ratio
df['basement_ratio'] = df['Area of the basement'] / df['Area of the house(excluding basement)']

# Floor Area Ratio
df['floor_area_ratio'] = df['living area'] / df['lot area']

# Display the updated DataFrame
print(df.head())


# In[33]:


correlation_matrix = df.corr()

plt.figure(figsize=(24, 18))  # Adjusted figure size
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")

plt.suptitle('Correlation Matrix', y=0.95)  # Adjusted y position of the title
plt.tight_layout()  # Adjust layout to prevent overlapping
plt.show()


# In[34]:


# # Save the updated DataFrame to a new CSV file
# df.to_csv('updated_dataset.csv', index=False)


# In[35]:


# df.head()


# In[36]:


# pip install scikit-learn


# In[37]:


# Convert 'number of bathrooms' and 'number of floors' to integers
df['number of bathrooms'] = df['number of bathrooms'].astype(int)
df['number of floors'] = df['number of floors'].astype(int)
df['floor_area_ratio']=df['floor_area_ratio'].astype(int)
df['basement_ratio']=df['basement_ratio'].astype(int)
df['age_of_renovation']=df['age_of_renovation'].astype(int)


# In[38]:


df.info()


# In[39]:


# Calculate the correlation matrix
correlation_matrix = df[['number of bedrooms', 'number of bathrooms', 'living area', 'lot area', 
                         'number of floors', 'waterfront present', 'number of views', 
                         'condition of the house', 'grade of the house', 
                         'Area of the house(excluding basement)', 'Area of the basement', 
                         'Built Year', 'Renovation Year', 'Postal Code', 'Lattitude', 'Longitude', 
                         'living_area_renov', 'lot_area_renov', 'Number of schools nearby', 
                         'Distance from the airport', 'age_of_property', 'total_sqft', 
                         'age_of_renovation', 'quality_of_location', 'basement_ratio', 
                         'floor_area_ratio', 'Price']].corr()

# Display the correlation matrix
print(correlation_matrix)

# Filter the correlation values with respect to the target variable (Price)
price_correlation = correlation_matrix['Price'].sort_values(ascending=False)

# Display the correlation values
print(price_correlation)


# In[40]:


# # Get the column names
# column_names = df.columns.tolist()

# # Print the column names
# print(column_names)


# In[41]:


df.describe()


# In[42]:


# #Splitting the dataset
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# X = df.drop(columns=['Price'])  # Features
# y = df['Price']  # Target variable
# # Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# i want to drop id and date column from dataset
# Drop the 'id' and 'Date' columns from the dataset
df = df.drop(columns=['id', 'Date'])

# Print the first few rows to verify the changes
print(df.head())


# In[43]:


df.head()


# In[44]:


# import seaborn as sns
# plt.figure(figsize=(12,10))
# cor=X_train.corr()
# sns.heatmap(cor,annot=True,cmap=plt.cm.CMRmap_r)
# plt.show()
#redunant attribute checking on full dataset column for removing unnecessary columns.
import seaborn as sns
import matplotlib.pyplot as plt

# Compute the correlation matrix for the entire dataset
correlation_matrix = df.corr()

# Set the size of the figure
plt.figure(figsize=(28, 36))

# Create the heatmap with annotations and a colormap
sns.heatmap(correlation_matrix, annot=True, cmap=plt.cm.CMRmap_r)

# Show the plot
plt.show()


# In[46]:


def correlation(dataset, threshold):
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if corr_matrix.iloc[i, j] > threshold or corr_matrix.iloc[i, j] < -threshold:
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr


# In[47]:


corr_features=correlation(df,0.7)
len(set(corr_features))


# In[48]:


corr_features


# In[49]:


corr_features.discard('Price')
corr_features


# In[50]:


# Drop the correlated features from the DataFrame
df = df.drop(corr_features,axis=1)

# Save the modified DataFrame to a new CSV file
df.to_csv("modified_dataset.csv", index=False)


# In[51]:


# Load the modified CSV file into a DataFrame
modified_df = pd.read_csv("modified_dataset.csv")

# Display the head of the modified DataFrame
modified_df.head()


# In[52]:


modified_df = modified_df.drop(columns=['Lattitude', 'Longitude','Postal Code'])

# Save the modified DataFrame to a new CSV file
modified_df.to_csv("modified_dataset.csv", index=False)


# In[53]:


# modified_df = modified_df.drop(columns=['Postal Code'])

# # Save the modified DataFrame to a new CSV file
# modified_df.to_csv("modified_dataset.csv", index=False)
modified_df.head()


# In[54]:


# Separate features (X) and target variable (y)
X = modified_df.drop('Price', axis=1)
y = modified_df['Price']


# In[55]:


X


# In[56]:


y


# In[57]:


modified_df = modified_df.drop(columns=['Built Year','Renovation Year','basement_ratio','Distance from the airport','Number of schools nearby','floor_area_ratio','condition of the house','Area of the basement'])

# Save the modified DataFrame to a new CSV file
modified_df.to_csv("modified_dataset.csv", index=False)


# In[58]:


# modified_df = modified_df.drop(columns=['Renovation Year'])

# # Save the modified DataFrame to a new CSV file
# modified_df.to_csv("modified_dataset.csv", index=False)
modified_df.head()


# In[ ]:





# In[59]:


from sklearn.model_selection import train_test_split

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# In[60]:


from sklearn.preprocessing import StandardScaler

# Initialize the scaler for features
scaler = StandardScaler()

# Fit the scaler to the training data and transform the training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform the testing data using the scaler fitted on the training data
X_test_scaled = scaler.transform(X_test)

# Initialize the scaler for the target variable
target_scaler = StandardScaler()

# Fit and transform the training target variable
y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()

# Transform the testing target variable
y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1)).flatten()


# In[61]:


# #Features Scaling:- i am using Standardization because it's optimal for all case while Normalization preferred when the value is near the mean values 
# from sklearn.preprocessing import StandardScaler
# scaler=StandardScaler()
# # Fit the scaler to the training data and transform the training data
# X_train_scaled = scaler.fit_transform(X_train)

# # Transform the testing data using the scaler fitted on the training data
# X_test_scaled = scaler.transform(X_test)



# In[62]:


X_train_scaled


# In[63]:


X_test_scaled


# In[64]:


from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split



# In[65]:


from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Define individual base models
random_forest_model = RandomForestRegressor()
xgboost_model = XGBRegressor()

# Train individual base models
random_forest_model.fit(X_train_scaled, y_train)
xgboost_model.fit(X_train_scaled, y_train)

# Make predictions using individual base models
y_pred_rf = random_forest_model.predict(X_test_scaled)
y_pred_xgb = xgboost_model.predict(X_test_scaled)

y_pred_rf
y_pred_xgb




# In[66]:


# Evaluate Random Forest model
r2_rf = r2_score(y_test, y_pred_rf)

# Evaluate XGBoost model
r2_xgb = r2_score(y_test, y_pred_xgb)

print("Random Forest Model:")
print("R-squared:", r2_rf)

print("\nXGBoost Model:")
print("R-squared:", r2_xgb)


# In[67]:


# Combine predictions using averaging
ensemble_predictions = (y_pred_rf + y_pred_xgb) / 2

# Evaluate ensemble model
r2_rf_xgb = r2_score(y_test, ensemble_predictions)

print("R-squared:", r2_rf_xgb)


# In[68]:


# !pip install catboost


# In[69]:


from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Define individual base models
# random_forest_model = RandomForestRegressor()
xgboost_model = XGBRegressor()
catboost_model = CatBoostRegressor(verbose=0)

# Train individual base models
random_forest_model.fit(X_train, y_train)
xgboost_model.fit(X_train, y_train)
catboost_model.fit(X_train, y_train)

# Make predictions using individual base models
y_pred_rf = random_forest_model.predict(X_test)
y_pred_xgb = xgboost_model.predict(X_test)
y_pred_cb = catboost_model.predict(X_test)


r_cb=r2_score(y_test,y_pred_cb)
print("catboost:")
print("R-Squared: ",r_cb)

# Combine predictions using averaging
ensemble_predictions = (y_pred_rf + y_pred_xgb + y_pred_cb) / 3

# Evaluate ensemble model
r2_rf_xgb_cb = r2_score(y_test, ensemble_predictions)

print("Ensemble Model:")
print("R-squared:", r2_rf_xgb_cb)


# In[70]:


from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



# Define individual base models
random_forest_model = RandomForestRegressor()
xgboost_model = XGBRegressor()
catboost_model = CatBoostRegressor(verbose=0)

# Train individual base models
random_forest_model.fit(X_train_scaled, y_train)
xgboost_model.fit(X_train_scaled, y_train)
catboost_model.fit(X_train_scaled, y_train)

# Make predictions using individual base models
y_pred_rf = random_forest_model.predict(X_test_scaled)
y_pred_xgb = xgboost_model.predict(X_test_scaled)
y_pred_cb = catboost_model.predict(X_test_scaled)


r2_xg=r2_score(y_test,y_pred_xgb)
print("xgboost Model:")
print("R-squared:", r2_xg)



# Combine predictions using averaging
ensemble_predictions = (y_pred_rf + y_pred_xgb + y_pred_cb) / 3

# Evaluate ensemble model
r2_2 = r2_score(y_test, ensemble_predictions)

print("Ensemble Model:")
print("R-squared:", r2_2)


# In[71]:


# Define a threshold (e.g., mean of the target variable)
threshold = y_train.mean()

# Classify predictions
predicted_classes = (ensemble_predictions > threshold).astype(int)
actual_classes = (y_test > threshold).astype(int)

# Calculate True Positives, False Positives, False Negatives
true_positives = ((predicted_classes == 1) & (actual_classes == 1)).sum()
false_positives = ((predicted_classes == 1) & (actual_classes == 0)).sum()
false_negatives = ((predicted_classes == 0) & (actual_classes == 1)).sum()

# Calculate precision and recall
precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)

print("Precision:", precision)
print("Recall:", recall)


# In[72]:


from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score

# Define individual base models
models = {
    'Random Forest': RandomForestRegressor(),
    'XGBoost': XGBRegressor(),
    'CatBoost': CatBoostRegressor(verbose=0),
    'SVR': SVR(),
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'Catboost+RandomForest_gb': RandomForestRegressor()  # Add Catboost+RandomForest_gb here
}

# Train individual base models and compute R-squared scores
r2_scores = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2_scores[model_name] = r2_score(y_test, y_pred)

# Add CatBoost R-squared score from the provided code
r_cb = r2_score(y_test, y_pred_cb)
r2_scores['CatBoost'] = r_cb

#Adding Previous r2Score of different Combinations
r2_scores['Catboost+RandomForest_gb'] = r2_2
r2_scores['EnsembleModel_randomForest_XGB_CatBoost'] = r2_rf_xgb_cb

# Combine predictions using averaging
ensemble_predictions_avg = (y_pred_rf + y_pred_xgb + y_pred_cb) / 3
ensemble_r2_avg = r2_score(y_test, ensemble_predictions_avg)

# Update the scores dictionary with the ensemble R-squared
r2_scores['Ensemble (Averaged)'] = ensemble_r2_avg

# Select the model with the highest R-squared score
best_model_name = max(r2_scores, key=r2_scores.get)
best_model_r2 = r2_scores[best_model_name]

# Print R-squared scores of all models for comparison
for model_name, r2_score in r2_scores.items():
    print(f"{model_name}: R-squared = {r2_score}")

# Print the best model and its R-squared score
print("\nBest Model:", best_model_name)
print("R-squared:", best_model_r2)

# Get the best model
best_model = models[best_model_name] if best_model_name != 'Ensemble (Averaged)' else None


# In[73]:


import pickle 
# Save the selected model as a pickle file
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)


# In[ ]:




