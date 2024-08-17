import streamlit as st 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
# Ignore warnings
warnings.filterwarnings('ignore')
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

#st.set_option('deprecation.showPyplotGlobalUse', False)

st.set_page_config(page_title="Chart Visualizations", page_icon="📊")

st.title('DATA INSIGHTS')
# Read the excel file
data = pd.read_excel("C:\\Users\hp\Desktop\\empperfomance\\INX_Future_Inc_Employee_Performance_CDS_Project2_Data_V1.8 (1).xls", index_col=0)
# data = pd.read_excel('INX_Future_Inc_Employee_Performance_CDS_Project2_Data_V1.8.xls', index_col=0)
# Setting the EmpNumber as index to ensure that no rows of data in the table are identical

st.header('1. Employee Departments Analysis')

# Assuming percent_1 is defined with valid data
percent_1 = data['EmpDepartment'].value_counts().values

wedgeprops = {"linewidth": 0.1, 'width': 1, "edgecolor": "white"}

fig, ax = plt.subplots(figsize=(4, 6))
color = ["red", "pink", "lightblue", "orange", "green", "yellow"]
labels = data['EmpDepartment'].value_counts().index.tolist()

# Ensure the explode list matches the number of unique departments
explode = [0.1 if i == 2 else 0.2 if i == 3 else 0 for i in range(len(labels))]

ax.pie(percent_1, labels=labels, explode=explode, autopct="%0.2f%%",
       startangle=46, shadow=True, pctdistance=0.85, wedgeprops=wedgeprops, textprops={"fontsize": 13, "fontweight": "bold"},
       rotatelabels=False, colors=color)

ax.set_title("Employee Department Analysis", fontsize=18, fontweight='bold')


plt.tight_layout(pad=6)
st.pyplot(fig)
plt.show()


st.markdown('''**From the above chart, we see that:**
* The Sales Department made up majority of the Employee workforce with 31.08% (373) while the Data Science department was the least with 1.67%, had a few employees (20)
''')

st.header('2. Department-wise Performance')

st.subheader('a.) Total count of each department by the Performance rating')
fig_1 = plt.figure(figsize=(30, 20))
sns.countplot(data=data, x='EmpDepartment', hue='PerformanceRating')
st.pyplot(fig_1)

st.markdown(''' **From the above plot, we get the following insights:**

* Every department employee gives an Excellent performance rating more in number
* Sales Department: 87 employees had a performance rating of Good(2), 35 employees-Outstanding(4) an majority of the employees (251) had an Excellent (3) performance rating
* Human Resource Department: 10 employees had a Good performance rating, 38 had and Excellent performance and 6 employees were Outstanding.
* Development Department: 13 employees had a Good performance rating, 44 with Outstanding while the majority of 304 were Excellent
* Data Science Department: Majority of the employees(17) in the department had an Excellent performance while 2 were Outstanding and 1 was rated Good.
* Research & Development department: 68% of the employees had an Excellent performance rating while 12% were Outstanding and 20% Good performance rating
* Finance Department: employees with Excellent performance rating were the majority (30), with 15 having a Good performance rating and 4 being rated Outstanding''')

#st.subheader('b.) Checking the Overall percentage Departmental Average Performance')
#data.groupby('EmpDepartment')['PerformanceRating'].mean().plot(kind='pie',
 #                                                              figsize=(30, 15),
  #                                                             colors=['#8FBC8F','#FFFF00','#87CEEB','pink','red','orange'],
   #                                                            explode=[0, 0.07, 0, 0, 0, 0],
    #                                                           autopct="%1.2f%%")
#plt.title('Departmental Average Performance')
#st.pyplot(explode=explode)




department_counts = data['EmpDepartment'].nunique()

#Create the explode list with zeros, and set one element to 0.07 to match your original code
#explode = [0] * department_counts
#if department_counts > 1:
#    explode[1] = 0.07

st.subheader('b.) Checking the Overall percentage Departmental Performance')
plt.clf() # for claering the previous figure
data.groupby('EmpDepartment')['PerformanceRating'].mean().plot(kind='pie',
                                                               figsize=(10, 8),
                                                               colors=['#8FBC8F','#FFFF00','#87CEEB','pink','red','orange'],
                                                               explode=explode,
                                                               autopct="%1.2f%%")
plt.title('Departmental Average Performance')
st.pyplot(plt)



st.markdown(''' From the above charts, we see the best performing departments:

* Development department; mean of 3.085873 (17.51%)
* Data Science department; mean of 3.050000 (17.31%)
* Human Resource department; mean of 2.925926 (16.61%)
* Research & Development department; mean of 2.921283 (16.58%)
* Sales department; mean of 2.860590 (16.24%)
* Finance department; mean of 2.775510 (15.75%) ''')

st.header('3. Satisfaction Levels with Performance')
# Setting the style for the plots
sns.set(rc={"font.size": 20, "axes.titlesize": 20, "axes.labelsize": 25, "xtick.labelsize": 25, "ytick.labelsize": 25,
            "legend.fontsize": 15})

# Creating a figure for the plots
fig, ax = plt.subplots(2, 3, figsize=(40, 30))
fig.suptitle('Performance rating of the Ordinal Columns data types', fontsize=40, color='red')

# Plotting the Distribution plots of the Numerical columns
sns.countplot(data=data, x='EmpEducationLevel', ax=ax[0, 0], hue=data['PerformanceRating'])
ax[0, 0].set_title('EmpEducationLevel')

sns.countplot(data=data, x='EmpEnvironmentSatisfaction', ax=ax[0, 1], hue=data['PerformanceRating'])
ax[0, 1].set_title('EmpEnvironmentSatisfaction')

sns.countplot(data=data, x='EmpJobInvolvement', ax=ax[0, 2], hue=data['PerformanceRating'])
ax[0, 2].set_title('EmpJobInvolvement')

sns.countplot(data=data, x='EmpJobSatisfaction', ax=ax[1, 0], hue=data['PerformanceRating'])
ax[1, 0].set_title('EmpJobSatisfaction')

sns.countplot(data=data, x='EmpRelationshipSatisfaction', ax=ax[1, 1], hue=data['PerformanceRating'])
ax[1, 1].set_title('EmpRelationshipSatisfaction')

sns.countplot(data=data, x='EmpWorkLifeBalance', ax=ax[1, 2], hue=data['PerformanceRating'])
ax[1, 2].set_title('EmpWorkLifeBalance')
st.pyplot(fig)

st.markdown('''From the above plots, we get the following insights:

* Most of the employees with a Bachelors (rating=3) have an excellent score (3) than the rest
* The employees who had a High (rating=3) environment satisfaction performed better in the company
* Majority of the employees who had a High (rating=3) Job involvement had an excellent score (3)
* Employees who had a Very High (4) Job satisfaction performed excellently and were the most
* Most Employees with a Better (3) Work-life balance had an excellent performance compared to the rest
''')

st.header('4. Gender Analysis by Employee Department')

import matplotlib.pyplot as plt
import seaborn as sns

# Create the figure object
fig, ax = plt.subplots(figsize=(20, 15))

# Countplot for gender vs Employee department
sns.countplot(x=data["EmpDepartment"], hue=data['Gender'], palette="Accent", ax=ax)
ax.set_title("Gender distribution across Employee Department", fontweight="bold", fontsize=20)
ax.set_xlabel("Employee Department")
ax.set_ylabel("Count")

# Customize legend
legend = ax.legend(prop={"size": 10})
legend.set_title("Gender", prop={"size": 15, "weight": "bold"})
plt.setp(legend.get_texts(), color='black')
legend.draw_frame(False)

# Adjust layout
plt.tight_layout(pad=2)

# Pass the figure to st.pyplot()
st.pyplot(fig)


st.markdown('''From the analysis above, we get insights that:

* Male employees make up 60% of the workforce, while female 40%
* The male employees are more in each department with both the male and female employees being more in the sales department compared to the other departments.
''')




st.header('5. Employee Education Background Analysis')

# Set the figure
sns.set(rc={"font.size": 12, "axes.titlesize": 18, "axes.labelsize": 18, "xtick.labelsize": 14, "ytick.labelsize": 14,
            "legend.fontsize": 12, 'axes.grid': False, 'axes.facecolor': 'white'})

# Pie chart for employees education background
percent_1 = data['EducationBackground'].value_counts().values

wedgeprops = {"linewidth": 0.1, 'width': 1, "edgecolor": "w"}
color = ["#61ffff", "#cd853f", "#00ff00", "#ffff66", "#ff6e4a", "royalblue"]

plt.figure(figsize=(20, 8))

plt.subplot(1, 2, 1)

# Countplot chart for employees education background by department analysis
ax = sns.countplot(x=data['EducationBackground'], hue=data["EmpDepartment"], palette="tab10")
plt.title("\nEmployees Education Background analysis with Department\n", fontweight="bold", fontsize=19)
plt.xlabel("\nEmployees Education Background")

# Get unique values and their count for EducationBackground
unique_education = data['EducationBackground'].unique()
unique_education_count = len(unique_education)

# Set xticks and xtick labels dynamically
plt.xticks(ticks=range(unique_education_count), labels=unique_education)

# Display the plot in Streamlit
st.pyplot(plt)






# -------------------- ML Model ---------------------------

import joblib  # For saving and loading the model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# Train and Save the Model
# Features selected from the SelectK Method
X = data[['EmpEnvironmentSatisfaction', 'EmpLastSalaryHikePercent',
          'ExperienceYearsAtThisCompany', 'ExperienceYearsInCurrentRole',
          'YearsSinceLastPromotion']]
y = data['PerformanceRating']

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and training the random forest model
rf = RandomForestClassifier(n_estimators=100, random_state=0)
rf.fit(X_train, y_train)

# Saving the model
joblib.dump(rf, 'random_forest_model.pkl')

# Displaying training data value counts for reference
y_train.value_counts()

# Sidebar for User Input and Prediction
st.sidebar.header('Employee Performance Prediction')

# Load the trained model
model = joblib.load('random_forest_model.pkl')

# Function to get user input from sidebar
def get_user_input():
    EmpEnvironmentSatisfaction = st.sidebar.slider('EmpEnvironmentSatisfaction', min_value=1, max_value=4, value=3)
    EmpLastSalaryHikePercent = st.sidebar.slider('EmpLastSalaryHikePercent', min_value=0, max_value=100, value=15)
    ExperienceYearsAtThisCompany = st.sidebar.slider('ExperienceYearsAtThisCompany', min_value=0, max_value=40, value=10)
    ExperienceYearsInCurrentRole = st.sidebar.slider('ExperienceYearsInCurrentRole', min_value=0, max_value=20, value=5)
    YearsSinceLastPromotion = st.sidebar.slider('YearsSinceLastPromotion', min_value=0, max_value=20, value=2)

    user_data = {
        'EmpEnvironmentSatisfaction': EmpEnvironmentSatisfaction,
        'EmpLastSalaryHikePercent': EmpLastSalaryHikePercent,
        'ExperienceYearsAtThisCompany': ExperienceYearsAtThisCompany,
        'ExperienceYearsInCurrentRole': ExperienceYearsInCurrentRole,
        'YearsSinceLastPromotion': YearsSinceLastPromotion
    }
    features = pd.DataFrame(user_data, index=[0])
    return features

user_input = get_user_input()

st.subheader('User Input:')
st.markdown('The table is based on the values you selected in the sidebar.')
st.write(user_input)

# Prediction
if st.sidebar.button('Predict'):
    prediction = model.predict(user_input)
    st.subheader('Prediction:')
    st.write(f'Performance Rating: {prediction[0]}')