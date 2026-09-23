import joblib
import streamlit as st
import pandas as pd 
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_absolute_error, mean_squared_error,root_mean_squared_error,r2_score)
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler

st.write("Sallery prediction system")
df =pd.read_csv("job_salary_prediction_dataset.csv")

st.write(df.head())


#============preprocessing==============
nums_cols =df.select_dtypes(include=['int64',"float64"]).columns

text_cols =df.select_dtypes(include=['object','string']).columns

#============filling missing values============
df[nums_cols] =df[nums_cols].fillna(df[nums_cols].median())
for col in text_cols:
    df[col]=df[col].fillna(df[col].mode()[0])



#==========column divide============

label_cols=['education_level']
onehot_cols = ['job_title', 'industry', 'company_size', 'location', 'remote_work']  # order na wale
#==========label_encoder=========
label_encoder ={}

for col in text_cols:
    le =LabelEncoder()
    df[col]=le.fit_transform(df[col])
    label_encoder[col] =le
#=========one hot encoder=============
ohe =OneHotEncoder(sparse_output=False,handle_unknown='ignore')
encoded_data=ohe.fit_transform(df[onehot_cols])
encoded_df =pd.DataFrame(encoded_data, columns=ohe.get_feature_names_out(onehot_cols))
df= df.drop(columns=onehot_cols).reset_index(drop=True)
df=pd.concat([df,encoded_df],axis=1)
#=========x & y ===========
x = df.drop(columns=['salary'])
y = df['salary']


x_train,x_test,y_train,y_test =train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

scaler =StandardScaler()
x_trian_scaled =scaler.fit_transform(x_train)
x_test_scaled =scaler.transform(x_test)

#========model========

model =LinearRegression()

model.fit(x_trian_scaled,y_train)
y_pred =model.predict(x_test_scaled)

st.header("linear regression model performance")

mae =mean_absolute_error(y_test,y_pred)
mse =mean_squared_error(y_test,y_pred)
rmse =root_mean_squared_error(y_test,y_pred)
r2 =r2_score(y_test,y_pred)

st.write("MAE",mae)
st.write("MSE",mse)
st.write("RMSE",rmse)
st.write('R2_score',r2)

#===========Random forest regression========
model2 =RandomForestRegressor(
    n_estimators=150,
    max_depth=7,
    min_samples_split=2,
    min_samples_leaf=2,
    random_state=42
)

model2.fit(x_train,y_train)
y_pred=model2.predict(x_test)

st.header('random forest regression')

mae =mean_absolute_error(y_test,y_pred)
mse =mean_squared_error(y_test,y_pred)
rmse =root_mean_squared_error(y_test,y_pred)
r2_2 =r2_score(y_test,y_pred)

st.write("MAE",mae)
st.write("MSE",mse)
st.write("RMSE",rmse)
st.write('R2_score',r2)

#======= Best model use==========
if r2_2 >r2:
    best_model =model2
    st.info("Random forest in use")
else:
    best_model =model
    st.info('linear regression model')



st.write("Actual vs predicted values")
fig =plt.figure(figsize=(8,6))
plt.scatter(
    y_test,
    y_pred,
    marker="o",
    linestyle="--"
)
plt.xlabel("Actual values")
plt.ylabel("Predicted value")
plt.grid()
st.pyplot(fig)


st.header("Enter employee information")


job_title =st.selectbox(
    "select job title",
    label_encoder['job_title'].classes_
)

experience_year = st.number_input(
    "Enter Experience Years",
    min_value=2,
    value=2
)

education_level = st.selectbox(
    "select education level",
    label_encoder['education_level'].classes_
)
skill_count = st.number_input(
    "Enter Skills Count",
    min_value=2,
    value=3
)

certifications = st.number_input(
    "Enter Number of Certifications",
    min_value=0,
    value=1
)


industry =st.selectbox(
    "select your industr",
    label_encoder['industry'].classes_
)

company_size =st.selectbox(
    "select the size",
    label_encoder['company_size'].classes_
)

location =st.selectbox(
    "select the location",
    label_encoder['location'].classes_
)

remote_work = st.selectbox(
    'select jobs', 
    label_encoder['remote_work'].classes_
)

#=========text to number (jo onehot hue) ============
job_title_enc = label_encoder['job_title'].transform([job_title])[0]
industry_enc = label_encoder['industry'].transform([industry])[0]
company_size_enc = label_encoder['company_size'].transform([company_size])[0]
location_enc = label_encoder['location'].transform([location])[0]
remote_work_enc= label_encoder['remote_work'].transform([remote_work])[0]  

#=========onehot wale columns banate hain==========
onehot_input = pd.DataFrame({
    'job_title': [job_title_enc],
    'industry': [industry_enc],
    'company_size': [company_size_enc],
    'location': [location_enc],
    'remote_work': [remote_work_enc]
})
onehot_encoded = ohe.transform(onehot_input)
onehot_df = pd.DataFrame(onehot_encoded, columns=ohe.get_feature_names_out(onehot_cols))

#=========education_level==========
education_level_enc = label_encoder['education_level'].transform([education_level])[0]

#=========final new_data==========
new_data = pd.DataFrame({
    'experience_years': [experience_year],
    'education_level': [education_level_enc],
    'skills_count': [skill_count],
    'certifications': [certifications]     # ye line add karo
})
new_data = pd.concat([new_data, onehot_df], axis=1)
new_data = new_data[x.columns]   # order match zaroori hai

#=======prediction=========

if st.button("predict salery"):
    new_data_scaled =scaler.transform(
        new_data
    )

    prediction=best_model.predict(
        new_data_scaled
    )

    st.success(
        f"Predicted salery:{prediction[0]:.2f}"
    )


joblib.dump(model,"linear_regression_model.pkl")
joblib.dump(scaler,"linear scaler.pkl")

#python -m streamlit run salarry_prediction.py