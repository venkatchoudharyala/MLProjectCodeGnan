import streamlit as st
import pandas as pd
import pickle
import time
from sklearn.preprocessing import LabelEncoder
import plotly.express as px

st.set_page_config(page_title="WorkPlace Health", 
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded")

st.sidebar.title("Model Where Abouts..")

# Load the trained models
DTClassifier = pickle.load(open('PickleFiles/DecisionTree.pkl', 'rb'))
RFClassifier = pickle.load(open('PickleFiles/RandomForest.pkl', 'rb'))
SVM = pickle.load(open('PickleFiles/SVM.pkl', 'rb'))
KNN = pickle.load(open('PickleFiles/KNN.pkl', 'rb'))
LRClassifier = pickle.load(open('PickleFiles/LogisticRegression.pkl', 'rb'))
MLPC = pickle.load(open('PickleFiles/MLPC.pkl', 'rb'))

# Load the data
@st.cache(suppress_st_warning=True)
def LoadData():
    return pd.read_csv("Train.csv", delimiter=";")

data = LoadData()

# Define the mapping dictionaries for select boxes
EduFielOpt = {
    "Technical Degree": 0,
    "Medical": 1,
    "Human Resources": 2,
    "Marketing": 3,
    "Other": 4
}

DptOpt = {
    "Manufacturing Director": 0,
    "Sales Representative": 1,
    "Healthcare Representative": 2,
    "Research Director": 3,
    "Manager": 4,
    "Human Resources": 5,
    "Sales Executive": 6,
    "Laboratory Technician": 7,
    "Research Scientist": 8
}

GenOpt = {
    "Male": 0,
    "Female": 1
}

FlexTime = {
    "No": 0,
    "Yes": 1
}

ContrInd = {
    "Yes": 0,
    "No": 1
}

Married = {
    "Divorced": 0,
    "Married": 1,
    "Single": 2
}

JobRoleOpt = {
    "Manager": 0,
    "Engineer": 1,
    "Analyst": 2,
    "Sales": 3,
    "HR": 4
}

RemWork = {
    "Medium": 0,
    "Very High": 1,
    "High": 2,
    "Low": 3,
    "Very Low": 4
}

WorkLoad = {
    "Low": 0,
    "High": 1,
    "Medium": 2
}

# Streamlit App
def main():
    # Set the title and description of the app
    st.title("Workplace Stress Predictor")
    
    Algo = st.sidebar.selectbox("Select Algo", ("RandomForest", "DecisionTree", "SVM", "MLPC", "LogisticRegression", "KNN"))
    
    Frame1, Frame2 = st.tabs(["Predict Yours", "About Our Data"])
    
    with Frame1:
        with st.form(key="prediction_form"):
            st.write("Enter Your Details")

            # Create input fields for user to enter features
            Age = int(st.number_input("Age", value = 22))
            Education = st.slider("Education Level", 1,5,3)
            AvgDailyHours = int(st.number_input("Average Daily Hours", value = 8))
            Department = st.selectbox("Department", list(DptOpt.keys()))
            EducationField = st.selectbox("Education Field", list(EduFielOpt.keys()))
            Gender = st.selectbox("Gender", list(GenOpt.keys()))
            HasFlexibleTimings = st.selectbox("Has Flexible Timings", list(FlexTime.keys()))
            IsIndividualContributor = st.selectbox("Is Individual Contributor", list(ContrInd.keys()))
            JobInvolvement = st.slider("Job Involvement", 1, 4, 2)
            JobRole = st.selectbox("Job Role", list(JobRoleOpt.keys()))
            JobSatisfaction = st.slider("Job Satisfaction", 1, 4, 3)
            LeavesTaken = int(st.number_input("Leaves Taken", value = 22))
            MaritalStatus = st.selectbox("Marital Status", list(Married.keys()))
            MicromanagedAtWork = st.slider("Micromanaged at Work", 1, 5, 3)
            MonthlyIncome = int(st.number_input("Monthly Income", value = 25000))
            NumCompaniesWorked = int(st.number_input("Number of Companies Worked",value = 2))
            PercentSalaryHike = int(st.number_input("Percent Salary Hike", value = 45))
            PerformanceRating = st.slider("Performance Rating", 1, 4, 3)
            RelationshipSatisfaction = st.slider("Relationship Satisfaction", 1, 4, 2)
            RemoteWorkSatisfaction = st.selectbox("Remote Work Satisfaction", list(RemWork.keys()))
            SelfMotivationLevel = st.slider("Self Motivation Level", 1, 4, 3)
            TotalWorkingYears = int(st.number_input("Total Working Years", value = 2))
            TrainingTimesLastYear = int(st.number_input("Training Times Last Year", value = 3))
            WorkLifeBalance = st.slider("Work-Life Balance", 1, 4, 3)
            WorkLoadLevel = st.selectbox("Work Load Level", list(WorkLoad.keys()))
            YearsAtCompany = int(st.number_input("Years at Company",value = 2))
            YearsSinceLastPromotion = int(st.number_input("Years Since Last Promotion", value = 2))
            YearsWithCurrManager = int(st.number_input("Years with Current Manager", value = 1))
            if st.form_submit_button("Predict"):
                UserData = pd.DataFrame({
                    'Age': [Age],
                    'AvgDailyHours': [AvgDailyHours],
                    'Department': [DptOpt[Department]],
                    'Education': [Education],
                    'EducationField': [EduFielOpt[EducationField]],
                    'Gender': [GenOpt[Gender]],
                    'HasFlexibleTimings': [FlexTime[HasFlexibleTimings]],
                    'IsIndividualContributor': [ContrInd[IsIndividualContributor]],
                    'JobInvolvement': [JobInvolvement],
                    'JobRole': [JobRoleOpt[JobRole]],
                    'JobSatisfaction': [JobSatisfaction],
                    'LeavesTaken': [LeavesTaken],
                    'MaritalStatus': [Married[MaritalStatus]],
                    'MicromanagedAtWork': [MicromanagedAtWork],
                    'MonthlyIncome': [MonthlyIncome],
                    'NumCompaniesWorked': [NumCompaniesWorked],
                    'PercentSalaryHike': [PercentSalaryHike],
                    'PerformanceRating': [PerformanceRating],
                    'RelationshipSatisfaction': [RelationshipSatisfaction],
                    'RemoteWorkSatistfaction': [RemWork[RemoteWorkSatisfaction]],
                    'SelfMotivationLevel': [SelfMotivationLevel],
                    'TotalWorkingYears': [TotalWorkingYears],
                    'TrainingTimesLastYear': [TrainingTimesLastYear],
                    'WorkLifeBalance': [WorkLifeBalance],
                    'WorkLoadLevel': [WorkLoad[WorkLoadLevel]],
                    'YearsAtCompany': [YearsAtCompany],
                    'YearsSinceLastPromotion': [YearsSinceLastPromotion],
                    'YearsWithCurrManager': [YearsWithCurrManager]
                })

                le = LabelEncoder()
                PreProUserData = pd.DataFrame()
                for column in UserData.columns:
                    if UserData[column].dtype == object:
                        TempVal = le.fit_transform(UserData[column].astype('category'))
                        PreProUserData[column] = TempVal
                    else:
                        PreProUserData[column] = UserData[column]
                        
                #st.write("Your Details after Pre-processing")

                #st.dataframe(PreProUserData)

                LRPredicted = RFClassifier.predict(PreProUserData)

                # Display the predictions
                with st.spinner("We're almost there.."):
                    time.sleep(3)
                    st.subheader("Predictions")
                    if LRPredicted == 1:
                        st.warning("You are a bit STRESSED! Take some assistance..")
                    else:
                        st.balloons()
                        st.write("You are perfectly alright. Keep Rocking!!") 

    with Frame2:
        st.subheader("Our Data")
        st.dataframe(data)
        st.markdown("---")
        
        ModifiedData = data.copy()
        ModifiedData.dropna(inplace = True)
        Visu = px.scatter(ModifiedData, x='MonthlyIncome', y='JobSatisfaction', size='Age', color='PercentSalaryHike',
                 hover_name='EmployeeID', log_x=True, title='Job Satisfaction vs Target Variable')
        st.plotly_chart(Visu)

if __name__ == "__main__":
    main()
