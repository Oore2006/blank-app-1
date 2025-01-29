import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
 

st.title("🎈 My new app")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)

st.write("home")

# 
# Generate a sample dataframe
np.random.seed(42)
data = pd.DataFrame({
    'Category': ['A', 'B', 'C', 'D'],
    'Values': np.random.randint(10, 100, 4)
})

# Display the dataframe
st.subheader('Sample Data')
st.write(data)

# this is me trying 

#  This is the code for heart disease prediction 

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib 

class CardiovascularDiseasePredictor:
#    uploaded_file = st.file.uploader"Choose a csv file", type="csv"

#    if uploaded_file is not NONE: 
#       df = pd.read.csv(uploaded_file)
#       st.write("Data preview:")
#       st.write(df.head()) 


      
    def main():
      st.title("Cardiovascular Disease Prediction")

    # File uploader widget
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
        # Read the uploaded CSV file into a DataFrame
        df = pd.read_csv(uploaded_file)
        st.write("Data Preview:")
        st.write(df.head())

      

        def load_data(self, file_path):
            df = pd.read_csv(file_path)
            return df

        def preprocess_data(self, df):
        # Implement your preprocessing steps here
            X = df.drop('target', axis=1)  # Replace 'target' with your actual target column name
            y = df['target']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test

        def train_base_models(self, X_train, y_train):
            rf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=4, min_samples_split=10, random_state=42)
            gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
            lr = LogisticRegression(random_state=42)

            rf.fit(X_train, y_train)
            gb.fit(X_train, y_train)
            lr.fit(X_train, y_train)

            return rf, gb, lr

        def train_ensemble(self, X_train, y_train, rf, gb, lr):
            ensemble = VotingClassifier(estimators=[('rf', rf), ('gb', gb), ('lr', lr)], voting='hard')
            ensemble.fit(X_train, y_train)
            return ensemble

        def save_models(self, rf, gb, lr, ensemble):
            joblib.dump(rf, 'rf_model.pkl')
            joblib.dump(gb, 'gb_model.pkl')
            joblib.dump(lr, 'lr_model.pkl')
            joblib.dump(ensemble, 'ensemble_model.pkl')

        def main():
            st.title("Cardiovascular Disease Prediction")

        predictor = CardiovascularDiseasePredictor()

        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        st.write("Data Preview:")
        st.write(df.head())

        if st.button("Train Models"):
            X_train, X_test, y_train, y_test = predictor.preprocess_data(df)
            rf, gb, lr = predictor.train_base_models(X_train, y_train)
            ensemble = predictor.train_ensemble(X_train, y_train, rf, gb, lr)

            y_pred = ensemble.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"Ensemble Model Accuracy: {accuracy:.2f}")

            predictor.save_models(rf, gb, lr, ensemble)
            st.success("Models trained and saved successfully!")

        if __name__ == "__main__":
            main()


