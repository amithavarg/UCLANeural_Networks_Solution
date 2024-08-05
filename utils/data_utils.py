import pandas as pd

def load_data(file_path):
    print("Loading data from:", file_path)
    return pd.read_csv(file_path)

def preprocess_data(df):
    print("Preprocessing data")
    # Converting the target variable into a categorical variable
    df['Admit_Chance'] = (df['Admit_Chance'] >= 0.8).astype(int)
    # Dropping columns
    df = df.drop(['Serial_No'], axis=1)
    # Create dummy variables for all 'object' type variables except 'Loan_Status'
    df = pd.get_dummies(df, columns=['University_Rating', 'Research'])
    return df

def split_data(df):
    x = df.drop(['Admit_Chance'], axis=1)
    y = df['Admit_Chance']
    return x, y
