import statsmodels.api as sm
# from sklearn.externals import joblib
import joblib

# import scaler & scale variables
def preprocess(df):
    print("inside preProcess")
    
    scaler = joblib.load("./models/scaler2.pkl")
    data_scaled = scaler.transform(df)
    
    #add constant
    data_proccessed =  sm.add_constant(data_scaled)

    return data_proccessed



