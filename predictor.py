# from sklearn.externals import joblib
import joblib


# import model & run prediction
def predict(df):
    print("inside predictor\n")
    
    model = joblib.load("./models/estimator2.pkl")
    prediction = model.predict(df)[0]
    print(prediction)

    lower_bound = int(prediction*.85)
    upper_bound = int(prediction*1.15)

    return lower_bound,upper_bound 