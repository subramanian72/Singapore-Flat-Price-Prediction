import joblib

def predict_price(data):
    rf_classifier = joblib.load("rfrpriceregg.sav")
    return rf_classifier.predict(data)