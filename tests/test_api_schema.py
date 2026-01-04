from api.app import PredictRequest


def test_predict_request_schema():
    req = PredictRequest(
        age=50, sex=1, cp=3, trestbps=130, chol=250, fbs=0, restecg=0,
        thalach=150, exang=0, oldpeak=1.0, slope=2, ca=0, thal=3
    )
    assert req.age == 50
