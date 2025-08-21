# heva_explain.py
import shap

def explain_model(model, X_sample, feature_names):
    explainer = shap.TreeExplainer(model.named_steps["clf"])
    shap_values = explainer.shap_values(model.named_steps["scaler"].transform(X_sample))
    return shap_values
