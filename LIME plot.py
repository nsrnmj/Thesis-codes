import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

# Instantiate model and place it on the correct device
input_size = p  # Adjust according to the number of input features
model = Model(input_size)
if torch.cuda.is_available():
    model = model.to('cuda')

# Define feature names based on the input size
# feature_names = [f'feature_{i}' for i in range(input_size)]


# feature_names = df_test.columns.tolist()  #.drop(['time', 'death'], axis=1)


def f(data):
    # Ensure the input is a 2D numpy array
    if isinstance(data, np.ndarray):
        data = torch.tensor(data, dtype=torch.float32)
        if torch.cuda.is_available():
            data = data.to('cuda')

    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        predictions = model(data)  # Make predictions

    # Convert predictions back to numpy for further processing
    if predictions.is_cuda:
        predictions = predictions.cpu()

    predictions = predictions.numpy()  # Convert torch tensor to numpy array
    predictions_df = pd.DataFrame(predictions, columns=["pred_alpha", "pred_beta", "pred_gamma", "pred_lambda"])

    # Define a specific time point for survival probability calculation
    t_specific = 129  # You can choose any specific time point relevant to your analysis

    # Calculate survival probabilities at the specific time point
    survival_probs = weibull_surv(np.array([t_specific]), predictions_df["pred_alpha"].values,
                                  predictions_df["pred_beta"].values, predictions_df["pred_gamma"].values,
                                  predictions_df["pred_lambda"].values).flatten()

    # Apply the classification threshold
    threshold = 0.5
    predictions_binary = np.where(survival_probs <= threshold, 0,1)

    # Convert binary predictions to probabilities
    probabilities = np.zeros((len(predictions_binary), 2))
    probabilities[np.arange(len(predictions_binary)), predictions_binary] = 1

    return probabilities  # Return an array of probabilities

# Assuming test_x is a numpy array containing the test data
instance_idx = 0
background = test_x[:30].cpu().numpy()[instance_idx] # Get the first instance for testing

# Prepare the explainer
explainer = LimeTabularExplainer(test_x.cpu().numpy(), mode='classification', training_labels=None, feature_names=feature_names, class_names=['not_survived', 'survived'], discretize_continuous=True)

# Get the explanation for a single instance
explanation = explainer.explain_instance(background, f, num_features=input_size, top_labels=1)

# Plot the explanation
explanation.show_in_notebook(show_table=True, show_all=False)
