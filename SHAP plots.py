train_predict = model(train_x) # predict Weibull parameters using covariates

with torch.no_grad():
    train_predict = train_predict.clone().resize_(p,4)

# train_predict = model(train_x[:100]) # predict Weibull parameters using covariates
# train_predict = train_predict.resize_(p, 2)
train_predict = pd.DataFrame(train_predict.detach().numpy()) # convert to dataframe
train_predict.columns = ["pred_alpha", "pred_beta", "pred_gamma","pred_landa"] # name columns
train_result = df_train.copy()
train_result.reset_index(inplace = True) # reset the index (before concat - probably better way of doing this)
train_result = pd.concat([train_result, train_predict], axis=1) # results = test data plus predictions
train_result.set_index("index", drop=True, inplace=True) # recover the index (after concat - probably better way of doing this)
t_max = df_train["time"].max()
num_vals = max(math.ceil(t_max), 50)
t_vals = np.linspace(0, t_max, num_vals)
surv =  weibull_surv(t_vals, train_result["pred_alpha"].to_numpy(),
                     train_result["pred_beta"].to_numpy(),
                     train_result["pred_gamma"].to_numpy(),
                     train_result["pred_landa"].to_numpy())
surv = pd.DataFrame(data=surv, index=t_vals)

surv11 = np.array(surv)

def weibull_surv(t, a, b , c, d):

    S = np.empty((len(t),len(a)))

    for i in range(len(a)):
        S[:,i] = np.exp(-(a[i] * np.power(t,b[i])) - (c[i] * np.power(t,d[i])))

    return S



feature_names = df_test.drop(['time', 'death'], axis=1).columns.tolist()

def f(data):
    # Convert numpy data to tensor and ensure it is on the correct device
    if isinstance(data, np.ndarray):
        data = torch.tensor(data, dtype=torch.float32)
        if torch.cuda.is_available():
            data = data.to('cuda')

    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        predictions = model(data)  # Make predictions

    # Convert predictions back to numpy for SHAP compatibility
    if predictions.is_cuda:
        predictions = predictions.cpu()  # Move predictions to CPU if needed

    predictions = predictions.numpy()  # Convert torch tensor to numpy array
    predictions_df = pd.DataFrame(predictions, columns=["pred_alpha", "pred_beta", "pred_gamma", "pred_lambda"])

    # Define a specific time point for survival probability calculation
    t_specific = 90  # You can choose any specific time point relevant to your analysis

    # Calculate survival probabilities at the specific time point
    survival_probs = weibull_surv(np.array([t_specific]), predictions_df["pred_alpha"], predictions_df["pred_beta"],
                                  predictions_df["pred_gamma"], predictions_df["pred_lambda"]).flatten()
# survival_probs
    threshold =0.5
    survival_probs = np.where(survival_probs <= threshold, 0,1)
    return survival_probs  # Return one-dimensional array of survival probabilities at t_specific

# Prepare SHAP Explainer and Values
import shap
# Ensure background is in NumPy format
background = shap.sample(train_x, 300)
if isinstance(background, torch.Tensor):
    background = background.cpu().numpy()  # Convert to numpy if it's a tensor

f(background)


# Initialize the SHAP explainer
explainer_nn = shap.KernelExplainer(f, background)

# Compute SHAP values for the specific time point
shap_values_nn = explainer_nn.shap_values(test_x.cpu().numpy())  # Convert test data to numpy

shap.summary_plot(shap_values_nn, test_x[:304].cpu().numpy(), feature_names=df_test[:30].columns)

# feature_names = df_test.columns.tolist()
feature_names = df_test.drop(['time', 'death'], axis=1).columns.tolist()

# Visualize the SHAP summary plot
shap.summary_plot(shap_values_nn, test_x[:3003].cpu().numpy(), feature_names=feature_names, plot_type="bar", show=False)
plt.title("SHAP Summary Plot (Neural Network)")
plt.show()

feature_names = [
    'USMER',
    'MEDICAL_UNIT',
    'SEX',
    'PATIENT_TYPE',
    'PNEUMONIA',
    'AGE',
    'PREGNANT',
    'DIABETES',
    'COPD',
    'ASTHMA',
    'INMSUPR',
    'HIPERTENSION',
    'OTHER_DISEASE',
    'CARDIOVASCULAR',
    'OBESITY',
    'RENAL_CHRONIC',
    'TOBACCO',
    'CLASIFFICATION_FINAL'
]



# Plot SHAP dependence plot for specific feature pairs
def plot_shap_dependence(shap_values, X, feature1, feature2, feature_names):
    """
    Generate a SHAP dependence plot showing interactions between two features.

    Parameters:
        shap_values (numpy.ndarray): SHAP values array.
        X (numpy.ndarray): Input dataset.
        feature1 (str): The primary feature to plot.
        feature2 (str): The secondary feature to interact with.
        feature_names (list): List of feature names.
    """
    feature1_idx = feature_names.index(feature1)
    feature2_idx = feature_names.index(feature2)

    shap.dependence_plot(
        feature1_idx,
        shap_values,
        X,
        interaction_index=feature2_idx,
        feature_names=feature_names
    )

# Create SHAP dependence plots for all pairs of features
def create_all_pair_dependence_plots(shap_values, X, feature_names):
    """
    Generate SHAP dependence plots for all pairs of features.

    Parameters:
        shap_values (numpy.ndarray): SHAP values array.
        X (numpy.ndarray): Input dataset.
        feature_names (list): List of feature names.
    """
    num_features = len(feature_names)
    for i in range(num_features):
        for j in range(i + 1, num_features):
            plt.figure()
            plot_shap_dependence(shap_values, X, feature_names[i], feature_names[j], feature_names)
            plt.title(f"SHAP Interaction: {feature_names[i]} vs {feature_names[j]}")
            plt.show()

# Generate dependence plots for all pairs of features
create_all_pair_dependence_plots(shap_values_nn, test_x[:304].cpu().numpy(), feature_names)
