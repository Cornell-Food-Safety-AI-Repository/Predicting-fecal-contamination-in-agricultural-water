import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

# Function to convert 'N/P' values to binary
def convert_np_to_binary(value):
    if value == 'P':
        return 1
    elif value == 'N':
        return 0
    else:
        return pd.NA

# Load dataset
data = pd.read_csv('mstdata.csv')

# Apply the binary conversion function
for column in ['HF183_pa', 'Rum2Bac_pa', 'DG3_pa', 'GFD_pa']:
    data[column] = data[column].apply(convert_np_to_binary)

# Impute numerical missing values
numerical_columns = data.select_dtypes(include=['number']).columns
numerical_imputer = SimpleImputer(strategy='mean')
data[numerical_columns] = numerical_imputer.fit_transform(data[numerical_columns])

# One-hot encode categorical data
categorical_columns = data.select_dtypes(include=['object']).columns
onehot_encoder = OneHotEncoder()
transformer = ColumnTransformer(transformers=[('cat', onehot_encoder, categorical_columns)], remainder='passthrough')
data_transformed = transformer.fit_transform(data)
transformed_columns = transformer.get_feature_names_out()

# Create DataFrame with encoded data
data_encoded = pd.DataFrame(data_transformed, columns=transformed_columns)

# List target labels
target_labels = [col for col in data_encoded.columns if 'remainder__' in col]

# Parsing command line arguments
parser = argparse.ArgumentParser(description='Run machine learning models on the specified target label.')
parser.add_argument('--target_label', type=str, choices=target_labels, required=True, help='Target label for the prediction model')
parser.add_argument('--algorithm', type=str, choices=['RandomForest', 'LinearRegression', 'SVM', 'GradientBoosting'], required=True, help='Machine learning algorithm to use')
parser.add_argument('--n_estimators', type=int, default=100, help='Number of trees in the forest (for RandomForest and GradientBoosting)')
parser.add_argument('--max_depth', type=int, default=None, help='Max depth of the tree (for RandomForest and GradientBoosting)')
parser.add_argument('--C', type=float, default=1.0, help='Regularization parameter (for SVM)')
parser.add_argument('--kernel', type=str, default='rbf', help='Kernel type to be used in SVM')

args = parser.parse_args()

# Exclude specific feature columns for prediction
X = data_encoded.drop(target_labels, axis=1)
y = data_encoded[args.target_label]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Choose the model based on command line argument and hyperparameters
models = {
    'RandomForest': RandomForestRegressor(n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=42),
    'LinearRegression': LinearRegression(),
    'SVM': SVR(C=args.C, kernel=args.kernel),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=42)
}

model = models[args.algorithm]

# Train the model
model.fit(X_train, y_train)

# Perform prediction
y_pred = model.predict(X_test)

# Evaluate the model
if args.target_label == 'remainder__ecoli':
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"MSE: {mse:.3f}, R^2: {r2:.3f}")
else:
    # Binarize predictions for binary targets
    y_pred_binary = (y_pred > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred_binary)
    print(f"Accuracy: {accuracy:.3f}")
