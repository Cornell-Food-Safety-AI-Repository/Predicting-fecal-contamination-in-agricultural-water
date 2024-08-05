import pandas as pd
import argparse
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Function to convert 'N/P' values to binary
def convert_np_to_binary(value):
    if value == 'P':
        return 1
    elif value == 'N':
        return 0
    else:
        return pd.NA

# Parsing command line arguments
parser = argparse.ArgumentParser(description='Run machine learning models on the specified target label.')
parser.add_argument('--file_path', type=str, required=True, help='The path to your CSV file')
parser.add_argument('--target_label', type=str, required=True, help='Target label for the prediction model')
parser.add_argument('--algorithm', type=str, required=True, choices=['RandomForest', 'LinearRegression', 'SVM', 'GradientBoosting'], help='Machine learning algorithm to use')
parser.add_argument('--n_estimators', type=int, default=100, help='Number of trees in the forest (for RandomForest and GradientBoosting)')
parser.add_argument('--max_depth', type=int, default=None, help='Max depth of the tree (for RandomForest and GradientBoosting)')
parser.add_argument('--C', type=float, default=1.0, help='Regularization parameter (for SVM)')
parser.add_argument('--kernel', type=str, default='rbf', help='Kernel type to be used in SVM')

args = parser.parse_args()

# Load dataset
data = pd.read_csv(args.file_path)

# Apply the binary conversion function
for column in ['HF183_pa', 'Rum2Bac_pa', 'DG3_pa', 'GFD_pa']:
    data[column] = data[column].apply(convert_np_to_binary)

# Create a new label by combining the four specific labels
data['combined_label'] = data[['HF183_pa', 'Rum2Bac_pa', 'DG3_pa', 'GFD_pa']].max(axis=1)

# Print the number of ones in the combined_label column
num_ones = data['combined_label'].sum()
total_count = data['combined_label'].count()
print(f"Number of 1s in combined_label: {num_ones}")
print(f"Total number of entries in combined_label: {total_count}")

# Check for class imbalance
class_distribution = data['combined_label'].value_counts()
print("Class distribution in combined_label:")
print(class_distribution)

# Remove the original four labels and 'ecoli' from the dataset
data = data.drop(columns=['HF183_pa', 'Rum2Bac_pa', 'DG3_pa', 'GFD_pa', 'ecoli'])

# Separate features and target
X = data.drop(columns=['combined_label'])
y = data['combined_label']

# Print distribution of y
print("Distribution of y:")
print(y.value_counts())

# Impute numerical missing values
numerical_columns = X.select_dtypes(include=['number']).columns
numerical_imputer = SimpleImputer(strategy='mean')
X[numerical_columns] = numerical_imputer.fit_transform(X[numerical_columns])

# One-hot encode categorical data
categorical_columns = X.select_dtypes(include=['object']).columns
onehot_encoder = OneHotEncoder()
transformer = ColumnTransformer(transformers=[('cat', onehot_encoder, categorical_columns)], remainder='passthrough')
X_transformed = transformer.fit_transform(X)
transformed_columns = transformer.get_feature_names_out()

# Create DataFrame with encoded data
X_encoded = pd.DataFrame(X_transformed, columns=transformed_columns)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Print distribution of X_train
print("Distribution of X_train:")
print(X_train.describe())

# Print value counts for categorical columns in X_train
categorical_columns = [col for col in X_train.columns if X_train[col].nunique() < 10] # Assuming less than 10 unique values indicates categorical
for col in categorical_columns:
    print(f"Value counts for {col}:")
    print(X_train[col].value_counts())

# Choose the model based on command line argument and hyperparameters
models = {
    'RandomForest': RandomForestRegressor(n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=42),
    'LinearRegression': LinearRegression(),
    'SVM': SVR(C=args.C, kernel=args.kernel),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=42)
}

model = models[args.algorithm]

# Cross-validation to evaluate the model
cv_scores = cross_val_score(model, X_encoded, y, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Average cross-validation score: {cv_scores.mean()}")

# Train the model
model.fit(X_train, y_train)

# Perform prediction
y_pred = model.predict(X_test)

# Evaluate the model
if args.target_label == 'combined_label':
    # Binarize predictions for binary targets
    y_pred_binary = (y_pred > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred_binary)
    precision = precision_score(y_test, y_pred_binary)
    recall = recall_score(y_test, y_pred_binary)
    f1 = f1_score(y_test, y_pred_binary)
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")

    # Print confusion matrix
    cm = confusion_matrix(y_test, y_pred_binary)
    print("Confusion Matrix:")
    print(cm)
    
    # Plot confusion matrix (optional)
    # plt.figure(figsize=(10, 7))
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    # plt.xlabel('Predicted')
    # plt.ylabel('Actual')
    # plt.title('Confusion Matrix')
    # plt.show()
else:
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"MSE: {mse:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE: {mae:.3f}")
    print(f"R^2: {r2:.3f}")
