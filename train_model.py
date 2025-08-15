import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

print("--- Starting Model Evaluation Process ---")

# --- 1. Load the Trained Models and the New Dataset ---
try:
    # Load the models you trained earlier
    category_classifier = joblib.load("models/category_classifier.joblib")
    print("Category classifier loaded.")
    
    # Load your new, large dataset of real complaints
    df = pd.read_csv("real_complaints.csv")
    print("Real complaints dataset loaded successfully.")
    
    # Prepare the text data in the same way as the training script
    df['text'] = df['title'] + " " + df['description']
    df.dropna(subset=['text', 'category'], inplace=True)
    
    X_real = df['text']
    y_true = df['category'] # The actual, correct categories

except FileNotFoundError as e:
    print(f"\nERROR: Could not find a required file: {e.filename}")
    print("Please make sure 'models/category_classifier.joblib' and 'real_complaints.csv' exist.\n")
    exit()
except Exception as e:
    print(f"An error occurred: {e}")
    exit()

# --- 2. Make Predictions on the New Data ---
print("\nRunning predictions on the new dataset...")
y_pred = category_classifier.predict(X_real)
print("Predictions complete.")

# --- 3. Analyze and Report the Results ---
print("\n--- Model Performance Report ---")

# Calculate overall accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"Overall Model Accuracy: {accuracy:.2%}\n")

# Show a detailed report of precision, recall, and f1-score for each category
print("Detailed Classification Report:")
print(classification_report(y_true, y_pred, zero_division=0))

# --- 4. Find and Display Mismatches ---
print("\n--- Analyzing Incorrect Predictions ---")
mismatches = []
for i, (true_label, pred_label) in enumerate(zip(y_true, y_pred)):
    if true_label != pred_label:
        mismatches.append({
            "text": X_real.iloc[i],
            "true_category": true_label,
            "predicted_category": pred_label
        })

if not mismatches:
    print("🎉 Excellent! No mismatches found. The model is performing perfectly on this dataset.")
else:
    print(f"Found {len(mismatches)} incorrect predictions. Here are some examples:\n")
    for i, mismatch in enumerate(mismatches[:15]): # Print the first 15 mismatches
        print(f"--- Mismatch #{i+1} ---")
        print(f"Complaint Text: \"{mismatch['text'][:100]}...\"")
        print(f"    > Correct Category:   '{mismatch['true_category']}'")
        print(f"    > Model Predicted:    '{mismatch['predicted_category']}'\n")

# --- 5. (Optional) Visualize the Confusion Matrix ---
try:
    labels = sorted(list(y_true.unique()))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Category')
    plt.ylabel('True Category')
    plt.title('Model Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    print("\nSaved a 'confusion_matrix.png' image to visualize the results.")
except Exception as e:
    print(f"\nCould not generate confusion matrix plot. Error: {e}")
    print("You might need to install seaborn and matplotlib: pip install seaborn matplotlib")

print("\n--- Evaluation Complete ---")
