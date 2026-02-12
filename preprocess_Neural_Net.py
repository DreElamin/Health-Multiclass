import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

def main():
	df = pd.read_csv("dsppd.csv")

	xTrain, xTest, yTrain, yTest = preprocess(df)

	print(xTrain)
	print(yTrain.value_counts())
	print(xTest)
	print(yTest.value_counts())

	xTrain.to_csv('dsppd_xTrain.csv', index=False)
	yTrain.to_csv('dsppd_yTrain.csv', index=False)
	xTest.to_csv('dsppd_xTest.csv', index=False)
	yTest.to_csv('dsppd_yTest.csv', index=False)


def preprocess(df):
	#1. Drop negative outcomes
	#2. split into test/train
	#3. group rare diseases into 'other' category
	#4. one-hot encode Gender
	#5. binarize Fever, Cough, Fatigue, Difficulty Breathing
	#6. Encode Low:-1, Normal:0, High:1 for Cholesterol Level, Blood Pressure
	#7. standardize Age


	#1. drop negative outcomes
	#df = df[df["Outcome Variable"] == "Positive"]
	df = df.drop("Outcome Variable", axis=1)

	respiratory = {
	"Asthma", "Bronchitis", "Common Cold", "Influenza", "Pneumonia",
	"Allergic Rhinitis"
	}
	cardiovascular = {
	"Hypertension", "Stroke", "Coronary Artery Disease", "Heart Failure",
	"Myocardial Infarction"
	}
	endocrine_metabolic = {
	"Diabetes", "Hyperthyroidism", "Hypothyroidism", "Obesity",
	"Metabolic Syndrome"
	}
	gi_renal = {
	"Gastroenteritis", "Crohn's Disease", "Ulcerative Colitis",
	"Pancreatitis", "Kidney Disease", "Urinary Tract Infection",
	"Irritable Bowel Syndrome", "Kidney Stones"
	}
	musc_autoimmune = {
	"Rheumatoid Arthritis", "Osteoarthritis", "Osteoporosis", "Lupus",
	"Psoriasis", "Multiple Sclerosis"
	}
	neurological = {
	"Migraine", "Epilepsy", "Parkinson's Disease", "Alzheimer's Disease",
	"Multiple Sclerosis"
	}
	psychiatric = {
	"Depression", "Anxiety Disorders", "Bipolar Disorder",
	"Schizophrenia", "PTSD"
	}
	cancer = {
	"Liver Cancer", "Kidney Cancer", "Lung Cancer", "Breast Cancer",
	"Colon Cancer"
	}

	for disease in df['Disease']:
		if disease in respiratory: df.replace(disease, "respiratory", inplace=True)
		if disease in cardiovascular: df.replace(disease, "cardiovascular", inplace=True)
		if disease in endocrine_metabolic: df.replace(disease, "endocrine_metabolic", inplace=True)
		if disease in gi_renal: df.replace(disease, "gi_renal", inplace=True)
		if disease in musc_autoimmune: df.replace(disease, "musc_autoimmune", inplace=True)
		if disease in neurological: df.replace(disease, "neurological", inplace=True)
		if disease in psychiatric: df.replace(disease, "psychiatric", inplace=True)
		if disease in cancer: df.replace(disease, "cancer", inplace=True)

	print(df)
	
	#2. split into test/train
	y = df["Disease"]

	X = df.drop("Disease", axis=1)

	xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2)

	xTrain = xTrain.reset_index(drop=True)
	xTest = xTest.reset_index(drop=True)

	yTrain = yTrain.reset_index(drop=True)
	yTest = yTest.reset_index(drop=True)

	#3. group rare diseases into 'other' category
	val_counts = yTrain.value_counts()
	for val in yTrain.unique():
		if val_counts[val] <= 5: yTrain.replace(val, "Other", inplace=True)

	for val in yTest.unique():
		if val not in val_counts or val_counts[val] <= 5: yTest.replace(val, "Other", inplace=True)

	#4. Onehot encode Gender for train and test
	oneHot = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
	encoded_features = oneHot.fit_transform(xTrain[['Gender']])
	feature_names = oneHot.get_feature_names_out(['Gender'])
	oneHot_df = pd.DataFrame(encoded_features, columns=feature_names)

	xTrain = pd.concat([xTrain.drop(columns=['Gender'], axis=1), oneHot_df], axis=1)

	encoded_features = oneHot.transform(xTest[['Gender']])
	feature_names = oneHot.get_feature_names_out(['Gender'])
	oneHot_df = pd.DataFrame(encoded_features, columns=feature_names)

	xTest = pd.concat([xTest.drop(columns=['Gender'], axis=1), oneHot_df], axis=1)
	


	#5. binarize (yes/no -> 1/0) Fever, Cough, Fatigue, Difficulty Breathing
	for col in ["Fever", "Cough", "Fatigue", "Difficulty Breathing"]:
		xTest[col] = xTest[col].replace({'Yes': 1, 'No': 0})
		xTrain[col] = xTrain[col].replace({'Yes': 1, 'No': 0})

	#6. Encode Low:-1, Normal:0, High:1 for Cholesterol Level, Blood Pressure
	for col in ["Cholesterol Level", "Blood Pressure"]:
		xTest[col] = xTest[col].replace({'Low': -1, 'Normal': 0, 'High': 1})
		xTrain[col] = xTrain[col].replace({'Low': -1, 'Normal': 0, 'High': 1})

	#7. Standardize Age
	std = StandardScaler()
	xTrain["Age"] = std.fit_transform(xTrain[["Age"]])
	xTest["Age"] = std.transform(xTest[["Age"]])

	return xTrain, xTest, yTrain, yTest

if __name__ == "__main__":
    main()

