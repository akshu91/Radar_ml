import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    mean_squared_error
)
from sklearn.model_selection import train_test_split
import xgboost as xgb
import pickle

# === Config ===
CSV_PATH = 'sample_data.csv'
BURSTS_PER_LOCATION = 500
TOP_FEATURES = 30

# === 1. Load and aggregate ===
df = pd.read_csv(CSV_PATH)
df['location_id'] = df.index // BURSTS_PER_LOCATION

drop_cols = ['burstid', 'Object Type', 'Object Number', 'R', 'Angle', 'location_id']
feature_cols = [c for c in df.columns if c not in drop_cols]

agg_dict = {col: ['mean', 'std', 'min', 'max', 'median'] for col in feature_cols}
loc_stats = df.groupby('location_id').agg(agg_dict).reset_index()
loc_stats.columns = ['_'.join(col).strip() if col[1] else col[0] for col in loc_stats.columns.values]

for target in ['Object Type', 'Object Number', 'R', 'Angle']:
    loc_stats[target] = df.groupby('location_id')[target].first().values


# === 2. Remove rare classes ===
def drop_rare_classes(df, col):
    counts = df[col].value_counts()
    return df[df[col].isin(counts[counts >= 2].index)]


for target in ['Object Type', 'Object Number', 'Angle']:
    loc_stats = drop_rare_classes(loc_stats, target)


# === 3. Prepare features and targets ===
X = loc_stats.drop(['location_id', 'Object Type', 'Object Number', 'R', 'Angle'], axis=1)

y_object_type = loc_stats['Object Type']
y_object_number = loc_stats['Object Number']
y_R = loc_stats['R']
y_angle = loc_stats['Angle']

le_object_type = LabelEncoder()
le_object_number = LabelEncoder()
le_angle = LabelEncoder()

y_object_type_enc = le_object_type.fit_transform(y_object_type)
y_object_number_enc = le_object_number.fit_transform(y_object_number)
y_angle_enc = le_angle.fit_transform(y_angle)


# === 4. Stratified train/test splits ===
X_train1, X_test1, y_objtype_train, y_objtype_test = train_test_split(
    X, y_object_type_enc, test_size=0.2, random_state=42, stratify=y_object_type_enc
)

X_train2, X_test2, y_objnum_train, y_objnum_test = train_test_split(
    X, y_object_number_enc, test_size=0.2, random_state=42, stratify=y_object_number_enc
)

X_train3, X_test3, y_R_train, y_R_test = train_test_split(
    X, y_R, test_size=0.2, random_state=42
)

X_train4, X_test4, y_angle_train, y_angle_test = train_test_split(
    X, y_angle_enc, test_size=0.2, random_state=42, stratify=y_angle_enc
)


# === 5. Feature selection per target ===
def get_top_features(model, X, y, n=30):
    model.fit(X, y)
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1][:n]
    return X.columns[idx].tolist()


print("Selecting best features with XGBoost...")

xgb_cls = xgb.XGBClassifier(
    n_estimators=200, max_depth=8, random_state=0, verbosity=0, use_label_encoder=False
)
xgb_reg = xgb.XGBRegressor(
    n_estimators=200, max_depth=8, random_state=0, verbosity=0
)

top_objtype = get_top_features(xgb_cls, X_train1, y_objtype_train, TOP_FEATURES)
top_objnum = get_top_features(xgb_cls, X_train2, y_objnum_train, TOP_FEATURES)
top_R = get_top_features(xgb_reg, X_train3, y_R_train, TOP_FEATURES)
top_angle = get_top_features(xgb_cls, X_train4, y_angle_train, TOP_FEATURES)

print(f"\nTop features for Object Type: {top_objtype}")
print(f"Top features for Object Number: {top_objnum}")
print(f"Top features for R: {top_R}")
print(f"Top features for Angle: {top_angle}")


# === 6. Train final models ===
print("\nTraining XGBoost models...")

model_objtype = xgb.XGBClassifier(
    n_estimators=300, max_depth=12, random_state=42, verbosity=0, use_label_encoder=False
)
model_objnum = xgb.XGBClassifier(
    n_estimators=300, max_depth=12, random_state=42, verbosity=0, use_label_encoder=False
)
model_R = xgb.XGBRegressor(
    n_estimators=300, max_depth=12, random_state=42, verbosity=0
)
model_angle = xgb.XGBClassifier(
    n_estimators=300, max_depth=12, random_state=42, verbosity=0, use_label_encoder=False
)

model_objtype.fit(X_train1[top_objtype], y_objtype_train)
model_objnum.fit(X_train2[top_objnum], y_objnum_train)
model_R.fit(X_train3[top_R], y_R_train)
model_angle.fit(X_train4[top_angle], y_angle_train)


# === 7. Evaluate ===
print("\n--- VALIDATION RESULTS (Holdout set) ---")

print("\nObject Type accuracy:", accuracy_score(y_objtype_test, model_objtype.predict(X_test1[top_objtype])))
print("Confusion matrix:\n", confusion_matrix(y_objtype_test, model_objtype.predict(X_test1[top_objtype])))
print(classification_report(y_objtype_test, model_objtype.predict(X_test1[top_objtype])))

print("\nObject Number accuracy:", accuracy_score(y_objnum_test, model_objnum.predict(X_test2[top_objnum])))
print("Confusion matrix:\n", confusion_matrix(y_objnum_test, model_objnum.predict(X_test2[top_objnum])))
print(classification_report(y_objnum_test, model_objnum.predict(X_test2[top_objnum])))

print("\nR (distance) RMSE:", np.sqrt(mean_squared_error(y_R_test, model_R.predict(X_test3[top_R]))))

print("\nAngle accuracy:", accuracy_score(y_angle_test, model_angle.predict(X_test4[top_angle])))
print("Confusion matrix:\n", confusion_matrix(y_angle_test, model_angle.predict(X_test4[top_angle])))
print(classification_report(y_angle_test, model_angle.predict(X_test4[top_angle])))


# === 8. Save everything ===
with open('radar_xgb_models.pkl', 'wb') as f:
    pickle.dump({
        'model_objtype': model_objtype,
        'model_objnum': model_objnum,
        'model_R': model_R,
        'model_angle': model_angle,
        'le_object_type': le_object_type,
        'le_object_number': le_object_number,
        'le_angle': le_angle,
        'features_objtype': top_objtype,
        'features_objnum': top_objnum,
        'features_R': top_R,
        'features_angle': top_angle,
    }, f)

print("\nModels and encoders saved as radar_xgb_models.pkl")


# === 9. Prediction function ===
def predict_on_new_location(new_csv, bursts_per_location=500, model_file='radar_xgb_models.pkl'):
    with open(model_file, 'rb') as f:
        data = pickle.load(f)

    new_df = pd.read_csv(new_csv)
    new_df['location_id'] = 0

    drop_cols = ['burstid', 'Object Type', 'Object Number', 'R', 'Angle', 'location_id']
    feature_cols = [c for c in new_df.columns if c not in drop_cols]

    agg_dict = {col: ['mean', 'std', 'min', 'max', 'median'] for col in feature_cols}
    loc_features = new_df.groupby('location_id').agg(agg_dict).reset_index()
    loc_features.columns = ['_'.join(col).strip() if col[1] else col[0] for col in loc_features.columns.values]

    X_objtype = loc_features[data['features_objtype']]
    X_objnum = loc_features[data['features_objnum']]
    X_R = loc_features[data['features_R']]
    X_angle = loc_features[data['features_angle']]

    pred_objtype = data['le_object_type'].inverse_transform(
        data['model_objtype'].predict(X_objtype)
    )[0]
    pred_objnum = data['le_object_number'].inverse_transform(
        data['model_objnum'].predict(X_objnum)
    )[0]
    pred_R = data['model_R'].predict(X_R)[0]
    pred_angle = data['le_angle'].inverse_transform(
        data['model_angle'].predict(X_angle)
    )[0]

    print("\n--- Prediction for this location ---")
    print(f"Object Type: {pred_objtype}")
    print(f"Object Number: {pred_objnum}")
    print(f"R: {pred_R:.2f}")
    print(f"Angle: {pred_angle}")


# Example usage:
predict_on_new_location('test.csv')

