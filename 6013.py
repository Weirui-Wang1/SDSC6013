import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, precision_recall_curve
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost import XGBClassifier
import lightgbm
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import ADASYN
import joblib
import torch  # 新增，用于检查GPU是否可用

# 检查GPU是否可用
use_gpu = torch.cuda.is_available()

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# 1. 设置随机种子和matplotlib相关设置
np.random.seed(42)
# %matplotlib inline  # 如果在脚本中运行，不需要这行
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 2. 定义文件路径并检查文件是否存在
base_path = '.'  # 当前目录，可根据实际情况修改
required_files = [
    'application_train.csv', 'application_test.csv', 'bureau.csv', 'previous_application.csv',
    'POS_CASH_balance.csv', 'installments_payments.csv', 'bureau_balance.csv',
    'credit_card_balance.csv'
]

for file in required_files:
    file_path = os.path.join(base_path, file)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: {file_path} not found. Please check the file path.")
    else:
        print(f"Found {file_path}")

# 3. 加载主数据
print("Loading 'application_train.csv' file...")
train_data = pd.read_csv(os.path.join(base_path, 'application_train.csv'))
print("Loading 'application_test.csv' file...")
test_data = pd.read_csv(os.path.join(base_path, 'application_test.csv'))


# 4. 增强特征工程函数
def enhanced_feature_engineering(df):
    df = df.copy()
    # 构建新特征，这些特征从业务角度反映了信用相关的信息
    # CREDIT_ANNUITY_RATIO 反映了信用额度与年金的比例，可体现还款压力
    df['CREDIT_ANNUITY_RATIO'] = df['AMT_CREDIT'] / (df['AMT_ANNUITY'] + 1)
    # CREDIT_GOODS_PRICE_RATIO 反映了信用额度与商品价格的比例
    df['CREDIT_GOODS_PRICE_RATIO'] = df['AMT_CREDIT'] / (df['AMT_GOODS_PRICE'] + 1)
    # INCOME_PER_PERSON 反映了人均收入情况
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / (df['CNT_FAM_MEMBERS'] + 1)
    # EXT_SOURCE_INTERACTION 综合了外部数据源2和3的信息
    df['EXT_SOURCE_INTERACTION'] = df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    # CREDIT_EXT_SOURCE_2 结合了信用额度和外部数据源2的信息
    df['CREDIT_EXT_SOURCE_2'] = df['AMT_CREDIT'] * df['EXT_SOURCE_2']
    # AGE_INT 将年龄转换为整数形式
    df['AGE_INT'] = (df['DAYS_BIRTH'] / -365).astype(int)
    # EXT_SOURCE_1_2 综合了外部数据源1和2的信息
    df['EXT_SOURCE_1_2'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2']
    # CREDIT_INCOME_INTERACTION 结合了信用额度和人均收入的信息
    df['CREDIT_INCOME_INTERACTION'] = df['AMT_CREDIT'] * df['INCOME_PER_PERSON']
    return df


train_data = enhanced_feature_engineering(train_data)
test_data = enhanced_feature_engineering(test_data)

# 处理bureau.csv
print("Loading and processing 'bureau.csv'...")
bureau = pd.read_csv(os.path.join(base_path, 'bureau.csv'))
bureau = bureau[bureau['SK_ID_CURR'].isin(train_data['SK_ID_CURR'].tolist() + test_data['SK_ID_CURR'].tolist())].copy()
bureau_features = bureau.groupby('SK_ID_CURR').agg({
    'SK_ID_BUREAU': 'count',
    'AMT_CREDIT_SUM': 'mean',
    'DAYS_CREDIT': 'mean'
}).reset_index()
bureau_features.columns = ['SK_ID_CURR', 'BUREAU_COUNT', 'BUREAU_CREDIT_SUM_MEAN', 'BUREAU_DAYS_CREDIT_MEAN']

# 处理bureau_balance.csv
print("Loading and processing 'bureau_balance.csv'...")
bureau_balance = pd.read_csv(os.path.join(base_path, 'bureau_balance.csv'))
bureau_balance = bureau_balance[bureau_balance['SK_ID_BUREAU'].isin(bureau['SK_ID_BUREAU'])].copy()
bureau_balance_agg = bureau_balance.groupby('SK_ID_BUREAU').agg({
    'MONTHS_BALANCE': 'min',
    'STATUS': lambda x: (x == 'C').mean()
}).reset_index()
bureau_balance_agg.columns = ['SK_ID_BUREAU', 'BUREAU_BALANCE_MIN_MONTH', 'BUREAU_BALANCE_C_STATUS_RATIO']
bureau_balance_agg = bureau.merge(bureau_balance_agg, on='SK_ID_BUREAU', how='left').groupby('SK_ID_CURR').agg({
    'BUREAU_BALANCE_MIN_MONTH': 'mean',
    'BUREAU_BALANCE_C_STATUS_RATIO': 'mean'
}).reset_index()

# 处理previous_application.csv
print("Loading and processing 'previous_application.csv'...")
prev_app = pd.read_csv(os.path.join(base_path, 'previous_application.csv'))
prev_app = prev_app[
    prev_app['SK_ID_CURR'].isin(train_data['SK_ID_CURR'].tolist() + test_data['SK_ID_CURR'].tolist())].copy()
prev_app_features = prev_app.groupby('SK_ID_CURR').agg({
    'SK_ID_PREV': 'count',
    'AMT_APPLICATION': 'mean',
    'RATE_DOWN_PAYMENT': 'mean'
}).reset_index()
prev_app_features.columns = ['SK_ID_CURR', 'PREV_APP_COUNT', 'PREV_APP_AMT_MEAN', 'PREV_APP_RATE_DOWN_PAYMENT_MEAN']

# 处理installments_payments.csv
print("Loading and processing 'installments_payments.csv'...")
installments = pd.read_csv(os.path.join(base_path, 'installments_payments.csv'))
installments = installments[
    installments['SK_ID_CURR'].isin(train_data['SK_ID_CURR'].tolist() + test_data['SK_ID_CURR'].tolist())].copy()
# 计算逾期天数
installments['PAST_DUE_DAYS'] = (installments['DAYS_ENTRY_PAYMENT'] - installments['DAYS_INSTALMENT']).clip(lower=0)
installments_features = installments.groupby('SK_ID_CURR').agg({
    'SK_ID_PREV': 'count',
    'AMT_PAYMENT': 'mean',
    'PAST_DUE_DAYS': ['mean', lambda x: x.tail(3).mean()]
}).reset_index()
installments_features.columns = ['SK_ID_CURR', 'INSTALL_COUNT', 'INSTALL_AMT_MEAN', 'INSTALL_PAST_DUE_MEAN',
                                 'INSTALL_PAST_DUE_LAST3']

# 处理POS_CASH_balance.csv
print("Loading and processing 'POS_CASH_balance.csv'...")
pos_cash = pd.read_csv(os.path.join(base_path, 'POS_CASH_balance.csv'))
pos_cash = pos_cash[
    pos_cash['SK_ID_CURR'].isin(train_data['SK_ID_CURR'].tolist() + test_data['SK_ID_CURR'].tolist())].copy()
# 标记是否逾期
pos_cash['OVERDUE'] = (pos_cash['SK_DPD'] > 0).astype(int)
pos_cash_features = pos_cash.groupby('SK_ID_CURR').agg({
    'SK_ID_PREV': 'count',
    'OVERDUE': 'mean',
    'CNT_INSTALMENT': 'mean'
}).reset_index()
pos_cash_features.columns = ['SK_ID_CURR', 'POS_COUNT', 'POS_OVERDUE_RATE', 'POS_INSTALMENT_MEAN']

# 处理credit_card_balance.csv
print("Loading and processing 'credit_card_balance.csv'...")
credit_card = pd.read_csv(os.path.join(base_path, 'credit_card_balance.csv'))
credit_card = credit_card[
    credit_card['SK_ID_CURR'].isin(train_data['SK_ID_CURR'].tolist() + test_data['SK_ID_CURR'].tolist())].copy()
# 计算信用卡利用率
credit_card['CREDIT_UTILIZATION'] = credit_card['AMT_BALANCE'] / (credit_card['AMT_CREDIT_LIMIT_ACTUAL'] + 1)
credit_card_features = credit_card.groupby('SK_ID_CURR').agg({
    'SK_ID_PREV': 'count',
    'CREDIT_UTILIZATION': 'mean',
    'AMT_PAYMENT_TOTAL_CURRENT': 'mean'
}).reset_index()
credit_card_features.columns = ['SK_ID_CURR', 'CC_COUNT', 'CC_UTILIZATION_MEAN', 'CC_PAYMENT_MEAN']

# 6. 合并特征
train_data = train_data.merge(bureau_features, on='SK_ID_CURR', how='left')
train_data = train_data.merge(bureau_balance_agg, on='SK_ID_CURR', how='left')
train_data = train_data.merge(prev_app_features, on='SK_ID_CURR', how='left')
train_data = train_data.merge(installments_features, on='SK_ID_CURR', how='left')
train_data = train_data.merge(pos_cash_features, on='SK_ID_CURR', how='left')
train_data = train_data.merge(credit_card_features, on='SK_ID_CURR', how='left')
print(f"Training data shape after merging features: {train_data.shape}")

test_data = test_data.merge(bureau_features, on='SK_ID_CURR', how='left')
test_data = test_data.merge(bureau_balance_agg, on='SK_ID_CURR', how='left')
test_data = test_data.merge(prev_app_features, on='SK_ID_CURR', how='left')
test_data = test_data.merge(installments_features, on='SK_ID_CURR', how='left')
test_data = test_data.merge(pos_cash_features, on='SK_ID_CURR', how='left')
test_data = test_data.merge(credit_card_features, on='SK_ID_CURR', how='left')
print(f"Test data shape after merging features: {test_data.shape}")


# 7. 数据预处理函数
def preprocess_data(df):
    df = df.copy()
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    # 使用中位数填补数值型缺失值，中位数不受极端值影响，适合有偏态分布的数据
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    # 使用众数填补分类型缺失值，能最大程度维持分类变量的分布特征
    df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])
    le = LabelEncoder()
    for column in categorical_cols:
        df[column] = le.fit_transform(df[column])
    return df


train_data = preprocess_data(train_data)
test_data = preprocess_data(test_data)

# 8. 划分数据和选择特征
X = train_data.drop(columns=['TARGET', 'SK_ID_CURR'])
y = train_data['TARGET']
# 使用互信息进行特征选择，选择前100个特征
selector = SelectKBest(mutual_info_classif, k=100)
X_selected = selector.fit_transform(X, y)
selected_indices = selector.get_support()
selected_features = X.columns[selected_indices].tolist()
print(f"Selected features (top 100): {selected_features}")

X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42, stratify=y)

adasyn = ADASYN(sampling_strategy=0.9, random_state=42)
X_train, y_train = adasyn.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

test_X = test_data[selected_features]
test_X = scaler.transform(test_X)

# 9. 构建并训练基础模型
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale_pos_weight = neg_count / pos_count

xgb_params = {
    'learning_rate': 0.05,
    'max_depth': 8,
    'min_child_weight': 2,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'tree_method': 'gpu_hist' if use_gpu else 'hist',  # 修改
    'device': 'gpu' if use_gpu else 'cpu',  # 修改
    'n_jobs': -1,
    'eval_metric': 'auc',
    'scale_pos_weight': scale_pos_weight,
    'random_state': 42
}

models = {
    'XGBoost': XGBClassifier(**xgb_params, n_estimators=200),
    'LightGBM': LGBMClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=10, num_leaves=70,
        min_child_samples=15, random_state=42, class_weight='balanced',
        subsample=0.8, colsample_bytree=0.8,
        device='gpu' if use_gpu else 'cpu',  # 修改
        gpu_platform_id=0 if use_gpu else None,  # 修改
        gpu_device_id=0 if use_gpu else None,  # 修改
        n_jobs=-1
    ),
    'CatBoost': CatBoostClassifier(
        iterations=200, learning_rate=0.05, depth=8, random_state=42,
        class_weights=[1, scale_pos_weight], verbose=0,
        task_type='GPU' if use_gpu else 'CPU',  # 修改
        thread_count=-1
    )
}

# 选择这些模型的原因：
# XGBoost：具有高效的计算性能和强大的泛化能力，能够处理大规模数据和高维特征，对不平衡数据也有较好的处理能力。
# LightGBM：训练速度快，内存占用少，在处理大规模数据时表现出色，同时支持多种提升算法和特征组合。
# CatBoost：对分类特征有很好的处理能力，能够自动处理缺失值和异常值，减少了数据预处理的工作量。

kf = KFold(n_splits=3, shuffle=True, random_state=42)
base_predictions_train = {name: np.zeros(X_train.shape[0]) for name in models}
base_predictions_test = {name: np.zeros(X_test.shape[0]) for name in models}
test_predictions = {name: np.zeros(test_X.shape[0]) for name in models}

for name, model in models.items():
    print(f"Training {name} with cross-validation...")
    if name == 'XGBoost':
        for train_idx, val_idx in kf.split(X_train, y_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            dtrain = xgb.DMatrix(X_tr, label=y_tr)
            dval = xgb.DMatrix(X_val, label=y_val)
            bst = xgb.train(
                xgb_params,
                dtrain,
                num_boost_round=200,
                evals=[(dval, 'eval')],
                early_stopping_rounds=50,
                verbose_eval=False
            )
            base_predictions_train[name][val_idx] = bst.predict(dval)
        model.fit(X_train, y_train)
        base_predictions_test[name] = model.predict_proba(X_test)[:, 1]
        test_predictions[name] = model.predict_proba(test_X)[:, 1]
    else:
        for train_idx, val_idx in kf.split(X_train, y_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            if name == 'LightGBM':
                model.fit(X_tr, y_tr,
                          eval_set=[(X_val, y_val)],
                          eval_metric='auc',
                          callbacks=[lightgbm.early_stopping(stopping_rounds=50)])
            else:  # CatBoost
                model.fit(X_tr, y_tr,
                          eval_set=(X_val, y_val),
                          early_stopping_rounds=50,
                          verbose=0)
            base_predictions_train[name][val_idx] = model.predict_proba(X_val)[:, 1]
        model.fit(X_train, y_train)
        base_predictions_test[name] = model.predict_proba(X_test)[:, 1]
        test_predictions[name] = model.predict_proba(test_X)[:, 1]

    precision, recall, thresholds = precision_recall_curve(y_test, base_predictions_test[name])
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_threshold = thresholds[np.argmax(f1_scores)]
    y_pred_test = (base_predictions_test[name] >= best_threshold).astype(int)
    print(f"{name} Performance (Optimized Threshold):")
    print(f"Best Threshold: {best_threshold}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_test)}")
    print(f"Recall: {recall_score(y_test, y_pred_test)}")
    print(f"F1 Score: {f1_score(y_test, y_pred_test)}")
    print(f"ROC AUC: {roc_auc_score(y_test, base_predictions_test[name])}\n")

# 10. 模型融合（Stacking）
stacking_train = np.column_stack([base_predictions_train[name] for name in models])
stacking_test = np.column_stack([base_predictions_test[name] for name in models])
test_stacking = np.column_stack([test_predictions[name] for name in models])

meta_model = RandomForestClassifier(n_estimators=100, random_state=42)
meta_model.fit(stacking_train, y_train)
y_pred_proba_meta = meta_model.predict_proba(stacking_test)[:, 1]
test_y_pred_proba_meta = meta_model.predict_proba(test_stacking)[:, 1]

precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba_meta)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
best_threshold = thresholds[np.argmax(f1_scores)]
y_pred_adjusted = (y_pred_proba_meta >= best_threshold).astype(int)
test_y_pred_adjusted = (test_y_pred_proba_meta >= best_threshold).astype(int)

print("\nStacking Model Performance (Optimized Threshold):")
print(f"Best Threshold: {best_threshold}")
print(f"Accuracy: {accuracy_score(y_test, y_pred_adjusted)}")
print(f"Recall: {recall_score(y_test, y_pred_adjusted)}")
print(f"F1 Score: {f1_score(y_test, y_pred_adjusted)}")
print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba_meta)}")

# 保存测试集预测结果
test_predictions_df = pd.DataFrame({
    'SK_ID_CURR': test_data['SK_ID_CURR'],
    'XGBoost': test_predictions['XGBoost'],
    'LightGBM': test_predictions['LightGBM'],
    'CatBoost': test_predictions['CatBoost'],
    'Stacking': test_y_pred_proba_meta
})
test_predictions_df.to_csv('test_predictions.csv', index=False)

# 11. 生成报告相关内容
# 模型性能对比表格
results_df = pd.DataFrame({
    'Model': list(models.keys()) + ['Stacking'],
    'F1 Score': [f1_score(y_test, (base_predictions_test[name] >= best_threshold).astype(int)) for name in models] +
                [f1_score(y_test, y_pred_adjusted)]
})

# 绘制模型性能对比柱状图
plt.figure(figsize=(10, 6))
sns.barplot(data=results_df, x='Model', y='F1 Score')
plt.title('Model F1 Score Comparison')
plt.xlabel('Model')
plt.ylabel('F1 Score')
plt.savefig('f1_comparison.png')
plt.show()

# 12. 金融方面的分析
# 违约概率分布
for name, preds in base_predictions_test.items():
    plt.figure(figsize=(10, 6))
    sns.histplot(preds, kde=True)
    plt.title(f'{name} Default Probability Distribution')
    plt.xlabel('Default Probability')
    plt.ylabel('Count')
    plt.savefig(f'{name}_default_prob_distribution.png')
    plt.show()

# 信用等级划分
credit_rating_thresholds = [0.1, 0.3, 0.5, 0.7]
credit_ratings = ['A', 'B', 'C', 'D', 'E']

for name, preds in base_predictions_test.items():
    ratings = []
    for pred in preds:
        if pred < credit_rating_thresholds[0]:
            ratings.append(credit_ratings[0])
        elif pred < credit_rating_thresholds[1]:
            ratings.append(credit_ratings[1])
        elif pred < credit_rating_thresholds[2]:
            ratings.append(credit_ratings[2])
        elif pred < credit_rating_thresholds[3]:
            ratings.append(credit_ratings[3])
        else:
            ratings.append(credit_ratings[4])
    rating_counts = pd.Series(ratings).value_counts()
    print(f"{name} Credit Rating Distribution:")
    print(rating_counts)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=rating_counts.index, y=rating_counts.values)
    plt.title(f'{name} Credit Rating Distribution')
    plt.xlabel('Credit Rating')
    plt.ylabel('Count')
    plt.savefig(f'{name}_credit_rating_distribution.png')
    plt.show()


# 敏感性分析
def sensitivity_analysis(model, X_test, feature_index, step=0.1):
    original_preds = model.predict_proba(X_test)[:, 1]
    feature_values = X_test[:, feature_index].copy()
    sensitivities = []
    for multiplier in [0.9, 1.0, 1.1]:
        new_X_test = X_test.copy()
        new_X_test[:, feature_index] = feature_values * multiplier
        new_preds = model.predict_proba(new_X_test)[:, 1]
        sensitivity = np.mean(np.abs(new_preds - original_preds))
        sensitivities.append(sensitivity)
    return sensitivities


feature_names = np.array(selected_features)
for name, model in models.items():
    print(f"{name} Sensitivity Analysis:")
    sensitivities = []
    for i in range(X_test.shape[1]):
        sens = sensitivity_analysis(model, X_test, i)
        sensitivities.append(sens)
    sensitivities = np.array(sensitivities)
    sensitivity_df = pd.DataFrame(sensitivities, columns=['0.9x', '1.0x', '1.1x'], index=feature_names)
    print(sensitivity_df)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=sensitivity_df.index, y=sensitivity_df['1.1x'])
    plt.title(f'{name} Feature Sensitivity Analysis (1.1x)')
    plt.xlabel('Feature')
    plt.ylabel('Sensitivity')
    plt.xticks(rotation=90)
    plt.savefig(f'{name}_sensitivity_analysis.png')
    plt.show()

# 新增：特征重要性分析
for name, model in models.items():
    if name == 'XGBoost':
        feature_importance = model.feature_importances_
    elif name == 'LightGBM':
        feature_importance = model.feature_importances_
    elif name == 'CatBoost':
        feature_importance = model.get_feature_importance()
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    print(f"{name} Feature Importance:")
    print(feature_importance_df.head(10))
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10))
    plt.title(f'{name} Top 10 Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.savefig(f'{name}_feature_importance.png')
    plt.show()

# 新增：风险收益分析
# 假设贷款金额为1000，利率为0.1，违约损失率为0.5
loan_amount = 1000
interest_rate = 0.1
loss_given_default = 0.5

for name, preds in base_predictions_test.items():
    expected_profit = []
    for pred in preds:
        profit = (1 - pred) * loan_amount * interest_rate - pred * loan_amount * loss_given_default
        expected_profit.append(profit)
    average_profit = np.mean(expected_profit)
    print(f"{name} Average Expected Profit: {average_profit}")
    plt.figure(figsize=(10, 6))
    sns.histplot(expected_profit, kde=True)
    plt.title(f'{name} Expected Profit Distribution')
    plt.xlabel('Expected Profit')
    plt.ylabel('Count')
    plt.savefig(f'{name}_expected_profit_distribution.png')
    plt.show()

# 新增：贷款组合风险分析
for name, preds in base_predictions_test.items():
    ratings = []
    for pred in preds:
        if pred < credit_rating_thresholds[0]:
            ratings.append(credit_ratings[0])
        elif pred < credit_rating_thresholds[1]:
            ratings.append(credit_ratings[1])
        elif pred < credit_rating_thresholds[2]:
            ratings.append(credit_ratings[2])
        elif pred < credit_rating_thresholds[3]:
            ratings.append(credit_ratings[3])
        else:
            ratings.append(credit_ratings[4])
    portfolio_df = pd.DataFrame({'Rating': ratings, 'Default Probability': preds})
    portfolio_risk = portfolio_df.groupby('Rating').agg({'Default Probability': ['mean', 'count']})
    portfolio_risk.columns = ['Average Default Probability', 'Count']
    portfolio_risk['Expected Loss'] = portfolio_risk['Average Default Probability'] * loan_amount * loss_given_default * \
                                      portfolio_risk['Count']
    print(f"{name} Loan Portfolio Risk Analysis:")
    print(portfolio_risk)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=portfolio_risk.index, y=portfolio_risk['Expected Loss'])
    plt.title(f'{name} Loan Portfolio Expected Loss')
    plt.xlabel('Credit Rating')
    plt.ylabel('Expected Loss')
    plt.savefig(f'{name}_loan_portfolio_expected_loss.png')
    plt.show()

# 保存模型
joblib.dump(meta_model, 'stacking_model.pkl')

print("Model performance comparison:")
print(results_df)
