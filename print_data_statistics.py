# 文件名：data_statistics.py
import pandas as pd
import os


def get_row_count(file_path):
    """高效获取CSV文件行数"""
    with open(file_path, 'r') as f:
        row_count = sum(1 for line in f) - 1  # 减去标题行
    return row_count


def count_features_from_processed_data(base_path='.'):
    """统计处理后的特征数量"""
    try:
        # 加载主数据
        train = pd.read_csv(os.path.join(base_path, 'application_train.csv'), nrows=0)

        # 主表原始特征（排除标识列和目标列）
        original_features = len(train.columns) - 2  # 排除SK_ID_CURR和TARGET

        # 特征工程新增特征数
        enhanced_features = 8  # 根据enhanced_feature_engineering函数

        # 辅助表特征数（根据处理逻辑）
        auxiliary_features = {
            'bureau.csv': 3,
            'bureau_balance.csv': 2,
            'previous_application.csv': 3,
            'installments_payments.csv': 4,
            'POS_CASH_balance.csv': 3,
            'credit_card_balance.csv': 3
        }

        return original_features, enhanced_features, sum(auxiliary_features.values())

    except Exception as e:
        print(f"Error reading data: {str(e)}")
        return 0, 0, 0


def generate_data_report(base_path='.'):
    """生成完整数据报告"""
    # 定义需要统计的文件列表
    required_files = [
        'application_train.csv', 'application_test.csv',
        'bureau.csv', 'previous_application.csv',
        'POS_CASH_balance.csv', 'installments_payments.csv',
        'bureau_balance.csv', 'credit_card_balance.csv'
    ]

    # 存储统计结果
    statistics = {
        'tables': {},
        'features': {}
    }

    # 统计各表行数
    print("正在统计各表数据量...")
    for file in required_files:
        path = os.path.join(base_path, file)
        if os.path.exists(path):
            statistics['tables'][file] = get_row_count(path)
        else:
            statistics['tables'][file] = 0

    # 统计特征数量
    print("\n正在分析特征工程...")
    original, enhanced, auxiliary = count_features_from_processed_data(base_path)

    statistics['features'] = {
        'main_table_original': original,
        'enhanced_features': enhanced,
        'auxiliary_features': auxiliary,
        'total_before_selection': original + enhanced + auxiliary,
        'selected_features': 100  # 根据SelectKBest(k=100)
    }

    # 打印报告
    print("\n" + "=" * 40)
    print("数据统计报告".center(40))
    print("=" * 40)

    print("\n一、数据表统计:")
    for table, count in statistics['tables'].items():
        print(f"{table.ljust(25)}: {str(count).rjust(10)} 行")

    print("\n二、特征工程统计:")
    print(f"主表原始特征数: {statistics['features']['main_table_original']}")
    print(f"+ 特征工程新增特征: {statistics['features']['enhanced_features']}")
    print(f"+ 辅助表提取特征: {statistics['features']['auxiliary_features']}")
    print(f"= 合并总特征数: {statistics['features']['total_before_selection']}")
    print(f"最终选择特征数: {statistics['features']['selected_features']}")



if __name__ == "__main__":
    generate_data_report()