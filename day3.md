# 1. 手动加载抖音电商用户数据集，完成数据预处理；
```
import pandas as pd
import numpy as np

# 加载数据
file_name = "user_personalized_features.csv"
df = pd.read_csv(file_name)


print("Original Data Preview:")
print(df.head())
print("\nData Information:")
df.info()


df = df.drop(columns=['Unnamed: 0'])
print("DataFrame after dropping redundant columns:")
print(df.head())


# Min-Max Scaling for Income to [0, 1]
min_income = df['Income'].min()
max_income = df['Income'].max()
df['Income_MinMaxScaled'] = (df['Income'] - min_income) / (max_income - min_income)
print("\nIncome_MinMaxScaled:\n", df[['Income', 'Income_MinMaxScaled']].head())


mean_time_spent = df['Time_Spent_on_Site_Minutes'].mean()
std_time_spent = df['Time_Spent_on_Site_Minutes'].std()
df['Time_Spent_on_Site_Minutes_ZScaled'] = (df['Time_Spent_on_Site_Minutes'] - mean_time_spent) / std_time_spent
print("\nTime_Spent_on_Site_Minutes_ZScaled:\n", df[['Time_Spent_on_Site_Minutes', 'Time_Spent_on_Site_Minutes_ZScaled']].head())
```
# 2. 参考示例项目中的数据生成脚本，生成样本数据，手动给样本生成一些异常值、缺失值和重复值，并对其进行处理。
```

# src/data_generation.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_system_metrics(n_samples=5000, start_time="2025-07-12 06:00:00", fault_ratio=0.05):
    # 设置随机种子，以保证结果可复现
    np.random.seed(42)
    
    # 将开始时间字符串转化为 datetime 对象
    start_time = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    
    # 生成时间戳，间隔为 1 分钟
    timestamps = [start_time + timedelta(minutes=i) for i in range(n_samples)]
    
    # 随机选择设备（服务器）并为每个样本分配一个设备 ID
    devices = np.random.choice(['server_001', 'server_002'], size=n_samples, p=[0.5, 0.5])
    
    # 提取小时信息，用于计算 CPU 和 RAM 使用的周期性变化
    hours = np.array([t.hour for t in timestamps])
    
    # 模拟 CPU 使用率，模拟一个日周期的波动，并添加噪声
    cpu_usage = 50 + 20 * np.sin(2 * np.pi * hours / 24) + np.random.normal(0, 5, n_samples)
    
    # 模拟 RAM 使用率，模拟一个日周期的波动，并添加噪声
    ram_usage = 60 + 15 * np.sin(2 * np.pi * hours / 24) + np.random.normal(0, 5, n_samples)
    
    # 模拟磁盘 I/O 操作的随机值
    disk_io = np.random.normal(20, 5, n_samples)
    
    # 根据 CPU 使用率来模拟温度（温度受 CPU 使用影响）
    temperature = 35 + 0.2 * cpu_usage + np.random.normal(0, 2, n_samples)
    
    # 模拟错误计数，假设错误遵循泊松分布
    error_count = np.random.poisson(2, n_samples)
    
    # 对所有模拟数据进行限制，确保其在合理范围内
    cpu_usage = np.clip(cpu_usage, 0, 100)
    ram_usage = np.clip(ram_usage, 0, 100)
    disk_io = np.clip(disk_io, 0, 100)
    temperature = np.clip(temperature, 30, 50)
    error_count = np.clip(error_count, 0, 20)
    
    # 计算故障样本的数量
    n_faults = int(n_samples * fault_ratio)
    
    # 随机选择故障样本的索引
    fault_indices = np.random.choice(n_samples, size=n_faults, replace=False)
    
    # 创建标签数组，0 表示正常，1 表示故障
    labels = np.zeros(n_samples, dtype=int)
    labels[fault_indices] = 1
    
    # 为故障样本设置异常数据
    for idx in fault_indices:
        if np.random.random() < 0.9:  # 70%的故障是高负载故障
            # 高负载故障的模拟
            cpu_usage[idx] = np.random.uniform(90, 100)
            ram_usage[idx] = np.random.uniform(85, 95)
            temperature[idx] = np.random.uniform(45, 50)
            error_count[idx] = np.random.randint(10, 20)
        else:  # 30%的故障是低负载故障
            # 低负载故障的模拟
            cpu_usage[idx] = np.random.uniform(0, 10)
            ram_usage[idx] = np.random.uniform(0, 15)
            disk_io[idx] = np.random.uniform(0, 5)
            error_count[idx] = np.random.randint(5, 15)
    
    # 对 server_002 的温度增加 2 度，模拟不同设备的温度差异
    temperature[devices == 'server_002'] += 2
    
    # 确保温度值在合理范围内
    temperature = np.clip(temperature, 30, 50)
    
    # 将所有生成的数据放入 DataFrame 中
    data = pd.DataFrame({
        'timestamp': timestamps,  # 时间戳
        'device_id': devices,  # 设备 ID
        'cpu_usage': cpu_usage,  # CPU 使用率
        'ram_usage': ram_usage,  # RAM 使用率
        'disk_io': disk_io,  # 磁盘 I/O
        'temperature': temperature,  # 温度
        'error_count': error_count,  # 错误计数
        'label': labels  # 故障标签（0 表示正常，1 表示故障）
    })
    
    # 将数据保存为 CSV 文件
    data.to_csv("system_metrics2.csv", index=False)
    
    # 返回生成的数据
    #return data

if __name__ == "__main__":
    # 调用数据生成函数生成系统指标数据
    generate_system_metrics()

```
# 3. 使用同一个脚本，再生成一个样本数据集，测试将二者按行和按列合并
```
import pandas as pd
import numpy as np

# 加载数据
file_name = "system_metrics.csv"
file_name2 = "system_metrics2.csv"
df = pd.read_csv(file_name)
df2 = pd.read_csv(file_name2)

# concat (行连接)
print("按行连接数据:")
df_raw_connect = pd.concat([df, df2])
print(df_raw_connect.head())

# concat (列连接 - 需要对齐索引或使用 ignore_index=True)
df_colume_connect = pd.concat([df, df2], axis=1)
print("\n按列连接:")
print(df_colume_connect.head())
```
# 4. 训练一个LogisticRegresson模型，对一个分类数据集进行训练：要求使用pipeline和交叉验证。
```
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from sklearn.pipeline import Pipeline

# 1. 数据加载
digits = load_digits()
X, y = digits.data, digits.target


# 划分训练集和测试集（90%训练，10%测试）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 构建机器学习工作流Pipeline
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),  # 第一步: 特征标准化
    ('classifier', LogisticRegression(random_state=42))                 # 第二步: LogisticRegression，Pipeline中的最终模型，用于执行分类任务
])

# 定义 GridSearchCV 的超参数网格
param_grid = {
    'classifier__C': [0.1, 1.0, 10.0],  # 正则化参数
    'classifier__solver': ['lbfgs', 'liblinear']  # 优化器
}


# 初始化 GridSearchCV
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,  # 5 折交叉验证
    scoring='accuracy',
    n_jobs=-1  # 使用所有可用 CPU 核心
)

# 训练 GridSearchCV
grid_search.fit(X_train, y_train)

# 获取最佳模型
best_model = grid_search.best_estimator_

# 使用 cross_val_score 进行额外的交叉验证评估
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy')


# 预测训练集和测试集
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)


# 计算准确率
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# 打印结果
print("\n交叉验证准确率（每折）：")
print(cv_scores)
print("\n平均交叉验证准确率：")
print(f"{np.mean(cv_scores):.2f} ± {np.std(cv_scores):.2f}")
print("\n训练集准确率：")
print(f"{train_accuracy:.2f}")
print("\n测试集准确率：")
print(f"{test_accuracy:.2f}")
print("\n测试集分类报告：")
print(classification_report(y_test, y_test_pred))
print("\n最佳超参数：")
print(grid_search.best_params_)
print("\nGridSearchCV 最佳交叉验证准确率：")
print(f"{grid_search.best_score_:.2f}")
```
