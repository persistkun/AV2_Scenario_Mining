from pathlib import Path
import pandas as pd

# 👇 换成你挖到的场景ID
LOG_ID = "00010486-9a07-48ae-b493-cf4545855937"
DATASET_DIR = Path("./data/av2/sensor/val")

# 加载文件
log_dir = DATASET_DIR / LOG_ID
parquet_file = list(log_dir.glob("*.parquet"))[0]
df = pd.read_parquet(parquet_file)

# 打印所有信息
print("=== 这个场景的所有列名 ===")
for col in df.columns:
    print(f"- {col}")

print("\n=== 数据基本信息 ===")
print(f"总行数：{len(df)}，总列数：{len(df.columns)}")

print("\n=== 前3行数据预览 ===")
print(df.head(3))