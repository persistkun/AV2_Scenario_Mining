import pandas as pd
df = pd.read_csv("elite_danger_list.csv")
# 按照 TTC 从小到大排序，取前 5000 个最危险的
top_hard_cases = df.sort_values(by='min_ttc').head(5000)
top_hard_cases.to_csv("final_hard_cases.csv", index=False)
print("提纯完成，5000个冠军级难样本已就绪！")