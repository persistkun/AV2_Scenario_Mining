import pandas as pd
import numpy as np
import os
import glob
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import logging

# 日志配置：记录真正有意义的错误
logging.basicConfig(level=logging.ERROR, filename='mining_errors.log', 
                    format='%(asctime)s - %(levelname)s - %(message)s')

class AV2EliteMiner:
    def __init__(self, ttc_threshold=2.0):
        self.ttc_threshold = ttc_threshold
        # 定义可能的坐标和速度列名映射 (AV2 常用 tx_m, ty_m 或 position_x, position_y)
        self.col_map = {
            'x': ['position_x', 'tx_m', 'x'],
            'y': ['position_y', 'ty_m', 'y'],
            'vx': ['velocity_x', 'vx_m', 'vx'],
            'vy': ['velocity_y', 'vy_m', 'vy']
        }

    def get_real_col(self, df_cols, key):
        """自动在数据集中寻找匹配的列名"""
        for candidate in self.col_map[key]:
            if candidate in df_cols:
                return candidate
        raise KeyError(f"在数据集中找不到列: {key}")

    def compute_ttc_vectorized(self, ego_df, other_df):
        try:
            # 1. 动态获取列名
            cols = ego_df.columns
            cx, cy = self.get_real_col(cols, 'x'), self.get_real_col(cols, 'y')
            cvx, cvy = self.get_real_col(cols, 'vx'), self.get_real_col(cols, 'vy')

            # 2. 精简数据列，加速 Merge
            ego_sub = ego_df[['timestep', cx, cy, cvx, cvy]]
            obs_sub = other_df[['timestep', cx, cy, cvx, cvy]]

            # 3. 向量化合并
            merged = pd.merge(ego_sub, obs_sub, on='timestep', suffixes=('_ego', '_obs'))
            if merged.empty: return np.inf

            # 4. 向量化物理计算
            dx = merged[f'{cx}_ego'] - merged[f'{cx}_obs']
            dy = merged[f'{cy}_ego'] - merged[f'{cy}_obs']
            dvx = merged[f'{cvx}_ego'] - merged[f'{cvx}_obs']
            dvy = merged[f'{cvy}_ego'] - merged[f'{cvy}_obs']

            dist_sq = dx**2 + dy**2
            # 相对速度投影 (Dot Product)
            rel_speed_proj = -(dx * dvx + dy * dvy)
            
            # 只有互相靠近且距离不为0才有意义
            mask = (rel_speed_proj > 0) & (dist_sq > 0.01)
            ttc = np.where(mask, dist_sq / (rel_speed_proj + 1e-6), np.inf)
            
            return np.min(ttc)
        except Exception:
            return np.inf

    def process_one_parquet(self, file_path):
        try:
            df = pd.read_parquet(file_path)
            
            # AV2 特有列名检测
            if 'track_id' not in df.columns: return None
            
            # 识别主车 (focal_track_id 是关键)
            focal_id = df.get('focal_track_id', [None]).iloc[0]
            if pd.isna(focal_id): return None
            
            ego_df = df[df['track_id'] == focal_id]
            # 排除自车和非运动物体
            other_vehicles = df[df['track_id'] != focal_id]

            # 按物体分组，批量计算 TTC
            for obs_id, obs_df in other_vehicles.groupby('track_id'):
                min_ttc = self.compute_ttc_vectorized(ego_df, obs_df)
                
                if min_ttc < self.ttc_threshold:
                    scenario_id = os.path.basename(file_path).replace('.parquet', '')
                    return f"{scenario_id},{obs_id},{min_ttc:.3f}"
            return None
        except Exception as e:
            logging.error(f"Error processing {file_path}: {str(e)}")
            return None

if __name__ == "__main__":
    # 路径递归搜索，确保能穿透到子文件夹
    data_path = "./data/av2/sensor/train/**/*.parquet" 
    data_files = glob.glob(data_path, recursive=True)
    
    print(f"🔍 路径扫描完成！找到场景文件: {len(data_files)} 个")
    
    if len(data_files) == 0:
        print("❌ 路径错误！请确保运行路径下存在 data/av2/sensor/train")
    else:
        miner = AV2EliteMiner(ttc_threshold=1.5) # 1.5s 是 Corner Case 的黄金阈值
        
        # 根据你的 30 核 CPU，推荐使用 12-16 进程以平衡 SSD 读写压力
        workers = 12 
        print(f"🚀 冠军级挖掘引擎启动 | 核心数: {workers} | 正在加速...")
        
        # 初始化输出文件
        output_file = "elite_danger_list.csv"
        with open(output_file, "w") as f:
            f.write("scenario_id,object_id,min_ttc\n")

        with ProcessPoolExecutor(max_workers=workers) as executor:
            # 增加 chunksize 减少进程间通信开销
            for res in tqdm(executor.map(miner.process_one_parquet, data_files, chunksize=20), total=len(data_files)):
                if res:
                    with open(output_file, "a") as f:
                        f.write(res + "\n")

    print(f"✅ 挖掘任务结束！请检查 {output_file}")