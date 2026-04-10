import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm  # 导入进度条库

# --- 1. 车辆风险检测 (VEHICLE_RISK) ---
def find_low_ttc_cases(df, ttc_threshold=1.5):
    """检测两车之间的碰撞风险"""
    if 'focal_track_id' not in df.columns: return False
    focal_id = df['focal_track_id'].iloc[0]
    ego_df = df[df['track_id'] == focal_id].copy()
    
    # 动态获取类别列名
    cat_col = 'category' if 'category' in df.columns else 'object_type'
    # 修正点：使用变量 cat_col 而不是字符串 'cat_col'
    v_mask = (df['track_id'] != focal_id) & (df[cat_col].str.contains('VEHICLE', case=False, na=False))
    others_df = df[v_mask].copy()
    
    if ego_df.empty or others_df.empty: return False

    time_col = 'timestep' if 'timestep' in df.columns else 'timestamp_ns'
    merged = pd.merge(ego_df, others_df, on=time_col, suffixes=('_ego', '_other'))
    
    dist = np.sqrt((merged['position_x_ego'] - merged['position_x_other'])**2 + 
                   (merged['position_y_ego'] - merged['position_y_other'])**2)
    v_ego = np.sqrt(merged['velocity_x_ego']**2 + merged['velocity_y_ego']**2)
    v_other = np.sqrt(merged['velocity_x_other']**2 + merged['velocity_y_other']**2)
    
    rel_speed = v_ego - v_other
    ttc = np.where((rel_speed > 0.8) & (dist < 40), dist / rel_speed, np.inf)
    
    return ttc.min() < ttc_threshold

# --- 2. 弱势群体冲突检测 ---
def find_vru_conflict(df, ttc_threshold=2.0):
    if 'focal_track_id' not in df.columns: return False
    focal_id = df['focal_track_id'].iloc[0]
    ego_df = df[df['track_id'] == focal_id].copy()

    cat_col = 'category' if 'category' in df.columns else 'object_type'
    vru_mask = df[cat_col].str.contains('PEDESTRIAN|CYCLIST|MOTORCYCLIST', case=False, na=False)
    vru_df = df[vru_mask].copy()

    if ego_df.empty or vru_df.empty: return False

    time_col = 'timestep' if 'timestep' in df.columns else 'timestamp_ns'
    merged = pd.merge(ego_df, vru_df, on=time_col, suffixes=('_ego', '_vru'))

    dist = np.sqrt((merged['position_x_ego'] - merged['position_x_vru'])**2 + 
                   (merged['position_y_ego'] - merged['position_y_vru'])**2)
    v_ego = np.sqrt(merged['velocity_x_ego']**2 + merged['velocity_y_ego']**2)
    
    ttc = dist / (v_ego + 0.1) 
    return ((ttc < ttc_threshold) & (dist < 15)).any()

# --- 3. 统一调度中心 ---
def mine_everything(df):
    if find_vru_conflict(df):
        return True, "VRU_CRITICAL"
    if find_low_ttc_cases(df):
        return True, "VEHICLE_RISK"
    return False, "NORMAL"

# --- 4. 主程序：批量作业 ---
if __name__ == "__main__":
    DATA_DIR = "data"
    OUTPUT_DIR = "output"
    output_path = os.path.join(OUTPUT_DIR, "danger_list_labeled.txt")
    
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    data_files = glob.glob(os.path.join(DATA_DIR, "**/*.parquet"), recursive=True)
    total_files = len(data_files)
    
    print(f"🚀 雷达启动！目标文件总数: {total_files}")
    
    # 使用 tqdm 包装循环，desc 是描述，unit 是单位
    pbar = tqdm(data_files, desc="挖掘进度", unit="file")
    
    count = 0
    for file in pbar:
        try:
            df = pd.read_parquet(file)
            is_hit, label = mine_everything(df)
            
            if is_hit:
                log_id = os.path.basename(os.path.dirname(file))
                with open(output_path, "a") as f:
                    f.write(f"{log_id} | {label}\n")
                
                # 使用 pbar.write 打印，避免破坏进度条显示
                pbar.write(f"🎯 命中 [{label}]: {log_id}")
                count += 1
            
            # 在进度条右侧实时更新捕获数量
            pbar.set_postfix({"捕获": count})

        except Exception as e:
            pbar.write(f"❌ 场景解析出错: {e}")

    print(f"\n✅ 挖掘任务圆满完成！共捕获 {count} 个高价值场景。")
    print(f"📂 最终名单已保存至: {output_path}")