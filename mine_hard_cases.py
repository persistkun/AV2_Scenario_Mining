import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm
import sys

# --- 1. 车辆风险检测 (VEHICLE_RISK) ---
def find_low_ttc_cases(df, ttc_threshold=1.5):
    if 'focal_track_id' not in df.columns: return False
    focal_id = df['focal_track_id'].iloc[0]
    ego_df = df[df['track_id'] == focal_id].copy()
    
    cat_col = 'category' if 'category' in df.columns else 'object_type'
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

# --- 2. 弱势群体冲突检测 (鬼探头) ---
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

# --- 4. 主程序 ---
if __name__ == "__main__":
    DATA_DIR = "/mnt/c/AV2_Scenario_Mining/data/av2/sensor/train" 
    csv_output = "final_hard_cases_linux.csv"
    
    # 1. 跨系统扫文件（这步没法跳过，但只做一次）
    print("🔍 正在扫描磁盘文件，这可能需要几分钟，请不要断开...", flush=True)
    data_files = glob.glob(os.path.join(DATA_DIR, "**/*.parquet"), recursive=True)
    total_files = len(data_files)
    
    if total_files == 0:
        print("❌ 错误：没找到任何文件，请确认路径正确！")
        sys.exit()

    print(f"🚀 发现 {total_files} 个场景！开始挖掘难样本...", flush=True)

    # 2. 准备实时保存的文件头
    with open(csv_output, 'w') as f:
        f.write("scenario_id,label\n")

    # 3. 核心循环，带详细进度条
    count = 0
    # tqdm 是灵魂：它会告诉你每一秒处理了多少，还需要多久
    pbar = tqdm(data_files, desc="挖掘进度", unit="file", dynamic_ncols=True)
    
    for file in pbar:
        try:
            # 这里的 engine='fastparquet' 能显著提升读取速度
            df = pd.read_parquet(file, engine='fastparquet')
            is_hit, label = mine_everything(df)
            
            if is_hit:
                log_id = os.path.basename(os.path.dirname(file))
                # 实时写入文件，防止意外
                with open(csv_output, 'a') as f:
                    f.write(f"{log_id},{label}\n")
                
                count += 1
                # 在进度条上方打印命中信息，很有成就感
                pbar.write(f"🎯 命中 [{label}]: {log_id} | 当前共捕获: {count}")
            
            # 更新右侧的实时统计
            pbar.set_postfix({"已捕获": count}, refresh=True)

        except Exception as e:
            # pbar.write(f"⚠️ 跳过受损场景: {file}")
            continue

    print(f"\n✅ 任务完成！总共捕获了 {count} 个极难场景，快去训练吧！")