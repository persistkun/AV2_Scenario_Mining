import pandas as pd
import numpy as np
import os
import glob

def find_low_ttc_cases(df, ttc_threshold=0.05):
    """
    基于物理规则挖掘低 TTC (碰撞时间) 场景
    """
    # 1. 识别关键车辆 (AV2 场景中通常有一个 focal_track_id 作为主视角)
    # 或者是直接根据你之前打印出来的 track_id 识别自车
    # 这里的逻辑建议适配 AV2 的真实列名
    if 'focal_track_id' in df.columns:
        focal_id = df['focal_track_id'].iloc[0]
        ego_df = df[df['track_id'] == focal_id].copy()
    else:
        # 如果没有 focal_track_id，默认取出现频率最高或特定 ID
        return False

    if ego_df.empty:
        return False
        
    # 2. 提取其他车辆 (排除自车)
    other_vehicles_df = df[df['track_id'] != focal_id].copy()
    
    # 3. 按时间戳对齐自车和其他车
    # 注意：AV2 时间戳列名通常是 'timestep' 或 'timestamp_ns'
    time_col = 'timestep' if 'timestep' in df.columns else 'timestamp_ns'
    
    merged_df = pd.merge(ego_df, other_vehicles_df, on=time_col, suffixes=('_ego', '_other'))
    
    # 4. 计算相对距离 (欧几里得距离)
    # 根据你的数据确认列名是 'position_x' 还是 'vx' 等
    dx = merged_df['position_x_ego'] - merged_df['position_x_other']
    dy = merged_df['position_y_ego'] - merged_df['position_y_other']
    merged_df['distance'] = np.sqrt(dx**2 + dy**2)
    
    # 5. 计算相对速度 (标量简化版)
    # 如果数据里自带 velocity_x/y 直接用，没有的话需要 diff 计算
    v_ego = np.sqrt(merged_df['velocity_x_ego']**2 + merged_df['velocity_y_ego']**2)
    v_other = np.sqrt(merged_df['velocity_x_other']**2 + merged_df['velocity_y_other']**2)
    
    # 相对速度：自车比他车快多少
    merged_df['relative_speed'] = v_ego - v_other
    
    # 6. 计算 TTC
    # 只看距离小于 50 米且相对速度大于 0 的情况
    mask = (merged_df['relative_speed'] > 0.8) & (merged_df['distance'] < 50)
    merged_df['ttc'] = np.where(mask, 
                                merged_df['distance'] / merged_df['relative_speed'], 
                                np.inf)
    
    # 7. 判定
    danger_moments = merged_df[merged_df['ttc'] < ttc_threshold]
    
    if not danger_moments.empty:
        min_ttc = danger_moments['ttc'].min()
        print(f"⚠️ 发现潜在危险瞬间! 最小 TTC: {min_ttc:.2f}s")
        return True
        
    return False

if __name__ == "__main__":
    DATA_DIR = "data"
    OUTPUT_DIR = "output"
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    search_path = os.path.join(DATA_DIR, "**/*.parquet")
    data_files = glob.glob(search_path, recursive=True) 
    
    total_files = len(data_files)
    print(f"🚀 开始扫描 {total_files} 个场景文件...")
    
    danger_logs = []
    output_path = os.path.join(OUTPUT_DIR, "danger_list.txt")

    # 改进：用 enumerate 来显示进度 (第几个 / 总共几个)
    for idx, file in enumerate(data_files):
        try:
            # 打印实时进度
            if idx % 10 == 0: # 每 10 个文件报一次进度，省得刷屏太快
                print(f"进度: [{idx}/{total_files}] ... 正在处理: {os.path.basename(file)}")

            df = pd.read_parquet(file)
            if find_low_ttc_cases(df):
                log_id = os.path.basename(os.path.dirname(file))
                print(f"🎯 命中 Corner Case: {log_id}")
                
                # 改进：即时写入模式 ('a' 代表 append，追加写入)
                with open(output_path, "a") as f:
                    f.write(f"{log_id}\n")
                
                danger_logs.append(log_id)

        except Exception as e:
            print(f"❌ 处理出错: {e}")

    print(f"✅ 扫描结束！总共找到 {len(danger_logs)} 个场景。")