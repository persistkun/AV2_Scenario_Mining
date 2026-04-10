from pathlib import Path
import json

# 指向你的验证集目录
dataset_dir = Path("./data/av2/sensor/val")

# 获取所有场景文件夹
log_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]

print(f"📊 总场景数: {len(log_dirs)}")

# 遍历前 5 个场景（别一次看太多，电脑会卡）
for i, log_dir in enumerate(log_dirs[:5]):
    print(f"\n🔍 场景 {i+1}: {log_dir.name}")
    
    # 查看场景内部文件
    files = list(log_dir.iterdir())
    file_names = [f.name for f in files]
    
    print(f"📂 包含文件: {file_names}")
    
    # 尝试读取 JSON 配置（如果有）
    json_file = log_dir / "log_map_archive_xxx.json" # 名字会变，用 glob 找
    for f in files:
        if f.suffix == ".json":
            json_file = f
            break
            
    if json_file.exists():
        with open(json_file, 'r') as f:
            data = json.load(f)
        print(f"🗺️ 地图区域: {data.get('map_ranges', 'N/A')}")
    else:
        print("⚠️ 未找到 JSON 配置文件")

print("\n✅ 任务完成：你已经能批量读取场景结构！")