from pathlib import Path

def peek_inside():
    # 指向真实的验证集目录
    dataset_dir = Path("data/av2/sensor/val")
    
    # 获取所有的场景文件夹
    log_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
    
    if not log_dirs:
        print("❌ 还是没找到场景文件夹，检查一下路径名有没有拼错？")
        return
        
    first_log_dir = log_dirs[0]
    print(f"\n📂 正在查看场景文件夹: {first_log_dir.name}")
    print("-" * 40)
    print("📦 里面包含以下文件/子文件夹：")
    
    # 打印里面的所有内容
    for item in first_log_dir.iterdir():
        if item.is_dir():
            print(f"  📁 [文件夹] {item.name}")
        else:
            print(f"  📄 [文件]   {item.name}")
            
if __name__ == "__main__":
    peek_inside()