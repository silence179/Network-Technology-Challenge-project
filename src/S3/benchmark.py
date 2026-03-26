import subprocess
import time
import sys

def run_and_time(script_name):
    print(f"[{script_name}] 开始运行...")
    start_time = time.time()
    
    try:
        # 使用 sys.executable 确保使用当前同一个 Python 解释器
        # 不捕获输出，让其直接打印到终端，这样你能看到实时进度
        subprocess.run([sys.executable, script_name], check=True)
        end_time = time.time()
        duration = end_time - start_time
        print(f"[{script_name}] 运行完成！耗时: {duration:.2f} 秒\n")
        return duration
    except subprocess.CalledProcessError as e:
        print(f"\n[错误] 运行 {script_name} 失败，退出码: {e.returncode}")
        return None
    except KeyboardInterrupt:
        print(f"\n[非正常退出] 用户手动中断了 {script_name} 的运行。")
        return None

def main():
    print("="*40)
    print("      脚本运行速度自动化性能比对      ")
    print("="*40)
    
    print("\n提醒：如果你使用的是完整数据集，测试可能需要较长时间。")
    print("建议可以在脚本里先配置只跑一个小块（比如 0_59900）来快速验证。\n")

    time_original = run_and_time("s3.py")
    time_optimized = run_and_time("s3_optimized.py")
    
    print("="*40)
    print("               测试结果               ")
    print("="*40)
    
    if time_original is not None and time_optimized is not None:
        print(f"原版脚本 (s3.py) 耗时:       {time_original:.2f} 秒")
        print(f"优化版脚本 (s3_optimized.py) 耗时: {time_optimized:.2f} 秒")
        
        if time_optimized > 0:
            speedup = time_original / time_optimized
            print(f"\n🎉 性能提升: 优化版比原版快了 {speedup:.2f} 倍！")
            
            time_saved = time_original - time_optimized
            print(f"节省了 {time_saved:.2f} 秒的时间。")
    else:
        print("由于某个脚本未成功运行完毕，无法计算速度差异。")

if __name__ == "__main__":
    main()
