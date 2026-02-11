import sarenv
import os

generator = sarenv.DataGenerator()

center_point = (104.0, 30.0)  # 经度, 纬度
output_directory = "./sarenv_dataset"
environment_type = "mixed"       # urban/rural/mixed
environment_climate = "temperate"  # temperate/arid/tropical

os.makedirs(output_directory, exist_ok=True)

# 只生成 small/medium/large，避免 xlarge 报错
generator.export_dataset(
    center_point,
    output_directory,
    environment_type,
    environment_climate,
    sizes=["small", "medium", "large"]
)

print(f"SARenv 数据集生成完成，目录: {output_directory}")

