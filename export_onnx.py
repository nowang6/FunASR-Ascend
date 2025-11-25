from funasr import AutoModel

model = AutoModel(model="paraformer-zh-streaming")

# 导出模型，可以指定 output_dir 参数来设置保存位置
# 如果不指定 output_dir，默认保存在模型文件所在的目录（通常是缓存目录）
res = model.export(quantize=False, output_dir="./output")

print(f"导出模型保存位置: {res}")
