from model import mgn
import torch
import yaml
# Định nghĩa các tham số của model
# args =  config_str
  # Tham số của model (nếu có)

# Khởi tạo model
# num_classes = 751
# config_file = '/home/linhphuong/Documents/documents/persion_re_id/MGN-pytorch-modify/config.yml'
# with open(config_file, 'r') as cf:
#     config_str = "\n" + cf.read()

# args = yaml.safe_load(config_str)
# args = config_str
# num_classes = args['num_classes']
# model = mgn.MGN(args)
class ConfigObject:
    pass

# Đường dẫn đến tệp YAML
yaml_file_path = '/home/linhphuong/Documents/documents/persion_re_id/MGN-pytorch-modify/config.yml'

# Đọc tệp YAML
with open(yaml_file_path, 'r') as file:
    yaml_data = yaml.safe_load(file)

# Tạo một đối tượng từ dữ liệu YAML
config = ConfigObject()
for key, value in yaml_data.items():
    setattr(config, key, value)
model = mgn.MGN(config)
print(model)
# Đường dẫn đến file chứa trọng số đã lưu
path_to_weights = '/home/linhphuong/Documents/documents/persion_re_id/MGN-pytorch-modify/model_1.pth'

model_load = torch.load(path_to_weights)
for key in list(model_load.keys()):
    new_key = key.replace('model.', '')
    model_load[new_key] = model_load.pop(key)
# Tải trọng số đã lưu
model.load_state_dict(model_load)

# Chuyển model sang chế độ đánh giá (evaluation mode)
model.eval()