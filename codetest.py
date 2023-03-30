import os
from flax import traverse_util
from flax.training import checkpoints
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
params = checkpoints.restore_checkpoint('pcaTest2', target=None)["params"]
# 使用traverse_util提供的flatten_dict方法將所有參數攤平
flat_params = traverse_util.flatten_dict(params)

# 顯示每一層的參數
for k, v in flat_params.items():
    if k == ('Dense_0', 'kernel'):
        print(v[0][:5])