!pip install -q -U "torchao>=0.16" 2>&1 | tail -1
import torchao, peft
from peft.import_utils import is_torchao_available
print('torchao', torchao.__version__, '| torchao ok:', is_torchao_available())
