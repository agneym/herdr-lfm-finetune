!pip install -q "transformers>=4.55" peft datasets accelerate
import transformers, peft, torch
print('transformers', transformers.__version__, '| peft', peft.__version__, '| cuda', torch.cuda.is_available())
