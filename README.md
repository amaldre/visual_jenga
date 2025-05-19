# visual_jenga

## Use Molmo

- **1**: Install dependencies : 

```bash
pip install einops torchvision
```

- **2**: Then import the following libraries : 

```bash
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
```

- **3**: To load the processor, use :

```bash
processor = AutoProcessor.from_pretrained(
    'allenai/Molmo-7B-O-0924',
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto'
)```

- **4**: To load the model, use :

```bash
model = AutoModelForCausalLM.from_pretrained(
    'allenai/Molmo-7B-O-0924',
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto'
)
```

## Use SAM 2 :