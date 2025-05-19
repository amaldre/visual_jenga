# visual_jenga

## To do before starting the project

### (OPTIONNEL) Use Molmo

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
)
```

- **4**: To load the model, use :

```bash
model = AutoModelForCausalLM.from_pretrained(
    'allenai/Molmo-7B-O-0924',
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto'
)
```

### Use SAM 2 :

To install SAM 2, you need `python>=3.10`, as well as `torch>=2.5.1` and `torchvision>=0.20.1`.

- **1** : Clone the rep :

```bash
git clone https://github.com/facebookresearch/sam2.git && cd sam2

pip install -e .
```