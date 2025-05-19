from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

processor = AutoProcessor.from_pretrained(
    'allenai/Molmo-7B-O-0924',
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto'
)

model = AutoModelForCausalLM.from_pretrained(
    'allenai/Molmo-7B-O-0924',
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto'
)