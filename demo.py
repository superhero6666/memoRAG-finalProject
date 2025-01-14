from memorag import MemoRAG

# Initialize MemoRAG pipeline
pipe = MemoRAG(
    # mistralai/Mistral-7B-Instruct-v0.3
    mem_model_name_or_path="shenzhi-wang/Llama3.1-8B-Chinese-Chat",
    ret_model_name_or_path="BAAI/bge-m3",
    gen_model_name_or_path="shenzhi-wang/Llama3.1-8B-Chinese-Chat", # Optional: if not specify, use memery model as the generator
    cache_dir="cache/",  # Optional: specify local model cache directory
    access_token="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",  # Optional: Hugging Face access token
    beacon_ratio=16
)

# 使用正确的编码打开文件
with open("examples/weicheng.txt", encoding='utf-8') as f:
    context = f.read()

query = "主人公是什么样的人？"

# Memorize the context and save to cache
pipe.memorize(context, save_dir="/cache/weicheng/", print_stats=True)

# Generate response using the memorized context
res = pipe(context=context, query=query, task_type="memorag", max_new_tokens=256)
print(f"MemoRAG generated answer: \n{res}")




