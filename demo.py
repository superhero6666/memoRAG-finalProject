from memorag import MemoRAG

# Initialize MemoRAG pipeline
pipe = MemoRAG(
    mem_model_name_or_path="mistralai/Mistral-7B-Instruct-v0.3",
    ret_model_name_or_path="BAAI/bge-m3",
    gen_model_name_or_path="mistralai/Mistral-7B-Instruct-v0.3", # Optional: if not specify, use memery model as the generator
    cache_dir="cache/",  # Optional: specify local model cache directory
    access_token="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",  # Optional: Hugging Face access token
    beacon_ratio=16
)

context = open("examples/harry_potter.txt").read()
query = "How many times is the Chamber of Secrets opened in the book?"

# Memorize the context and save to cache
pipe.memorize(context, save_dir="cache/harry_potter/", print_stats=True)

# Generate response using the memorized context
res = pipe(context=context, query=query, task_type="memorag", max_new_tokens=256)
print(f"MemoRAG generated answer: \n{res}")




