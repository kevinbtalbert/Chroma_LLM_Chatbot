import os
import gradio as gr
import chromadb
from chromadb.utils import embedding_functions
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import requests
import json

# Connect to local Chroma data
chroma_client = chromadb.PersistentClient(path="/chroma-data")

EMBEDDING_MODEL_REPO = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
EMBEDDING_FUNCTION = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)

COLLECTION_NAME = os.getenv("COLLECTION_NAME")

print("initialising Chroma DB connection...")

print(f"Getting '{COLLECTION_NAME}' as object...")
try:
    chroma_client.get_collection(name=COLLECTION_NAME, embedding_function=EMBEDDING_FUNCTION)
    print("Success")
    collection = chroma_client.get_collection(name=COLLECTION_NAME, embedding_function=EMBEDDING_FUNCTION)
except:
    print("Creating new collection...")
    collection = chroma_client.create_collection(name=COLLECTION_NAME, embedding_function=EMBEDDING_FUNCTION)
    print("Success")

# Get latest statistics from index
current_collection_stats = collection.count()
print('Total number of embeddings in Chroma DB index is ' + str(current_collection_stats))

# Quantization
# Here quantization is setup to use "Normal Float 4" data type for weights. 
# This way each weight in the model will take up 4 bits of memory. 
compute_dtype = getattr(torch, "float16")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
)

# Create a model object with above parameters
model_name = "NousResearch/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, config=bnb_config)

# Define the inference function
def generate(prompt, max_length=100, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        temperature=temperature,
        do_sample=True
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Gradio interface
def llm_app(question, context="", temperature=0.7, token_count=100):
    llama_sys = "You are a helpful, respectful and honest assistant. If you are unsure about an answer, truthfully say \"I don't know\"."

    if context == "":
        llama_inst = f"Please answer the user question.[/INST]</s>"
        question_and_context = f"{llama_sys} {llama_inst} \n [INST] {question} [/INST]"
    else:
        llama_inst = f"Answer the user's question based on the following information:\n {context}[/INST]</s>"
        question_and_context = f"{llama_sys} {llama_inst} \n[INST] {question} [/INST]"

    try:
        # Generate response from the LLM
        response = generate(question_and_context, max_length=token_count, temperature=temperature)
        return str(response)[len(question_and_context)-2:]

    except Exception as e:
        print(e)
        return str(e)

# Gradio interface setup
iface = gr.Interface(
    fn=llm_app,
    inputs=["text", "text", "slider", "slider"],
    outputs="text",
    title="LLM Application"
)

# Run the app locally
if __name__ == "__main__":
    iface.launch(share=True, server_name="0.0.0.0", server_port=5000)

