import os
import configparser
import gradio as gr
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import chromadb
from chromadb.utils import embedding_functions
import time

def read_config(file_path):
    config = configparser.ConfigParser()
    config.read(file_path)
    return config['DEFAULT']

# Read configuration from the config.ini file
config_file_path = os.path.join(os.path.dirname(__file__), 'config.ini')
config = read_config(config_file_path)

# Retrieve configuration variables
EMBEDDING_MODEL_REPO = config['EMBEDDING_MODEL_REPO']
EMBEDDING_MODEL_NAME = config['EMBEDDING_MODEL_NAME']
LLM_MODEL_NAME = config['LLM_MODEL_NAME']
COLLECTION_NAME = config['COLLECTION_NAME']
CHROMA_DATA_FOLDER = config['CHROMA_DATA_FOLDER']
HOST = config['HOST']
APP_PORT = int(config['APP_PORT'])
HF_ACCESS_TOKEN = config['HF_ACCESS_TOKEN']

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Clear GPU memory
torch.cuda.empty_cache()

# Connect to local Chroma data
chroma_client = chromadb.PersistentClient(path=CHROMA_DATA_FOLDER)
EMBEDDING_FUNCTION = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)

print("Initializing Chroma DB connection...")

try:
    collection = chroma_client.get_collection(name=COLLECTION_NAME, embedding_function=EMBEDDING_FUNCTION)
    print("Collection found.")
except:
    print("Creating new collection...")
    collection = chroma_client.create_collection(name=COLLECTION_NAME, embedding_function=EMBEDDING_FUNCTION)
    print("Collection created.")

# Get latest statistics from index
current_collection_stats = collection.count()
print('Total number of embeddings in Chroma DB index is ' + str(current_collection_stats))

# Quantization configuration
compute_dtype = getattr(torch, "float16")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
)

# Load model with model offloading
print("Loading model...")
with init_empty_weights():
    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME, quantization_config=bnb_config, token=HF_ACCESS_TOKEN)
print(f"Model loaded on device: {next(model.parameters()).device}")

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, trust_remote_code=True, token=HF_ACCESS_TOKEN)
tokenizer.pad_token = tokenizer.eos_token

# Define the inference function
def generate_response(prompt, max_new_tokens=100, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)  # Move inputs to device
    print(f"Inputs loaded on device: {inputs['input_ids'].device}")
    with torch.cuda.amp.autocast():
        output = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

def main():
    print("Configuring gradio app")

    DESC = "This AI-powered assistant showcases the flexibility of Cloudera Machine Learning to work with 3rd party solutions for LLMs and Vector Databases, as well as internally hosted models and vector DBs. The prototype does not yet implement chat history and session context - every prompt is treated as a brand new one."
    
    demo = gr.ChatInterface(
        fn=get_responses, 
        title="Enterprise Custom Knowledge Base Chatbot",
        description=DESC,
        additional_inputs=[
            gr.Radio(['Local Mistral 7B'], label="Select Foundational Model", value="Local Mistral 7B"), 
            gr.Slider(minimum=0.01, maximum=1.0, step=0.01, value=0.5, label="Select Temperature (Randomness of Response)"),
            gr.Radio(["50", "100", "250", "500", "1000"], label="Select Number of Tokens (Length of Response)", value="250"),
            gr.Radio(['Chroma'], label="Vector Database Choices", value="Chroma")
        ],
        retry_btn=None,
        undo_btn=None,
        clear_btn=None,
        autofocus=True
    )

    print("Launching gradio app")
    demo.launch(share=True, show_error=True, server_name=HOST, server_port=APP_PORT)
    print("Gradio app ready")

def get_responses(message, history, model, temperature, token_count, vector_db):
    if model == "Local Mistral 7B" and vector_db == "Chroma":
        context_chunk, metadata = get_nearest_chunk_from_chroma_vectordb(collection, message)
        # Format the prompt to include context but only show the answer
        prompt = f"Context: {context_chunk}\n\nQuestion: {message}\n\nAnswer:"
        response = generate_response(prompt, max_new_tokens=int(token_count), temperature=float(temperature))
        final_response = response.split("Answer:")[-1].strip()
        final_response += f"\n\nMetadata: {metadata}"
        for i in range(len(final_response)):
            time.sleep(0.02)
            yield final_response[:i+1]

def get_nearest_chunk_from_chroma_vectordb(collection, question):
    response = collection.query(
        query_texts=[question],
        n_results=1
    )
    return response['documents'][0][0], response['metadatas'][0][0]

if __name__ == "__main__":
    main()
