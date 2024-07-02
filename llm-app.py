import os
import configparser
import gradio as gr
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import chromadb
from chromadb.utils import embedding_functions

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

# Connect to local Chroma data
chroma_client = chromadb.PersistentClient(path=CHROMA_DATA_FOLDER)
EMBEDDING_FUNCTION = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)

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
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME, config=bnb_config)

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
    demo.launch(share=True, enable_queue=True, show_error=True, server_name='127.0.0.1', server_port=int(os.getenv('CDSW_READONLY_PORT')))
    print("Gradio app ready")

def get_responses(message, history, model, temperature, token_count, vector_db):
    if model == "Local Mistral 7B":
        if vector_db == "Chroma":
            context_chunk, metadata = get_nearest_chunk_from_chroma_vectordb(collection, message)
            response = generate(message, context_chunk, temperature, token_count)
            response = f"{response}\n\nMetadata: {metadata}"
            for i in range(len(response)):
                time.sleep(0.02)
                yield response[:i+1]

def get_nearest_chunk_from_chroma_vectordb(collection, question):
    response = collection.query(
        query_texts=[question],
        n_results=1
    )
    return response['documents'][0][0], response['metadatas'][0][0]

def generate(question, context, temperature, token_count):
    mistral_sys = f"<s>[INST]You are a helpful, respectful and honest assistant. If you are unsure about an answer, truthfully say \"I don't know\"."
    if context == "":
        mistral_inst = f"Please answer the user question.[/INST]</s>"
        question_and_context = f"{mistral_sys} {mistral_inst} \n [INST] {question} [/INST]"
    else:
        mistral_inst = f"Answer the user's question based on the following information:\n {context}[/INST]</s>"
        question_and_context = f"{mistral_sys} {mistral_inst} \n[INST] {question} [/INST]"
        
    inputs = tokenizer(question_and_context, return_tensors="pt")
    output = model.generate(
        inputs["input_ids"],
        max_length=token_count,
        temperature=temperature,
        do_sample=True
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

if __name__ == "__main__":
    main()
