import ollama

def generate_response(prompt, messages, model="qwen2.5:3b-instruct-q8_0", rag_context=None):
    """Generate a response using the specified model and prompt."""
    system_prompt = "You are a helpful AI Agent. Respond to user queries strictly based on the provided context." \
    " If the context does not contain the answer, respond with 'I don't know'."
    
    messages.append({"role": "system", "content": system_prompt})
    
    if rag_context:
        messages.append({"role": "user", "content": f"Context: {rag_context}\n\n{prompt}"})
    else:
        messages.append({"role": "user", "content": prompt})
    
    response = ollama.chat(
        messages=messages,
        model=model
    )
    
    return response['message']['content']