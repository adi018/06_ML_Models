# This example demonstrates how to create a simple agent using the OpenAI API
# and the ConversableAgent class from the autogen library.
# It uses a local Ollama server to handle the API requests.
import ollama
from autogen import ConversableAgent

# Define the custom function that AutoGen will use to talk to Ollama
def ollama_completion_function(messages, **kwargs):
    response = ollama.chat(
        model="llama3.2",  # or whatever model you have installed
        messages=messages
    )
    return {"role": "assistant", "content": response["message"]["content"]}

# Provide a valid llm_config that includes a model and the custom completion function
llm_config = {
    "model": "llama3.2:latest",  # Specify the model you want to use
    "completion_function": ollama_completion_function,
    "api_type": "openai",  # Required even though we’re overriding the call
    "api_base": "",         # Empty or dummy to satisfy pydantic
    "api_key": "dummy"      # Required field, not actually used
}

agent = ConversableAgent(
    name="chatbot",
    llm_config=llm_config,
    code_execution_config=False,
    human_input_mode="NEVER",
)

# Generate a reply using the agent
response = agent.generate_reply(
    messages=[{"role": "user", "content": "Tell me a joke about robots."}]
)

print(response)


