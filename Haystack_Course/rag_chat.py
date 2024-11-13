import warnings
warnings.filterwarnings('ignore')

import os
import rich

from dotenv import load_dotenv

load_dotenv()

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
OPENAI_API_TOKEN = os.getenv("OPENAI_API_TOKEN")
WS_API_TOKEN = os.getenv("WS_API_TOKEN")

import json

from typing import List, Optional
from haystack.utils import Secret
from haystack import Pipeline, component, Document
from haystack.components.builders import PromptBuilder
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.components.generators.chat.openai import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.components.routers import ConditionalRouter
from haystack.components.websearch.serper_dev import SerperDevWebSearch 

from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore
from haystack_integrations.components.retrievers.elasticsearch import ElasticsearchEmbeddingRetriever


document_store = ElasticsearchDocumentStore(hosts = "http://localhost:9200")

from haystack.components.converters.txt import TextFileToDocument
from haystack.components.preprocessors.document_splitter import DocumentSplitter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack.components.converters import OutputAdapter


@component
class QueryExpander:

    def __init__(self, open_ai_key: str , prompt: Optional[str] = None, model: str = "gpt-4o-mini"):

        self.query_expansion_prompt = prompt
        self.model = model
        self.open_ai_key = open_ai_key
        if prompt == None:
          self.query_expansion_prompt = """
          You are part of an information system that processes users queries.
          You expand a given query into {{ number }} queries that are similar in meaning.

          Structure:
          Follow the structure shown below in examples to generate expanded queries.
          The idea is to obtain as output a string containing a list with {{ number }} queries.

          Examples:
          Query: "climate change effects"
          Output: ["impact of climate change", "consequences of global warming", "effects of environmental changes"]

          Query: ""machine learning algorithms""
          Output: ["neural networks", "clustering", "supervised learning", "deep learning"]

          Query: "renewable energy sources"
          Output: ["solar power", "wind energy", "hydropower", "benefits of renewable energy", "green energy technologies"]

          Query: "mental health in adolescents"
          Output: ["teen depression", "anxiety in teenagers", "mental health support for youth", "effects of social media on mental health", "counseling for adolescents"]

          Query: "cybersecurity risks"
          Output: ["data breaches", "phishing attacks", "network security threats", "cybersecurity best practices", "malware protection"]

          Your Task:
          Query: "{{query}}"
          Output:
          """
        builder = PromptBuilder(self.query_expansion_prompt)
        llm = OpenAIGenerator(model = self.model, api_key = self.open_ai_key)
        self.pipeline = Pipeline()
        self.pipeline.add_component(name="builder", instance=builder)
        self.pipeline.add_component(name="llm", instance=llm)
        self.pipeline.connect("builder", "llm")

    @component.output_types(queries=List[str])
    def run(self, query: str, number: int = 5):
        result = self.pipeline.run({'builder': {'query': query, 'number': number}})
        print(result)
        expanded_query = json.loads(result['llm']['replies'][0]) + [query]
        
        return {"queries": list(expanded_query)}
    

@component
class MultiQueryElasticsearchEmbeddingRetriever:

    def __init__(self, retriever: ElasticsearchEmbeddingRetriever,  top_k: int = 3):
        self.retriever = retriever
        self.embedder = SentenceTransformersTextEmbedder("sentence-transformers/all-MiniLM-L6-v2")
        self.results = []
        self.ids = set()
        self.top_k = top_k

        self.pipeline = Pipeline()
        self.pipeline.add_component(name="embedder", instance=self.embedder)
        self.pipeline.add_component(name="retriever", instance=self.retriever)
        self.pipeline.connect("embedder", "retriever")
        

    def add_document(self, document: Document):
        if document.id not in self.ids:
            self.results.append(document)
            self.ids.add(document.id)

    @component.output_types(documents=List[Document])
    def run(self, queries: List[str], top_k: int = None):
        if top_k != None:
          self.top_k = top_k
        for query in queries:
          result = self.pipeline.run({"embedder": {'text':query}, "retriever": {"top_k": self.top_k}})
          for doc in result['retriever']['documents']:
            self.add_document(doc)
        self.results.sort(key=lambda x: x.score, reverse=True)
        return {"documents": self.results}
    
system_message = ChatMessage.from_system("You are a helpful AI assistant using provided supporting documents and conversation history to assist humans")

user_message_template ="""Given the conversation history and the provided supporting documents, give a brief answer to the question.
Note that supporting documents are not part of the conversation. 

If the answer is not contained within the documents, reply with 'no_answer'.

    Conversation history:
    {% for memory in memories %}
        {{ memory.content }}
    {% endfor %}

    Supporting documents:
    {% for doc in documents %}
        {{ doc.content }}
    {% endfor %}

    \nQuestion: {{query}}
    \nAnswer:
"""

user_message = ChatMessage.from_user(user_message_template)
query_rephrase_template = """
        Rewrite the question for search while keeping its meaning and key terms intact.
        If the conversation history is empty, DO NOT change the query.
        Use conversation history only if necessary, and avoid extending the query with your own knowledge.
        If no changes are needed, output the current question as is.

        If the answer is not contained within the documents, reply with 'no_answer'.

        Conversation history:
        {% for memory in memories %}
            {{ memory.content }}
        {% endfor %}

        User Query: {{query}}
        Rewritten Query:
"""
routes = [
    {
        "condition": "{{'no_answer' in replies[0]|lower}}",
        "output": "{{query}}",
        "output_name": "go_to_websearch", 
        "output_type": str,
    },
    {
        "condition": "{{'no_answer' not in replies[0]|lower}}",
        "output": "{{replies[0]}}",
        "output_name": "answer",
        "output_type": str,
    },
]
system_message_for_web = ChatMessage.from_system("""You are a helpful AI assistant using provided supporting documents from web to assist humans""")

prompt_for_websearch = """
Answer the following query given the documents retrieved from the web.
Your answer must indicate that your answer was generated from websearch, using the title "From Websearch:"
You can also reference the URLs that the answer was generated from

Query: {{query}}
Documents:
{% for document in documents %}
  {{document.content}}
{% endfor %}
"""

prompt_for_websearch = ChatMessage.from_user(prompt_for_websearch)
chat_generator = OpenAIChatGenerator(model="gpt-4o-mini-2024-07-18", api_key=Secret.from_token(OPENAI_API_TOKEN))


chat_agent = Pipeline()
chat_agent.add_component("expander", QueryExpander(open_ai_key=Secret.from_token(OPENAI_API_TOKEN)))
chat_agent.add_component("query_rephrase_prompt_builder", PromptBuilder(query_rephrase_template))
chat_agent.add_component("query_rephrase_llm", OpenAIGenerator(api_key=Secret.from_token(OPENAI_API_TOKEN)))
chat_agent.add_component("list_to_str_adapter", OutputAdapter(template="{{ replies[0] }}", output_type=str))
chat_agent.add_component("retriever", MultiQueryElasticsearchEmbeddingRetriever(retriever=ElasticsearchEmbeddingRetriever(document_store=document_store, top_k = 3)))
chat_agent.add_component("prompt_builder", ChatPromptBuilder(variables=["query", "documents", "memories"], required_variables=["query", "documents", "memories"]))
chat_agent.add_component("generator", chat_generator)
chat_agent.add_component("router", ConditionalRouter(routes=routes))
chat_agent.add_component("websearch", SerperDevWebSearch(api_key=Secret.from_token(WS_API_TOKEN))) 
chat_agent.add_component("prompt_builder_for_websearch", ChatPromptBuilder(variables=["query", "documents"], required_variables=["query", "documents"]))
chat_agent.add_component("llm_for_websearch",  OpenAIChatGenerator(model="gpt-4o-mini-2024-07-18", api_key=Secret.from_token(OPENAI_API_TOKEN)))


chat_agent.connect("query_rephrase_prompt_builder.prompt", "query_rephrase_llm")
chat_agent.connect("query_rephrase_llm.replies", "list_to_str_adapter")
chat_agent.connect("list_to_str_adapter.output", "expander")
chat_agent.connect("expander.queries", "retriever.queries")
chat_agent.connect("retriever.documents", "prompt_builder.documents")
chat_agent.connect("prompt_builder.prompt", "generator.messages")
chat_agent.connect("generator.replies", "router.replies")
chat_agent.connect("router.go_to_websearch", "websearch.query")
chat_agent.connect("router.go_to_websearch", "prompt_builder_for_websearch.query")
chat_agent.connect("websearch.documents", "prompt_builder_for_websearch.documents")
chat_agent.connect("prompt_builder_for_websearch.prompt", "llm_for_websearch.messages")

import streamlit as st

template = [system_message, user_message]
template_web = [system_message_for_web, prompt_for_websearch]

# Define the chat function as provided
def chat(template, messages, user_input): 
    # Run the chat agent with the specified parameters
    response = chat_agent.run({"query_rephrase_prompt_builder": {"query": user_input, "memories":messages}, 
                               "prompt_builder": {"template":template, "query": user_input, "memories":messages},
                               "router": {"query": user_input},
                               "prompt_builder_for_websearch": {"template":template_web,}},
                                include_outputs_from={"expander", "generator", "router"})
    
    if response['generator']['replies'][0].content != 'no_answer':
        messages.append(ChatMessage.from_user(user_input))
        messages.extend(response['generator']['replies'])
        return response['generator']['replies'][0].content
    else:
        messages.append(ChatMessage.from_user(user_input))
        messages.extend(response['llm_for_websearch']['replies'])
        return response['llm_for_websearch']['replies'][0].content

def main():
    st.title("💬 ChatGPT-style Chatbot")
    st.caption("🚀 A Streamlit chatbot powered by OpenAI")
    
    # Initialize session state for messages
    if "messages_chatbot" not in st.session_state:
        st.session_state["messages_chatbot"] = [{"role": "assistant", "content": "How can I help you?"}]
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Display chat history
    for msg in st.session_state.messages_chatbot:
        st.chat_message(msg["role"]).write(msg["content"])

    # User input section
    if prompt := st.chat_input():
        # Store the user message
        st.session_state.messages_chatbot.append({"role": "user", "content": prompt})
        
        st.chat_message("user").write(prompt)
        
        # Call the chat function
        bot_reply = chat(template, st.session_state.messages, prompt)
        
        # Store the bot's response
        st.session_state.messages_chatbot.append({"role": "assistant", "content": bot_reply})
        #print(st.session_state["messages"] )
        st.chat_message("assistant").write(bot_reply)


# Run Streamlit app
if __name__ == "__main__":
    main()




