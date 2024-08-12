#!/usr/bin/env python
# coding: utf-8

# <br>
# 
# # <font color="#76b900">**Notebook 2:** LLM Services and AI Foundation Models</font>
# 
# <br>
# 
# In this notebook, we will explore LLM services! We'll discuss the reasons for and against deploying LLMs on edge devices alongside ways to deliver powerful models to end users through scalable server deployments like those accessible through the NVIDIA AI Foundation Endpoints.
# 
# <br>
# 
# ### **Learning Objectives:**
# 
# - Understanding the pros and cons of running LLM services locally vs in a scalable cloud environment.
# - Getting familiar with the AI Foundation Model Endpoint schemes, including:
#     - The raw low-level connection interface facilitated by packages like `curl` and `requests`
#     - The abstractions created to make this interface function seamlessly with open-sourced software like LangChain.
# - Getting comfortable with retrieving LLM generations from the pool of endpoints and being able to select a subset of models to build your software on.
# 
# <br>
# 
# ### **Questions To Think About:**
# 
# 1. What kind of model access should you give a person developing an LLM stack, and how does it compare to the access you need to provide to end-users of an AI-powered web application?
# 2. When considering which devices to support, what kinds of rigid assumptions are you making about their local compute resources and what types of fallbacks should you implement?
#     - What if you wanted to deliver a jupyter labs interface with access to a private LLM deployment to customers.
#     - What if now you wanted to support their local jupyter lab environment with your private LLM deployment?
#     - Would anything have to change if you decided to support embedded devices (i.e. Jetson Nano)?
# 3. **[Harder]** Assume you have Stable Diffusion, Mixtral, and Llama-13B deployed on your own compute instance in a cloud environment sharing the same GPU resource. You currently do not have a business use case for Stable Diffusion, but your teams are experimenting with the other two for LLM applications. Should you remove Stable Diffusion from your deployment?
# 
# <br>
# 
# ### **Notebook Source:**
# 
# - This notebook is part of a larger [**NVIDIA Deep Learning Institute**](https://www.nvidia.com/en-us/training/) course titled [**Building RAG Agents with LLMs**](https://www.nvidia.com/en-sg/training/instructor-led-workshops/building-rag-agents-with-llms/). If sharing this material, please give credit and link back to the original course.
# 
# <br>
# 

# ----
# 
# <br>
# 
# ## **Part 1**: Getting Large Models Into Your Environment
# 
# Recall from the last notebook that our current environment has several microservices running on an allocated cloud instance: `docker-router`, `jupyter-notebook-server`, `frontend`, and `llm-service` (among others). 
# 
# - **jupyter-notebook-server**: The service that is running this jupyter labs session and hosting out python environment. 
# - **docker_router**: A service to help us at least observe and monitor our microservices.
# - **frontend**: A live website microservice that delivers us a simple chat interface. 
# 
# This notebook will focus more on the `llm-service` microservice, which you will be using (at least under the hood) to interface with a selection of [**foundation models**](https://www.nvidia.com/en-us/ai-data-science/foundation-models/)! Specifically, you'll be using a subset of the [**NVIDIA AI Foundation Models**](https://catalog.ngc.nvidia.com) to prototype AI-enabled pipelines and orchestrate non-trivial natural-language-backed applications.
# 
# $$---$$
# 
# 
# Across just about every domain, deploying massive deep learning models is a common yet challenging task. Today's models, such as Llama 2 (70B parameters) or ensemble models like Mixtral 7x8B, are products of advanced training methods, vast data resources, and powerful computing systems. Luckily for us, these models have already been trained and many use cases can already be achieved with off-the-shelf solutions. The real hurdle, however, lies in effectively hosting these models.
# 
# **Deployment Scenarios for Large Models:**
# 
# 1. **High-End Datacenter Deployment:**
# > An uncompressed, unquantized model on a data center stack equipped with GPUs like NVIDIA's [A100](https://www.nvidia.com/en-us/data-center/a100/)/[H100](https://www.nvidia.com/en-us/data-center/h100/)/[H200](https://www.nvidia.com/en-us/data-center/h200/) to facilitate fast inference and experimentation.
# > - **Pros**: Ideal for scalable deployment and experimentation, this stack is ideal for either large training workflows or for supporting multiple users or models at the same time.  
# > - **Cons:** It is inefficient to allocate this resource for each user of your service unless the use cases involve model training/fine-tuning or interfacing with lower-level model components.
# 
# 2. **Modest Datacenter/Specialized Consumer Hardware Deployment:**
# > Quantized and further-optimized models can be run (one or two per instance) on more conservative datacenter GPUs such as [L40](https://www.nvidia.com/en-us/data-center/l40/)/[A30](https://www.nvidia.com/en-us/data-center/products/a30-gpu/)/[A10](https://www.nvidia.com/en-us/data-center/products/a10-gpu/) or even on some modern consumer GPUs such as the higher-VRAM [RTX 40-series GPUs](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/).
# > - **Pros:** This setup balances inference speed with manageable limitations for single-user applications. These sessions can also be deployed on a per-user basis to run one or two large models at a time with raw access to model internals (even if they need quantization).
# > - **Cons:** Deploying an instance for each user is still costly at scale, though it may be justifiable for some niche workloads. Alternatively, assuming that users can access these resources in their local environments is likely unreasonable.
# 
# 3. **Consumer Hardware Deployment:**
# > Though heavily limited in ability to propagate data through a neural network, most consumer hardware does have a graphical user interface (GUI), a web browser with internet access, some amount of memory (can safely assume at least 1 GB), and a decently-powerful CPU.
# > - **Cons:** Most hardware at the moment cannot run more than one local large model at a time in any configuration, and running even one model will require significant amounts of resource management and optimizing restrictions.
# > - **Pros:** This is a reasonable and inclusive starting assumption when considering what kinds of users your services should support.
# 

# In this course, your environment will be quite representative of typical consumer hardware; though we can kickstart and prototype with microservices, we are constrained by a CPU-only compute environment that will struggle to run an LLM model. While this is a significant limitation, we will still be able to take advantage of fast LLM capabilities via:
# - Access to a compute-capable service for hosting large models.
# - A streamlined interface for command input and result retrieval.
# 
# With our foundation in microservices and port-based connections, we are well-positioned to explore effective interfacing options for getting LLM access for our development environment!

# ----
# 
# <br>
# 
# ## **Part 2:** Hosted Large Model Services
# 
# In our pursuit to provide access to Large Language Models (LLMs) in a resource-constrained environment like ours, characterized by CPU-only instances, we'll evaluate various hosting options:
# 
# **Black-Box Hosted Models:**
# > Services such as [**OpenAI**](https://openai.com/) offer APIs to interact with black-box models like GPT-4. These powerful, well-integrated services can provide simple interfaces to complex pipelines that automatically track memory, call additional models, and incorporate multimodal interfaces as necessary to simplify typical use scenarios. At the same time, they maintain operational opacity and often lack a straightforward path to self-hosting.
# > - **Pros:** Easy to use out-of-the-box with shallow barriers to entry for an average user.
# > - **Cons:** Black-box deployments suffer from potential privacy concerns, limited customization, and cost implications at scale.
# 
# **Self-Hosted Models:**
# 
# > Behind the scenes of just about all scaled model deployments is one or more giant models running in a data center with scalable resources and lightning-fast bandwidth at their disposal. Though necessary to deploy large models at scale and maintain strong control over the provided interfaces, these systems often require expertise to set up and generally do not work well for supporting non-developer workflows for only one individual at a time. Such systems are much better for supporting many simultaneous users, multiple models, and custom interfaces.
# > - **Pros:** They offer the capability to integrate custom datasets and APIs and are primarily designed to support numerous users concurrently.
# > - **Cons:** These setups demand technical expertise to set up and properly configure.
# 
# To get the best of both worlds, we will utilize the [**NVIDIA NGC Service**](https://www.nvidia.com/en-us/gpu-cloud/). NGC offers a suite of developer tools for designing and deploying AI solutions. Central to our needs are the [NVIDIA AI Foundation Models](https://www.nvidia.com/en-us/ai-data-science/foundation-models/), which are pre-tuned and pre-optimized models designed for easy out-of-the-box scalable deployment (as-is or with further customization). Furthermore, NGC hosts accessible model endpoints for querying live foundation models in a [scalable DGX-accelerated compute environment](https://www.nvidia.com/en-us/data-center/dgx-platform/).

# ----
# 
# <br>
# 
# ## **Part 3:** Getting Started With Hosted Inference

# **When deploying a model for scaled inference, the steps you generally need to take are as follows:**
# - Identify the models you would like users to access, and allocate resources to host them.
# - Figure out what kinds of controls you would like users to have, and expose ways for them to access it.
# - Create monitoring schemes to track/gate usage, and set up systems to scale and throttle as necessary.
# 
# For this course, you'll use the models deployed by NVIDIA, which are hosted as **LLM NIMs.** NIMs are microservices that are optimized to run AI workloads for scaled inference deployment. They work just fine for local inference and offer standardized APIs, but are primarily designed to work especially well in scaled environments. These particular models are deployed on NVIDIA DGX Cloud as shared functions and are advertised through an OpenAPI-style API gateway. Let's unpack what that means:
# 
# **On The Cluster Side:** These microservices are hosted on a Kubernetes-backed platform that scales the load across a minimum and maximum number of DGX Nodes and are delivered behind a single function. In other words:
# - A large-language model is downloaded to and deployed on a **GPU-enabled compute node** (i.e. a powerful CPU and 4xH100-GPU environment which is physically-integrated in a DGX Pod).
# - On start, a selection of these compute nodes are kickstarted such that, whenever a user sends a request to the function, one of those nodes will receive the request.
#     - Kubernetes will route this traffic appropriately. If there is an idle compute node, it will receive the traffic. If all of them are working, the request will be queued up and a node will pick it up as soon as possible.
#     - In our case, these nodes will still pick up requests very fast since in-flight batching is enabled, meaning each node can take in up to 256 active requests at a time as-they-come before they get completely "full". (256 is a hyperparameter on deployment).
# - As load begins to increase, auto-scaling will kick in and more nodes will be kickstarted to avoid request handling delays.
# 
# The following image shows an arbitrary function invocation with a custom (non-OpenAPI) API. This was the initial way in which the public endpoints were advertised, but is now an implementation detail. 

# <!-- > <img style="max-width: 1000px;" src="imgs/ai-playground-api.png" /> -->
# <!-- > <img src="https://drive.google.com/uc?export=view&id=1ckAIZoy7tvtK1uNqzA9eV5RlKMbVqs1-" width=1000px/> -->
# > <img src="https://dli-lms.s3.amazonaws.com/assets/s-fx-15-v1/imgs/ai-playground-api.png" width=800px/>

# **On The Gateway Side:** To make this API more standard, an API gateway server is used to aggregate these functions behind a common API known as OpenAPI. This specification is subscribed to by many including OpenAI, so using the OpenAI client is a valid interface: 
# 
# <!-- > <img style="max-width: 800px;" src="imgs/mixtral_api.png" /> -->
# > <img src="https://dli-lms.s3.amazonaws.com/assets/s-fx-15-v1/imgs/mixtral_api.png" width=800px/>

# For this course, you will want to use a more specialized interface that connects to an LLM orchestration framework called LangChain (more on that later). For on your end, you will be using the more tailored interface like `ChatNVIDIA` from the [`langchain_nvidia_ai_endpoints`](https://python.langchain.com/docs/integrations/chat/nvidia_ai_endpoints/) library. *More on that later.*

# **On The User Side:** Incorporating these endpoints into your client, you can design integrations, pipelines, and user experiences that leverage these generative AI capabilities to endow your applications with reasoning and generative abilities. A popular example of such an application is [**OpenAI's ChatGPT**](https://chat.openai.com/), which is an orchestration of endpoints including GPT4, Dalle, and others. Though it may sometimes look like a single intelligent model, it is merely an aggregation of model endpoints with software engineering to help manage state and context control. This will be reinforced throughout the course, and by the end you should have an idea for how you could go about making a similar chat assistant for an arbitrary use-case. 

# <!-- > <img style="max-width: 700px;" src="imgs/openai_chat.png" /> -->
# > <img src="https://dli-lms.s3.amazonaws.com/assets/s-fx-15-v1/imgs/openai_chat.png" width=700px/>
# 

# ----
# 
# <br>
# 
# ## **Part 4: [Exercise]** Trying Out The Foundation Model Endpoints

# In this section, you will start using the endpoints that will get you through the rest of the course! 
# 
# **From Your Own Environment**: You would want to go to [`build.nvidia.com`](https://build.nvidia.com/) and find a model you're like to use. For example, you could go to [**the MistralAI's Mixtral-8x7b model**](https://build.nvidia.com/mistralai/mixtral-8x7b-instruct) to see an example of how to use the model, links for further readings, and some buttons like "Apply To Self-Host" and "Get API Key."
# 
# - Clicking **"Apply To Self-Host"** will guide you to information about NVIDIA Microservices and give you some avenues to sign up (i.e. early access/NVIDIA AI Enterprise pathway) or enter a notification list (General Access pathway).
# 
# - Clicking **"Get API Key"** will generate an API key starting with "nvapi-" which you can provide to the API endpoints via a network request!
# 
# If you were to do this, you would need to add the API key to the notebook like so:

# In[ ]:


# import os
# os.environ["NVIDIA_API_KEY"] = "nvapi-..."


# <br/>
# 
# **From Your Course Environment**: For the sake of the course, we will be directly using these models and will access the models through a proxy server set up in the `llm_client` directory (namely [**`llm_client/client_server.py`**](llm_client/client_server.py)). The details surrounding its implementation is outside the scope of the course, but will give you unlimited access to a selection of models by: 
# - Exposing some endpoints that will pass your request through to a selection of models.
# - Filling in an API key from within the llm_client microservice so that you don't run our of credits. 
# 
# ***The same code can also be used as useful starting point for implementing your own GenAI gateway service like [`integrate.api.nvidia.com`](https://docs.api.nvidia.com/nim/reference/nvidia-embedding-2b-infer) or [`api.openai.com`](https://platform.openai.com/docs/api-reference/introduction).***
# 
# <br/>

# ### **4.1.** Manual Python Requests
# 
# As we said before, you can interact with microservices or remote APIs using Python's `requests` library, and will generally follow the following process:
# - **Importing Libraries:** We start by importing requests for HTTP requests and json for handling JSON data.
# - **API URL and Headers:** Define the URL of the API endpoint and headers, including authorization (API key) and data format preferences.
# - **Data Payload:** Specify the data you want to send; here, it’s a simple query.
# - **Making the Request:** Use `requests.post` to send a POST request. You can replace post with `get`, `put`, etc., depending on the API's requirements.
# - **Response Handling:** Check the status code to determine if the request was successful (200 means success) and then process the data.
# 
# To establish a bit about the service, we can try to see what kinds of endpoints and models it provides: 

# In[ ]:


import requests

invoke_url = "http://llm_client:9000"
headers = {"content-type": "application/json"}

requests.get(invoke_url, headers=headers, stream=False).json()


# In[ ]:


import requests

invoke_url = "http://llm_client:9000/v1/models"
# invoke_url = "https://api.openai.com/v1/models"
# invoke_url = "https://integrate.api.nvidia.com/v1"
# invoke_url = "http://llm_client:9000/v1/models/mistralai/mixtral-8x7b-instruct-v0.1"
# invoke_url = "http://llm_client:9000/v1/models/mistralaimixtral-8x7b-instruct-v0.1"
headers = {
    "content-type": "application/json",
    # "Authorization": f"Bearer {os.environ.get('NVIDIA_API_KEY')}",
    # "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
}

print("Available Models:")
response = requests.get(invoke_url, headers=headers, stream=False)
# print(response.json())  ## <- Raw Response. Very Verbose
for model_entry in response.json().get("data", []):
    print(" -", model_entry.get("id"))

print("\nExample Entry:")
invoke_url = "http://llm_client:9000/v1/models/mistralai/mixtral-8x7b-instruct-v0.1"
requests.get(invoke_url, headers=headers, stream=False).json()


# <br/>
# 
# We will not be operating much on this level of abstraction for this course, but it's worth going through the basic process to confirm that, yes, these requests are coming through our microservice in pretty much the same way as if the server were hosted remotely. You can assume for the remainder of the course that an interaction like the following is taking place under the hood of your clients.

# In[ ]:


from getpass import getpass
import os

## Where are you sending your requests?
invoke_url = "http://llm_client:9000/v1/chat/completions"

## If you wanted to use your own API Key, it's very similar
# if not os.environ.get("NVIDIA_API_KEY", "").startswith("nvapi-"):
#     os.environ["NVIDIA_API_KEY"] = getpass("NVIDIA_API_KEY: ")
# invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"

## If you wanted to use OpenAI, it's very similar
# if not os.environ.get("OPENAI_API_KEY", "").startswith("sk-"):
#     os.environ["OPENAI_API_KEY"] = getpass("OPENAI_API_KEY: ")
# invoke_url = "https://api.openai.com/v1/models"

## Meta communication-level info about who you are, what you want, etc.
headers = {
    "accept": "text/event-stream",
    "content-type": "application/json",
    # "Authorization": f"Bearer {os.environ.get('NVIDIA_API_KEY')}",
    # "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
}

## Arguments to your server function
payload = {
    "model": "mistralai/mixtral-8x7b-instruct-v0.1",
    "messages": [{"role":"user","content":"Tell me hello in French"}],
    "temperature": 0.5,   
    "top_p": 1,
    "max_tokens": 1024,
    "stream": True                
}


# In[ ]:


import requests
import json

## Use requests.post to send the header (streaming meta-info) the payload to the endpoint
## Make sure streaming is enabled, and expect the response to have an iter_lines response.
response = requests.post(invoke_url, headers=headers, json=payload, stream=True)

## If your response is an error message, this will raise an exception in Python
try: 
    response.raise_for_status()  ## If not 200 or similar, this will raise an exception
except Exception as e:
    # print(response.json())
    print(response.json())
    raise e

## Custom utility to make live a bit easier
def get_stream_token(entry: bytes):
    """Utility: Coerces out ['choices'][0]['delta'][content] from the bytestream"""
    if not entry: return ""
    entry = entry.decode('utf-8')
    if entry.startswith('data: '):
        try: entry = json.loads(entry[5:])
        except ValueError: return ""
    return entry.get('choices', [{}])[0].get('delta', {}).get('content')

## If the post request is honored, you should be able to iterate over 
for line in response.iter_lines():
    
    ## Without Processing: data: {"id":"...", ... "choices":[{"index":0,"delta":{"role":"assistant","content":""}...}...
    # if line: print(line.decode("utf-8"))

    ## With Processing: An actual stream of tokens printed one-after-the-other as they come in
    print(get_stream_token(line), end="")


# <br>
# 
# #### **[NOTES]**
# 
# **You may notice that the chat models expect "messages" as input:**
# 
# This may be unexpected if you're more used to raw LLM interfaces like those of local HuggingFace models, but it will look pretty standard to users of OpenAI models. By enforcing a restricted interface instead of a raw text completion one, the service can have more control over what the users can do. There are plenty of pros and cons to this interface, with some noteworthy ones below:
# - A service might restrict the use of a specific role type or parameter (i.e. system message restriction, priming message to get arbitrary generation, etc).
# - A service might enforce custom prompt formats and implement extra options under the hood that rely on the chat interface.
# - A service might use stronger assumptions to implement deeper optimizations in the inference pipeline.
# - A service might mimic another popular interface to leverage existing ecosystem compatibilities.
# 
# All of these are valid reasons, and it's important to consider which interface options are best for your particular use cases when choosing or deploying your own service.
# 
# **You may notice that there are two fundamental ways of querying the models:**
# 
# You can **invoke without streaming**, in which case the service response will come all at once after it has been computed in full. This is great when you need the entire output of the model before doing anything else; for example, when you want to print out the whole result or use it for downstream tasks. The response body will look something like this:
# 
# ```json
# {
#     "id": "d34d436a-c28b-4451-aa9c-02eed2141ed3",
#     "choices": [{
#         "index": 0,
#         "message": { "role": "assistant", "content": "Bonjour! ..." },
#         "finish_reason": "stop"
#     }],
#     "usage": {
#         "completion_tokens": 450,
#         "prompt_tokens": 152,
#         "total_tokens": 602
#     }
# }
# ```
# 
# You can also **invoke with streaming**, in which case the service will send out a series of requests until a final request is sent out. This is great when you can use the responses of the service as it becomes available (which is very good for language model components that print the output directly to the user as it gets generated). In this case, the response body will look a lot more like this:
# 
# ```json
# data:{"id":"...","choices":[{"index":0,"delta":{"role":"assistant","content":"Bon"},"finish_reason":null}]}
# data:{"id":"...","choices":[{"index":0,"delta":{"role":"assistant","content":"j"},"finish_reason":null}]}
# ...
# data:{"id":"...","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":"stop"}]}
# data:[DONE]
# ```
# 
# Both of these options can be done with relative ease using Python's `requests` library, but using the interface as-is will result in a lot of repetitive code. Luckily, we have some systems that make this significantly easier to use and incorporate into larger projects!

# <br/>
# 
# ### **4.2.** OpenAI Client Request
# 
# It's good to know that this interface exists, but using it as-is will result in a lot of repetitive code and extra complexity. Luckily, we have some systems that make this significantly easier to use and incorporate into larger projects! One layer of abstraction above using the requests is to use a more opinionated client like that of OpenAI. Since both NVIDIA and OpenAI subscribe to the OpenAPI specification, we can borrow their client instead. Note that under the hood, the same processes are still being done, probably facilitated by a lower-level client like that of [**httpx**](https://github.com/projectdiscovery/httpx) or [**aiohttp**](https://github.com/aio-libs/aiohttp). 

# In[ ]:


## Using General OpenAI Client
from openai import OpenAI

# client = OpenAI()  ## Assumes OPENAI_API_KEY is set

# client = OpenAI(
#     base_url = "https://integrate.api.nvidia.com/v1",
#     api_key = os.environ.get("NVIDIA_API_KEY", "")
# )

client = OpenAI(
    base_url = "http://llm_client:9000/v1",
    api_key = "I don't have one"
)

completion = client.chat.completions.create(
    model="mistralai/mixtral-8x7b-instruct-v0.1",
    # model="gpt-4-turbo-2024-04-09",
    messages=[{"role":"user","content":"Hello World"}],
    temperature=1,
    top_p=1,
    max_tokens=1024,
    stream=True,
)

## Streaming with Generator: Results come out as they're generated
for chunk in completion:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")


# In[ ]:


## Non-Streaming: Results come from server when they're all ready
completion = client.chat.completions.create(
    model="mistralai/mixtral-8x7b-instruct-v0.1",
    # model="gpt-4-turbo-2024-04-09",
    messages=[{"role":"user","content":"Hello World"}],
    temperature=1,
    top_p=1,
    max_tokens=1024,
    stream=False,
)

completion


# <br/>
# 
# ### **4.3.** ChatNVIDIA Client Request
# 
# So far, we've seen communication happen on two layers of abstraction: **raw requests** and **API client**. In this course, we will want to do LLM orchestration with a framework called LangChain, so we'll need to go one layer of abstraction higher to a **Framework Connector**.
# 
# The goal of a **connector** is to convert an arbitrary API from its native core into one that a target code-base would expect. In this course, we'll want to take advantage of LangChain's thriving chain-centric ecosystem, but the raw `requests` API will not take us all the way there. Under the hood, every LangChain chat model that isn't hosted locally has to rely on such an API, but the developer-facing API is a much cleaner [`LLM` or `SimpleChatModel`-style interface](https://python.langchain.com/docs/modules/model_io/) with default parameters and some simple utility functions like `invoke` and `stream`.
# 
# To start off our exploration into the LangChain interface, we will use the `ChatNVIDIA` connector to interface with our `chat/completions` endpoints. This model is part of the LangChain extended ecosystem and can be installed locally via `pip install langchain-nvidia-ai-endpoints`.

# In[ ]:


## Using ChatNVIDIA
from langchain_nvidia_ai_endpoints import ChatNVIDIA

## NVIDIA_API_KEY pulled from environment
llm = ChatNVIDIA(model="mistralai/mixtral-8x7b-instruct-v0.1")
# llm = ChatNVIDIA(model="mistralai/mixtral-8x7b-instruct-v0.1", mode="open", base_url="http://llm_client:9000/v1")
llm.invoke("Hello World")


# In[ ]:


llm.client.last_inputs


# In[ ]:


# llm.client.last_response
llm.client.last_response.json()


# <br/>
# 
# #### **[NOTES]**
# 
# - **The course uses a modified fork of the `ai-endpoints` connector with several features which are more useful for our course environment.** These features are not yet in the main version and are being proactively incorporated alongside other developments and requirements from the [**LlamaIndex**](https://docs.llamaindex.ai/en/stable/examples/embeddings/nvidia/) and [**Haystack**](https://docs.haystack.deepset.ai/docs/nvidiagenerator) counterparts. 
# 
# - **ChatNVIDIA is defaulting the `llm_client` microservice because we set some environment variables to make it happen**: 

# In[ ]:


import os

{k:v for k,v in os.environ.items() if k.startswith("NVIDIA_")}
## Translation: Use the base_url of llm_client:9000 for the requests,
## and use "open"api-spec access for model discovery and url formats


# <br/>
# 
# **Throughout the course, feel free to try out the model of your choice.** Below are the selection of models provided as part of this course, out of which a selection should be working at any given time.

# In[ ]:


model_list = ChatNVIDIA.get_available_models(list_none=False)

for model_card in model_list:
    model_name = model_card.id
    llm = ChatNVIDIA(model=model_name)
    print(f"TRIAL: {model_name}")
    try: 
        for token in llm.stream("Tell me about yourself! 2 sentences.", max_tokens=100):
            print(token.content, end="")
    except Exception as e: 
        print(f"EXCEPTION: {e}")
    print("\n\n" + "="*84)

