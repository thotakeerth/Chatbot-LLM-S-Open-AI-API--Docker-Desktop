#!/usr/bin/env python
# coding: utf-8

# <br>
# 
# # <font color="#76b900">**Notebook 3:** LangChain Expression Language</font>
# 
# <br>
# 
# In the previous notebook, we introduced some of the services we'll be taking advantage of for our LLM applications, including some external LLM platforms and a locally-hosted front-end service. Both of these components incorporate LangChain, but it hasn't been spotlighted as a major focus yet. It is expected that you have some experience with LangChain and LLMs, but this notebook is intended catch you up to the level necessary to progress through the rest of the course!
# 
# This notebook is designed to guide you through the integration and application of LangChain, a leading orchestration library for Large Language Models (LLMs), with the AI Foundation Endpoints from last time. Whether you are a seasoned developer or new to LLMs, this course will enhance your understanding and skills in building sophisticated LLM applications.
# 
# <br>
# 
# ### **Learning Objectives:**
# 
# - Learning how to leverage chains and runnables to orchestrate interesting LLM systems.  
# - Getting familiar with using LLMs for external conversation and internal reasoning.
# - Be able to start up and run a simple Gradio interface inside your notebook.
# 
# <br>
# 
# ### **Questions To Think About:**
# 
# - What kinds of utilities might be necessary to keep information flowing through the pipeline **(primer for next notebook)**.
# - When you encounter the `gradio` interface, consider where all you have seen this style of interface before. Some possible places may include [HuggingFace Spaces](https://huggingface.co/spaces)...
# - Near the end of the section, you'll learn that you can pass around chains as routes and access them across environments via ports. What kinds of requirements should you advertise if you are trying to receive chains from other microservices?
# 
# <br>
# 
# ### **Notebook Source:**
# 
# - This notebook is part of a larger [**NVIDIA Deep Learning Institute**](https://www.nvidia.com/en-us/training/) course titled [**Building RAG Agents with LLMs**](https://www.nvidia.com/en-sg/training/instructor-led-workshops/building-rag-agents-with-llms/). If sharing this material, please give credit and link back to the original course.
# 
# <br>
# 
# 
# ### **Environment Setup:**

# In[6]:


## Necessary for Colab, not necessary for course environment
get_ipython().run_line_magic('pip', 'install -q langchain langchain-nvidia-ai-endpoints gradio')

import os
os.environ["NVIDIA_API_KEY"] = "nvapi-..."

## If you encounter a typing-extensions issue, restart your runtime and try again
from langchain_nvidia_ai_endpoints import ChatNVIDIA
ChatNVIDIA.get_available_models()


# -----
# 
# <br>
# 
# ## **Part 6:** Wrap-Up
# 
# The goal of this notebook was to onboard you into the LangChain Expression Language scheme as well as provide exposure to `gradio` and `LangServe` interfaces for serving LLM functionality! There will be more of this in the subsequent notebook, but this notebook pushes towards intermediate and emerging paradigms in LLM agent development.
# 
# ### <font color="#76b900">**Great Job!**</font>
# 
# ### **Next Steps:**
# 1. **[Optional]** Take a few minutes to look over the `frontend` directory for the deployment recipe and underlying functionality.
# 2. **[Optional]** Revisit the **"Questions To Think About" Section** at the top of the notebook and think about some possible answers.
# 
# ---

# <br>
# 
# ### **Considering Your Models**
# 
# Going back to the [**NGC Catalog**](https://catalog.ngc.nvidia.com/ai-foundation-models), we'll be able to find a selection of interesting models that you can invoke from your environment. These models are all there because there is valid use for them in production pipelines, so it's a good idea to look around and find out which models are best for your use cases.
# 
# **The code provided includes some models already listed, but you may want (or need) to upgrade to other models if you notice a strictly-better option or a model is no longer available. *This comment will apply throughout the rest of the course, so keep that in mind!***

# ----
# 
# <br>
# 
# ## **Part 1:** What Is LangChain?

# LangChain is an popular LLM orchestration library to help set up systems that have one or more LLM components. The library is, for better or worse, extremely popular and changes rapidly based on new developments in the field, meaning that somebody can have a lot of experience in some parts of LangChain while having little-to-no familiarity with other parts (either because there are just so many different features or the area is new and the features have only recently been implemented).
# 
# This notebook will be using the **LangChain Expression Language (LCEL)** to ramp up from basic chain specification to more advanced dialog management practices, so hopefully the journey will be enjoyable and even seasoned LangChain developers might learn something new!

# <!-- > <img style="max-width: 400px;" src="imgs/langchain-diagram.png" /> -->
# > <img src="https://dli-lms.s3.amazonaws.com/assets/s-fx-15-v1/imgs/langchain-diagram.png" width=400px/>
# <!-- > <img src="https://drive.google.com/uc?export=view&id=1NS7dmLf5ql04o5CyPZnd1gnXXgO8-jbR" width=400px/> -->

# ----
# 
# <br>
# 
# ## **Part 2:** Chains and Runnables
# 
# When exploring a new library, it's important to note what are the core systems of the library and how are they used.
# 
# In LangChain, the main building block *used to be* the classic **Chain**: a small module of functionality that does something specific and can be linked up with other chains to make a system. So for all intents and purposes, it is a "building-block system" abstraction where the building blocks are easy to create, have consistent methods (`invoke`, `generate`, `stream`, etc), and can be linked up to work together as a system. Some example legacy chains include `LLMChain`, `ConversationChain`, `TransformationChain`, `SequentialChain`, etc.
# 
# More recently, a new recommended specification has emerged that is significantly easier to work with and extremely compact, the **LangChain Expression Language (LCEL)**. This new format relies on a different kind of primitive - a **Runnable** - which is simply an object that wraps a function. Allow dictionaries to be implicitly converted to Runnables and let a **pipe |** operator create a Runnable that passes data from the left to the right (i.e. `fn1 | fn2` is a Runnable), and you have a simple way to specify complex logic!
# 
# Here are some very representative example Runnables, created via the `RunnableLambda` class:

# In[2]:


from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from functools import partial

################################################################################
## Very simple "take input and return it"
identity = RunnableLambda(lambda x: x)  ## Or RunnablePassthrough works

################################################################################
## Given an arbitrary function, you can make a runnable with it
def print_and_return(x, preface=""):
    print(f"{preface}{x}")
    return x

rprint0 = RunnableLambda(print_and_return)

################################################################################
## You can also pre-fill some of values using functools.partial
rprint1 = RunnableLambda(partial(print_and_return, preface="1: "))

################################################################################
## And you can use the same idea to make your own custom Runnable generator
def RPrint(preface=""):
    return RunnableLambda(partial(print_and_return, preface=preface))

################################################################################
## Chaining two runnables
chain1 = identity | rprint0
chain1.invoke("Hello World!")
print()

################################################################################
## Chaining that one in as well
output = (
    chain1           ## Prints "Welcome Home!" & passes "Welcome Home!" onward
    | rprint1        ## Prints "1: Welcome Home!" & passes "Welcome Home!" onward
    | RPrint("2: ")  ## Prints "2: Welcome Home!" & passes "Welcome Home!" onward
).invoke("Welcome Home!")

## Final Output Is Preserved As "Welcome Home!"
print("\nOutput:", output)


# ----
# 
# <br>
# 
# ## **Part 3:** Dictionary Pipelines with Chat Models
# 
# There's a lot you can do with runnables, but it's important to formalize some best practices. At the moment, it's easiest to use *dictionaries* as our default variable containers for a few key reasons:
# 
# **Passing dictionaries helps us keep track of our variables by name.**
# 
# Since dictionaries allow us to propagate named variables (values referenced by keys), using them is great for locking in our chain components' outputs and expectations.
# 
# **LangChain prompts expect dictionaries of values.**
# 
# It's quite intuitive to specify an LLM Chain in LCEL to take in a dictionary and produce a string, and equally easy to raise said string back up to be a dictionary. This is very intentional and is partially due to the above reason. 

# <br>
# 
# ### **Example 1:** A Simple LLM Chain
# 
# One of the most fundamental components of classical LangChain is the `LLMChain` that accepts a **prompt** and an **LLM**:
# 
# - A prompt, usually retrieved from a call like `PromptTemplate.from_template("string with {key1} and {key2}")`, specifies a template for creating a string as output. A dictionary `{"key1" : 1, "key2" : 2}` could be passed in to get the output `"string with 1 and 2"`.
#     - For chat models like `ChatNVIDIA`, you would use `ChatPromptTemplate.from_messages` instead.
# - An LLM takes in a string and returns a generated string.
#     - Chat models like `ChatNVIDIA` work with messages instead, but it's the same idea! Using an **StrOutputParser** at the end will extract the content from the message.
# 
# The following is a lightweight example of a simple chat chain as described above. All it does is take in an input dictionary and use it fill in a system message to specify the overall meta-objective and a user input to specify query the model.

# In[4]:


from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

## Simple Chat Pipeline
chat_llm = ChatNVIDIA(model="meta/llama3-70b-instruct")

prompt = ChatPromptTemplate.from_messages([
    ("system", "Only respond in rhymes"),
    ("user", "{input}")
])

rhyme_chain = prompt | chat_llm | StrOutputParser()

print(rhyme_chain.invoke({"input" : "Tell me about humans!"}))


# <br>
# 
# In addition to just using the code command as-is, we can try using a [**Gradio interface**](https://www.gradio.app/guides/creating-a-chatbot-fast) to play around with our model. Gradio is a popular tool that provides simple building blocks for creating custom generative AI interfaces! The below example shows how you can make an easy gradio chat interface with this particular example chain:

# In[ ]:


import gradio as gr

#######################################################
## Non-streaming Interface like that shown above

def rhyme_chat(message, history):
    return rhyme_chain.invoke({"input" : message})
    gr.ChatInterface(rhyme_chat).launch()

#######################################################
## Streaming Interface

def rhyme_chat_stream(message, history):
    ## This is a generator function, where each call will yield the next entry
    buffer = ""
    for token in rhyme_chain.stream({"input" : message}):
        buffer += token
        yield buffer

## Uncomment when you're ready to try this.
demo = gr.ChatInterface(rhyme_chat_stream).queue()
window_kwargs = {} # or {"server_name": "0.0.0.0", "root_path": "/7860/"}
demo.launch(share=True, debug=True, **window_kwargs) 

## IMPORTANT!! When you're done, please click the Square button (twice to be safe) to stop the session.


# <br>
# 
# ### **Example 2: Internal Response**
# 
# Sometimes, you also want to have some quick reasoning that goes on behind the scenes before your response actually comes out to the user. When performing this task, you need a model with a strong instruction-following prior assumption built-in.
# 
# The following is an example "zero-shot classification" pipeline which will try to categorize a sentence into one of a couple of classes.
# 
# **In order, this zero-shot classification chain:**
# - Takes in a dictionary with two required keys, `input` and `options`.
# - Passes it through the zero-shot prompt to get the input to our LLM.
# - Passes that string to the model to get the result.
# 
# **Task:** Pick out several models that you think would be good for this kind of task and see how well they perform! Specifically:
# - **Try to find models that are predictable across multiple examples.** If the format is always easy to parse and extremely predictable, then the model is probably ok.
# - **Try to find models that are also fast!** This is important because internal reasoning generally happens behind the hood before the external response gets generated. Thereby, it is a blocking process which can slow down start of "user-facing" generation, making your system feel sluggish.

# In[ ]:


from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

## TODO: Try out some more models and see if there are better options
instruct_llm = ChatNVIDIA(model="mistralai/mistral-7b-instruct-v0.2")

sys_msg = (
    "Choose the most likely topic classification given the sentence as context."
    " Only one word, no explanation.\n[Options : {options}]"
)

## One-shot classification prompt with heavy format assumptions.
zsc_prompt = ChatPromptTemplate.from_messages([
    ("system", sys_msg),
    ("user", "[[The sea is awesome]]"),
    ("assistant", "boat"),
    ("user", "[[{input}]]"),
])

## Roughly equivalent as above for <s>[INST]instruction[/INST]response</s> format
zsc_prompt = ChatPromptTemplate.from_template(
    f"{sys_msg}\n\n"
    "[[The sea is awesome]][/INST]boat</s><s>[INST]"
    "[[{input}]]"
)

zsc_chain = zsc_prompt | instruct_llm | StrOutputParser()

def zsc_call(input, options=["car", "boat", "airplane", "bike"]):
    return zsc_chain.invoke({"input" : input, "options" : options}).split()[0]

print("-" * 80)
print(zsc_call("Should I take the next exit, or keep going to the next one?"))

print("-" * 80)
print(zsc_call("I get seasick, so I think I'll pass on the trip"))

print("-" * 80)
print(zsc_call("I'm scared of heights, so flying probably isn't for me"))


# <br>
# 
# ### **Example 3: Multi-Component Chains**
# 
# The previous example showed how we can coerce a dictionary into a string by passing it through a `prompt -> LLM` chain, so that's one easy structure to motivate the container choice. But is it just as easy to convert the string output back up to a dictionary?
# 
# **Yes, it is!** The simplest way is actually to use the LCEL *"implicit runnable"* syntax, which allows you to use a dictionary of functions (including chains) as a runnable that runs each function and maps the value to the key in the output dictionary.
# 
# The following is an example which exercises these utilities while also providing a few extra tools you may find useful in practice.

# In[ ]:


from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from functools import partial

################################################################################
## Example of dictionary enforcement methods
def make_dictionary(v, key):
    if isinstance(v, dict):
        return v
    return {key : v}

def RInput(key='input'):
    '''Coercing method to mold a value (i.e. string) to in-like dict'''
    return RunnableLambda(partial(make_dictionary, key=key))

def ROutput(key='output'):
    '''Coercing method to mold a value (i.e. string) to out-like dict'''
    return RunnableLambda(partial(make_dictionary, key=key))

def RPrint(preface=""):
    return RunnableLambda(partial(print_and_return, preface=preface))

################################################################################
## Common LCEL utility for pulling values from dictionaries
from operator import itemgetter

up_and_down = (
    RPrint("A: ")
    ## Custom ensure-dictionary process
    | RInput()
    | RPrint("B: ")
    ## Pull-values-from-dictionary utility
    | itemgetter("input")
    | RPrint("C: ")
    ## Anything-in Dictionary-out implicit map
    | {
        'word1' : (lambda x : x.split()[0]),
        'word2' : (lambda x : x.split()[1]),
        'words' : (lambda x: x),  ## <- == to RunnablePassthrough()
    }
    | RPrint("D: ")
    | itemgetter("word1")
    | RPrint("E: ")
    ## Anything-in anything-out lambda application
    | RunnableLambda(lambda x: x.upper())
    | RPrint("F: ")
    ## Custom ensure-dictionary process
    | ROutput()
)

up_and_down.invoke({"input" : "Hello World"})


# In[ ]:


## NOTE how the dictionary enforcement methods make it easy to make the following syntax equivalent
up_and_down.invoke("Hello World")


# ----
# 
# <br>
# 
# ## **Part 4: [Exercise]** Rhyme Re-themer Chatbot
# 
# Below is a poetry generation example that showcases how you might organize two different tasks under the guise of a single agent. The system calls back to the simple Gradio example, but extends it with some boiler-plate responses and logic behind the scenes.
# 
# It's primary feature is as follows:
# - On the first response, it will generate a poem based on your response.
# - On subsequent responses, it will keep the format and structure of your original rhyme while modifying the topic of the poem.
# 
# **Problem:** At present, the system should function just fine for the first part, but the second part is not yet implemented.
# 
# **Objective:** Implement the rest of the `rhyme_chat2_stream` method such that the agent is able to function normally.
# 
# To make the gradio component easier to reason with, a simplified `queue_fake_streaming_gradio` method is provided that will simulate the gradio chat event loop with the standard Python `input` method

# In[ ]:


ChatNVIDIA.get_available_models(filter="mistralai/", list_none=False)


# In[ ]:


from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from copy import deepcopy

instruct_llm = ChatNVIDIA(model="mistralai/mixtral-8x22b-instruct-v0.1")  ## Feel free to change the models

prompt1 = ChatPromptTemplate.from_messages([("user", (
    "INSTRUCTION: Only respond in rhymes"
    "\n\nPROMPT: {input}"
))])

prompt2 =  ChatPromptTemplate.from_messages([("user", (
    "INSTRUCTION: Only responding in rhyme, change the topic of the input poem to be about {topic}!"
    " Make it happy! Try to keep the same sentence structure, but make sure it's easy to recite!"
    " Try not to rhyme a word with itself."
    "\n\nOriginal Poem: {input}"
    "\n\nNew Topic: {topic}"
))])

## These are the main chains, constructed here as modules of functionality.
chain1 = prompt1 | instruct_llm | StrOutputParser()  ## only expects input
chain2 = prompt2 | instruct_llm | StrOutputParser()  ## expects both input and topic

################################################################################
## SUMMARY OF TASK: chain1 currently gets invoked for the first input.
##  Please invoke chain2 for subsequent invocations.

def rhyme_chat2_stream(message, history, return_buffer=True):
    '''This is a generator function, where each call will yield the next entry'''

    first_poem = None
    for entry in history:
        if entry[0] and entry[1]:
            ## If a generation occurred as a direct result of a user input,
            ##  keep that response (the first poem generated) and break out
            first_poem = "\n\n".join(entry[1].split("\n\n")[1:-1])
            break

    if first_poem is None:
        ## First Case: There is no initial poem generated. Better make one up!

        buffer = "Oh! I can make a wonderful poem about that! Let me think!\n\n"
        yield buffer

        ## iterate over stream generator for first generation
        inst_out = ""
        chat_gen = chain1.stream({"input" : message})
        for token in chat_gen:
            inst_out += token
            buffer += token
            yield buffer if return_buffer else token

        passage = "\n\nNow let me rewrite it with a different focus! What should the new focus be?"
        buffer += passage
        yield buffer if return_buffer else passage

    else:
        ## Subsequent Cases: There is a poem to start with. Generate a similar one with a new topic!

        yield f"Not Implemented!!!"; return ## <- TODO: Comment this out
        
        ########################################################################
        ## TODO: Invoke the second chain to generate the new rhymes.

        # buffer = f"Sure! Here you go!\n\n" ## <- TODO: Uncomment these lines
        # yield buffer
        
        ## iterate over stream generator for second generation

        ## END TODO
        ########################################################################

        passage = "\n\nThis is fun! Give me another topic!"
        buffer += passage
        yield buffer if return_buffer else passage

################################################################################
## Below: This is a small-scale simulation of the gradio routine.

def queue_fake_streaming_gradio(chat_stream, history = [], max_questions=3):

    ## Mimic of the gradio initialization routine, where a set of starter messages can be printed off
    for human_msg, agent_msg in history:
        if human_msg: print("\n[ Human ]:", human_msg)
        if agent_msg: print("\n[ Agent ]:", agent_msg)

    ## Mimic of the gradio loop with an initial message from the agent.
    for _ in range(max_questions):
        message = input("\n[ Human ]: ")
        print("\n[ Agent ]: ")
        history_entry = [message, ""]
        for token in chat_stream(message, history, return_buffer=False):
            print(token, end='')
            history_entry[1] += token
        history += [history_entry]
        print("\n")

## history is of format [[User response 0, Bot response 0], ...]
history = [[None, "Let me help you make a poem! What would you like for me to write?"]]

## Simulating the queueing of a streaming gradio interface, using python input
queue_fake_streaming_gradio(
    chat_stream = rhyme_chat2_stream,
    history = history
)


# In[ ]:


## Simple way to initialize history for the ChatInterface
chatbot = gr.Chatbot(value = [[None, "Let me help you make a poem! What would you like for me to write?"]])

## IF USING COLAB: Share=False is faster
gr.ChatInterface(rhyme_chat2_stream, chatbot=chatbot).queue().launch(debug=True, share=True)


# ----
# 
# <br>
# 
# ## **Part 5: [Exercise]** Using Deeper LangChain Integrations
# 
# This exercise that gives you an opportunity to investigate some example code regarding [**LangServe**](https://www.langchain.com/langserve). Specifically, we refer to the [**`frontend`**](frontend) directory as well as the [**`35_langserve.ipynb`**](35_langserve.ipynb) notebook.
# 
# - Visit [**`35_langserve.ipynb`**](35_langserve.ipynb) and run the provided script to start up a server with several active routes.
# - Once done, verify that the following [**LangServe `RemoteRunnable`**](https://python.langchain.com/docs/langserve) works. The goal of a [**`RemoteRunnable`**](https://python.langchain.com/docs/langserve) is to make it easy to host a LangChain chain as an API endpoint, so the following is just a test to make sure that it works.
#     - If it doesn't work the first time, there may be an order-of-operations issue. Feel free to try and restart the langserve notebook.
#  
# **After these steps are done, the following type of connection will become accessible from an arbitrary notebook in the course:**

# In[ ]:


from langserve import RemoteRunnable
from langchain_core.output_parsers import StrOutputParser

llm = RemoteRunnable("http://0.0.0.0:9012/basic_chat/") | StrOutputParser()
for token in llm.stream("Hello World! How is it going?"):
    print(token, end='')


# <br>
# 
# Among the active users of this endpoint is the `frontend`, which makes reference to it in its [**`frontend_server.py`**](./frontend/frontend_server.py) implementation:
# 
# ```python
# ## Necessary Endpoints
# chains_dict = {
#     'basic' : RemoteRunnable("http://lab:9012/basic_chat/"),
#     'retriever' : RemoteRunnable("http://lab:9012/retriever/"),  ## For the final assessment
#     'generator' : RemoteRunnable("http://lab:9012/generator/"),  ## For the final assessment
# }
# 
# basic_chain = chains_dict['basic']
# 
# ## Retrieval-Augmented Generation Chain
# 
# retrieval_chain = (
#     {'input' : (lambda x: x)}
#     | RunnableAssign(
#         {'context' : itemgetter('input') 
#         | chains_dict['retriever'] 
#         | LongContextReorder().transform_documents
#         | docs2str
#     })
# )
# 
# output_chain = RunnableAssign({"output" : chains_dict['generator'] }) | output_puller
# rag_chain = retrieval_chain | output_chain
# ```
# 
# As a result, deploying the '/basic_chat' chain should implement the **"Basic"** chat feature in the frontend interface. As a reminder, you can access the frontend via the following generated link: 

# In[ ]:


get_ipython().run_cell_magic('js', '', 'var url = \'http://\'+window.location.host+\':8090\';\nelement.innerHTML = \'<a style="color:green;" target="_blank" href=\'+url+\'><h1>< Link To Gradio Frontend ></h1></a>\';\n')


# **You will be revisiting this idea when you start working on the assessment.**

# -----
#     
# **Note:** This strategy for deploying and relying on LangServe APIs within this type of environment is very non-standard and is made specifically to give students some interesting code to look at. More stable configurations are achievable with optimized single-function containers, and can be found in [**the NVIDIA/GenerativeAIExamples GitHub repository.**](https://github.com/NVIDIA/GenerativeAIExamples/tree/main/RetrievalAugmentedGeneration)

# In[ ]:




