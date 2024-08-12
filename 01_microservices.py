#!/usr/bin/env python
# coding: utf-8

# <br>
# 
# # <font color="#76b900">**Notebook 1:** The Course Environment</font>
# 
# <br>
# 
# In this module, we'll take some time to introduce the course environment, learning about some of the setup requirements, workflows, and considerations.
# 
# **NOTE:** This notebook, though accessible through **Google Colab**, strongly relies on the ***DLI Course Environment*** to run all of the cells properly. However, since there are no todos in this section and it is mainly intended as a vehicle for understanding what's going on behind the scenes, the experience of just reading through this notebook will not compromise much of the experience. For this reason, we have left the cell outputs from our run included by default.
# 
# **Recommendation:** It is a good idea to pull up and familiarize yourself with the course environment for a bit, but this is optional. Feel free to wait until later. **When you're not using the environment, we'd recommend shutting down your session!**
# 
# <br>
# 
# ### **Learning Objectives:**
# 
# - Learn about the course environment through the lens of how it was created and why it was organized this way.
# - Understand how to use the Jupyter Labs interface to interact with the surrounding microservices using active network ports.
# 
# <br>
# 
# ### **Questions To Think About:**
# 
# 1. What kinds of resources do you expect an environment for this course to have, and how would it be different from your local compute environment?
# 2. How different would things be if one of your microservices were running on another host environment (publicly-accessible or gated)?
#     - **Same Idea, Different Question**: How hard would it be to mimic the functionality of a local microservice despite being served by a remote host, and are there any inherent drawbacks of doing this?
# 3. What kinds of microservices do you actually need to spin up on a per-user basis, and what kinds are better left to run persistently?
# 
# <br>
# 
# ### **Notebook Source:**
# 
# - This notebook is part of a larger [**NVIDIA Deep Learning Institute**](https://www.nvidia.com/en-us/training/) course titled [**Building RAG Agents with LLMs**](https://www.nvidia.com/en-sg/training/instructor-led-workshops/building-rag-agents-with-llms/). If sharing this material, please give credit and link back to the original course.
# 
# <br>
# 

# <br>
# 
# ## **Welcome To Your Cloud Environment**
# 
# This is a Jupyter Labs environment that you can use to work on the course content. In most courses, this environment will be a given interface with all the necessary components already running in the background. To help motivate further exploration, this course will also use it as a gateway to understanding microservices orchestration - especially for applications centered around **Large Language Models (LLMs)**. Let's start by exploring the key components of your cloud session.
# 

# <!-- <img src="https://drive.google.com/uc?export=view&id=11MGA5fkwA1XQAglQYQbOgjGTImO3TkLS" width=800/> -->
# <!-- <img src="imgs/simple-env.png" width=800/> -->
# > <img src="https://dli-lms.s3.amazonaws.com/assets/s-fx-15-v1/imgs/simple-env.png" width=800px/>

# ----
# 
# <br>
# 
# ## **Part 1:** Hosting Containers
# 
# When you access this Jupyter Notebook, an instance on a cloud platform like AWS or Azure is allocated to you by NVIDIA DLI (Deep Learning Institute). This cloud instance forms your base cloud environment and includes:
# 
# - A dedicated CPU, and possibly a GPU, for processing.
# - A pre-installed base operating system.
# - Some exposed ports which can be accessed via a known web address.
# 
# Though this gives you all the necessary resources to get started, it is essentially just a blank canvas by default. If we wanted, we could run some baked routines to download a few resources and expose the environment with full access. However, this may not be a great idea when other processes need to run in the background. Perhaps we'd want to spin up a database service, load in some large documents, or maybe set up a proxy service for a safe connection.
# 

# To transform our basic setup into a functional development space with a diverse range of processes, we've deployed a series of microservices in the background that a user or system could rely on. [**Microservices**](https://en.wikipedia.org/wiki/Microservices) are autonomous services performing specific functions and communicating through lightweight connection protocols. In your environment, this includes the Jupyter Labs server along with several other services that could be useful to investigate and experiment with.
# 
# <br>

# <!-- <img src="https://drive.google.com/uc?export=view&id=1r0BH_zROmGosrsUt_hhAY4azXc4wtjea" width=800/> -->
# <!-- <img src="imgs/docker-ms.png" width=1000/> -->
# > <img src="https://dli-lms.s3.amazonaws.com/assets/s-fx-15-v1/imgs/docker-ms.png" width=1000px/>

# Utilizing [**Docker**](https://www.docker.com/) for microservice orchestration, our setup makes it relatively simple to add new microservices that adhere to principles like **containerization** and **uniformity**:
# 
# - **Containerization:** This process encapsulates each service in a standalone container, comprising the necessary software components — code, runtime, libraries, and system tools. These containers interact with host resources and other services through network ports. Key advantages include:
#     - **Portability:** Facilitating easy transfer and deployment across diverse environments.
#     - **Isolation:** Ensuring independent operation of each container, minimizing service conflicts.
#     - **Scalability:** Simplifying the process of scaling services to meet varying demands or changing the *deployment topology* (which services are running on which resources, where they are located, and who is accessing them).
# 
# - **Uniformity:** Aiming for consistent operation across different environments, Docker ensures that each microservice performs reliably. However, some constraints are noteworthy:
#     - **Hardware Sensitivity:** Performance may vary in environments with differing hardware, underscoring the need for adaptable microservice design.
#     - **Environmental Factors:** Variables like network latency or storage capacity can impact container efficiency.
# 
# For a more comprehensive overview of Docker and containerization for microservice orcherstation, we'd recommend visiting [Docker's official documentation](https://docs.docker.com/) when you have the time. Understanding these principles will be quite useful for those interested in moving their ambitions towards practical deployment.

# ----
# 
# <br>
# 
# ## **Part 2:** The Jupyter Labs Microservice
# 
# Now that we've discussed general microservices, we can focus in on the one you've been interacting with all along: the **Jupyter Labs microservice**. This interactive web application allows you to write and run Python code (among many other things) using the software installed on the remote host! This should already be very familiar to you from web-based services like [Google Colab](https://colab.research.google.com/?utm_source=scs-index), but you may have never had to think about *why* this environment is there and how it's working behind the scenes. However, since we're talking about microservice orchestration for LLM applications, today is a good day to check it out!

# **Question:** Why is the Jupyter Notebook in our course environment?
# 
# **Answer:** Inside some docker-compose file [like the one in `composer`](composer/docker-compose.yaml), a container with the name `jupyter-notebook-server` was launched with the following profile:
# 
# ```yaml
#   lab:
#     container_name: jupyter-notebook-server
#     build:
#       context: ..
#       dockerfile: composer/Dockerfile
#     ports: # Maps a port on the host to a port in the container.
#     - "9010:9010"
#     - "9011:9011"
#     - "9012:9012"
# ```
# 
# In a simple sentence, this component creates a service with the container name `jupyter-notebook-server` which gets constructed by running the routine in [`composer/Dockerfile`](composer/Dockerfile) from the image specified at the top of that file (which you may notice is a slim image with `python` pre-installed).
# 
# After this construction is over and the launch doesn't error out, a user can access the running Jupyter labs session and interact with the provided interface!

# ----
# 
# <br>
# 
# ## **Part 3:** Interacting With Microservices As The Host

# We've established that this Jupyter-presenting microservice exists and we're interacting with it right now. So... what else is there? We referenced [`composer/docker-compose.yaml`](composer/docker-compose.yaml) earlier, and can investigate it to see what other components were created as part of our spin-up routine. This happens because a version of this file is ran from the host environment (outside of the microservices) using a command like this:

# ```sh
# > docker compose up -d
# ## Building may also happen here if that hasn't happened yet
# 
# Starting docker_router                 ... done
# Starting llm_client                    ... done
# Starting s-fx-15-v1-task4_assessment_1 ... done
# Recreating jupyter-notebook-server     ... done
# Recreating frontend                    ... done
# Recreating s-fx-15-v1-task4_nginx_1    ... done
# Recreating modifier                    ... done
# >
# ```

# ### **Interacting From *Outside* Our Jupyter Labs Microservice**
# 
# After our microservices have been started, we can try to check the status of the other microservices from the host environment via a simple command: `docker ps -a` (or a less verbose version):

# In[5]:


'''
> docker ps -a
CONTAINER ID   IMAGE                            COMMAND                  CREATED          STATUS                   PORTS                                                                     NAMES
7eff861362dc   s-fx-15-v1-task4_lab             "jupyter lab --ip 0.…"   14 minutes ago   Up 14 minutes            8888/tcp, 0.0.0.0:9010-9012->9010-9012/tcp, :::9010-9012->9010-9012/tcp   jupyter-notebook-server...
(too much info)


>  docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Ports}}"
NAMES                           IMAGE                            PORTS
s-fx-15-v1-task4_nginx_1        nginx:1.15.12-alpine             0.0.0.0:80->80/tcp, :::80->80/tcp
frontend                        s-fx-15-v1-task4_frontend        0.0.0.0:8090->8090/tcp, :::8090->8090/tcp
jupyter-notebook-server         s-fx-15-v1-task4_lab             8888/tcp, 0.0.0.0:9010-9012->9010-9012/tcp, :::9010-9012->9010-9012/tcp
llm_client                      s-fx-15-v1-task4_llm_client      0.0.0.0:9000->9000/tcp, :::9000->9000/tcp
docker_router                   s-fx-15-v1-task4_docker_router   0.0.0.0:8070->8070/tcp, :::8070->8070/tcp
s-fx-15-v1-task4_assessment_1   s-fx-15-v1-task4_assessment      0.0.0.0:81->8080/tcp, :::81->8080/tcp
''';


# This shows us our list of running containers and gives us a decent starting point to interface with our microservices from outside our containers. Some of the things we could do from this context include:
# 
# - Moving files to and from containers via routines like `scp` (secure copy protocol) or `docker cp`.
#     - `docker cp jupyter-notebook-server:/dli/task/paint-cat.jpg .`
# - Executing commands in a running container.
#     - `docker exec -it jupyter-notebook-server /bin/bash -c "ls"`
# - Querying for the logs of a container to see its status and execution processes.
#     - `docker logs jupyter-notebook-server`

# <br>
# 
# ### **Interacting From *Inside* Our Jupyter Labs Microservice**
# 
# From inside, a container can only interface with other containers via the exposed ports and the resources provided to them. To illustrate, note that this Jupyter Labs notebook doesn't even have Docker installed, much less have access to the host's Docker instance:

# In[6]:


## Should fail
get_ipython().system('docker ps -a')


# <br>
# 
# This is great in general for security purposes but might make it challenging to interact with other microservices. What exactly can we do from inside our container?
# 
# From the host environment, we could provide a very small window into the outside world via something like the `docker_router` service. The exact code used to create the service is available in [`docker_router/docker_router.py`](docker_router/docker_router.py) and [`docker_router/Dockerfile`](docker_router/Dockerfile), which will readily imply that `help` may be one of the things you can query. Below is an example of a shell network query command which can be used to invoke the `help` routine:

# In[7]:


## Should fail in colab, will work in course environment
get_ipython().system('curl -v docker_router:8070/help')


# <br>
# 
# The `curl` interface shown above can be very useful in general but is a bit unoptimized for our Python environment. Luckily, Python's `requests` library gives a much easier set of utilities to work with, so we'll query the containers path hinted at above using a more Pythonic interface as follows:

# In[8]:


## Should fail in colab, will work in course environment
import requests

## Curl request. Best for shell environments
get_ipython().system('curl -v docker_router:8070/containers')

## Print all running containers
requests.get("http://docker_router:8070/containers").json()

## Print the running microservices
for entry in requests.get("http://docker_router:8070/containers").json():
    if entry.get("status") == 'running':
        print(entry.get("name"))


# <br>
# 
# From this, we can at least know what other microservices are running and can consider what their purposes might be:
# - **docker_router**: The service we're interacting with to find this info.
# - **jupyter-notebook-server**: The service we discussed that is running this Jupyter session.
# - **frontend**: Probably some kind of web interface...
# - **llm_client**: Probably some kind of llm server?
# - **s-fx-<...>**: Some background services (data loader, proxy service, assessment manager), which will not be discussed.

# <!-- <img src="imgs/environment.png" width=800/> -->
# > <img src="http://dli-lms.s3.amazonaws.com/assets/s-fx-15-v1/imgs/environment.png" width=800px/>

# 
# Aside from the last few components, all of the details regarding these components can once again be found in the [`composer`](composer) directory.

# ----
# 
# <br>
# 
# ## **Part 4:** Checking Our Frontend
# 
# Moreso than anything, this notebook is here to open the environment up for exploration and give some potential directions to look over if the microservice construction details are of interest. Since you might interact with some of these microservices throughout the course, knowing how they were made will hopefully prove quite useful!
# 
# While we're at it, why don't we consider the main other microservice we'll need to interact with: **the frontend**. This microservice will host a webpage that you will need to interface with for the final assessment. Please run the following curl command to confirm that your frontend service is up and running! 

# In[10]:


## Commented out by default since it will yield a lot of output
get_ipython().system('curl -v frontend:8090')


# The command should return a `200 OK` response along with a webpage (i.e., a response that starts with `<!doctype html>`), which acts as a useful health check but isn't user-friendly. To access the webpage:
# 
# - **Raw Port Access (Default)**, you would change your URL to use the non-default port `8090` by entering `http://<...>.courses.nvidia.com:8090` in your browser. While this method works, it results in a barebones interface with some limitations, such as port protection mechanisms that might block access, reduced functionality due to incomplete integration with the default server settings, and potential security risks from exposing raw ports to users.
# - **Reverse-Proxy Access**, a different port is reverse-proxied and mapped to `http://<...>.courses.nvidia.com/8090` (where we will use `8091` since the application code has to change) . Reverse-proxying is beneficial because it hides the raw port from users, enhancing security by reducing direct exposure of backend services. It simplifies the URL structure, making it easier for users to access services without remembering specific port numbers. Additionally, reverse-proxying enables better load balancing and easier management of SSL certificates, providing a more seamless and secure user experience. Details out of scope for course, but feel free to check out [**`composer/nginx.conf`**](composer/nginx.conf) and [**`frontend/frontend_server_rproxy.py`**](frontend/frontend_server_rproxy.py).
# 
# **You can run the cell below to generate a link:**

# In[ ]:


get_ipython().run_cell_magic('js', '', 'var url = \'http://\'+window.location.host+\':8090\';\nelement.innerHTML = \'<a style="color:green;" target="_blank" href=\'+url+\'><h1>< Link To Gradio Frontend ></h1></a>\';\n')


# ***Before trying it out, be warned that the frontend microservice doesn't actually work... yet. You'll have several opportunities to interact with it and enable some functionality throughout the course, so make sure you're in the course environment for those components.***
