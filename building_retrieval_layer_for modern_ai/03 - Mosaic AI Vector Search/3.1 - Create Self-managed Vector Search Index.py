# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Create Self-managed Vector Search Index
# MAGIC
# MAGIC In the previous step, we chunked the raw PDF document pages into small sections, computed the embeddings, and saved it as a Delta Lake table. Our dataset is now ready. 
# MAGIC
# MAGIC Next, we'll configure Databricks Vector Search to ingest data from this table.
# MAGIC
# MAGIC Vector search index uses a Vector search endpoint to serve the embeddings (you can think about it as your Vector Search API endpoint). <br/>
# MAGIC Multiple Indexes can use the same endpoint. Let's start by creating one.
# MAGIC
# MAGIC
# MAGIC **Learning Objectives:**
# MAGIC
# MAGIC *By the end of this demo, you will be able to;*
# MAGIC
# MAGIC * Set up an endpoint for Vector Search.
# MAGIC
# MAGIC * Store the embeddings and their metadata using the Vector Search.
# MAGIC
# MAGIC * Inspect the Vector Search endpoint and index using the UI. 
# MAGIC
# MAGIC * Retrieve documents from the vector store using similarity search.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## REQUIRED - SELECT CLASSIC COMPUTE
# MAGIC Before executing cells in this notebook, please select your classic compute cluster in the lab. Be aware that **Serverless** is enabled by default.
# MAGIC
# MAGIC Follow these steps to select the classic compute cluster:
# MAGIC 1. Navigate to the top-right of this notebook and click the drop-down menu to select your cluster. By default, the notebook will use **Serverless**.
# MAGIC
# MAGIC 2. If your cluster is available, select it and continue to the next cell. If the cluster is not shown:
# MAGIC
# MAGIC    - Click **More** in the drop-down.
# MAGIC    
# MAGIC    - In the **Attach to an existing compute resource** window, use the first drop-down to select your unique cluster.
# MAGIC
# MAGIC **NOTE:** If your cluster has terminated, you might need to restart it in order to select it. To do this:
# MAGIC
# MAGIC 1. Right-click on **Compute** in the left navigation pane and select *Open in new tab*.
# MAGIC
# MAGIC 2. Find the triangle icon to the right of your compute cluster name and click it.
# MAGIC
# MAGIC 3. Wait a few minutes for the cluster to start.
# MAGIC
# MAGIC 4. Once the cluster is running, complete the steps above to select your cluster.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Requirements
# MAGIC
# MAGIC Please review the following requirements before starting the lesson:
# MAGIC
# MAGIC * To run this notebook, you need to use one of the following Databricks runtime(s): **17.3.x-cpu-ml-scala2.13**
# MAGIC
# MAGIC
# MAGIC **🚨 Important:** This demonstration relies on the resources established in the previous one. Please ensure you have completed the prior demonstration before starting this one.

# COMMAND ----------

# MAGIC %pip install -U -qqqq databricks-vectorsearch 'mlflow-skinny[databricks]==3.4.0' PyPDF2==3.0.0 databricks-sdk flashrank 
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## Demo Overview
# MAGIC
# MAGIC Demo Diagram
# MAGIC ![image_1770381864031.png](./image_1770381864031.png "image_1770381864031.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create a "Self-Managed" Vector Search Index
# MAGIC
# MAGIC Setting up a Databricks Vector Search index involves a few key steps. First, you need to decide on the method of providing vector embeddings. Databricks supports three options: 
# MAGIC
# MAGIC * providing a source Delta table containing text data
# MAGIC * **providing a source Delta table that contains pre-calculated embeddings**
# MAGIC * using the Direct Vector API to create an index on embeddings stored in a Delta table
# MAGIC
# MAGIC In this demo, we will go with the second method. 
# MAGIC
# MAGIC Next, we will **create a vector search endpoint**. And in the final step, we will **create a vector search index** from a Delta table. 
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

import time

# assign vs search endpoint by username
vs_endpoint_name = "archivx_vs" 
print(f"Assigned Vector Search endpoint name: {vs_endpoint_name}.")

# COMMAND ----------

def wait_for_vs_endpoint_to_be_ready(vsc, vs_endpoint_name):
  for i in range(180):
    endpoint = vsc.get_endpoint(vs_endpoint_name)
    status = endpoint.get("endpoint_status", endpoint.get("status"))["state"].upper()
    if "ONLINE" in status:
      return endpoint
    elif "PROVISIONING" in status or i <6:
      if i % 20 == 0: 
        print(f"Waiting for endpoint to be ready, this can take a few min... {endpoint}")
      time.sleep(10)
    else:
      raise Exception(f'''Error with the endpoint {vs_endpoint_name}. - this shouldn't happen: {endpoint}.\n Please delete it and re-run the previous cell: vsc.delete_endpoint("{vs_endpoint_name}")''')
  raise Exception(f"Timeout, your endpoint isn't ready yet: {vsc.get_endpoint(vs_endpoint_name)}")

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
from databricks.sdk import WorkspaceClient
import databricks.sdk.service.catalog as c

vsc = VectorSearchClient(disable_notice=True)

# check the status of the endpoint
wait_for_vs_endpoint_to_be_ready(vsc, vs_endpoint_name)
print(f"Endpoint named {vs_endpoint_name} is ready.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### View the Endpoint
# MAGIC
# MAGIC After the endpoint is created, you can view your endpoint on the [Vector Search Endpoints UI](#/setting/clusters/vector-search). Click on the endpoint name to see all indexes that are served by the endpoint.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Connect Delta Table with Vector Search Endpoint
# MAGIC
# MAGIC After creating the endpoint, we can create the **vector search index**. The vector search index is created from a Delta table and is optimized to provide real-time approximate nearest neighbor searches. The goal of the search is to identify documents that are similar to the query. 
# MAGIC
# MAGIC **Vector search indexes appear in and are governed by Unity Catalog.**

# COMMAND ----------

# the table we'd like to index
source_table_fullname = "genai.arxiv_schema.pdf_text_embeddings"

# where we want to store our index
vs_index_fullname = "genai.arxiv_schema.pdf_ix"

# COMMAND ----------

import time
def wait_for_index_to_be_ready(vsc, vs_endpoint_name, index_name):
  for i in range(180):
    idx = vsc.get_index(vs_endpoint_name, index_name).describe()
    index_status = idx.get('status', idx.get('index_status', {}))
    status = index_status.get('detailed_state', index_status.get('status', 'UNKNOWN')).upper()
    url = index_status.get('index_url', index_status.get('url', 'UNKNOWN'))
    if "ONLINE" in status:
      return
    if "UNKNOWN" in status:
      print(f"Can't get the status - will assume index is ready {idx} - url: {url}")
      return
    elif "PROVISIONING" in status:
      if i % 40 == 0: print(f"Waiting for index to be ready, this can take a few min... {index_status} - pipeline url:{url}")
      time.sleep(10)
    else:
        raise Exception(f'''Error with the index - this shouldn't happen. DLT pipeline might have been killed.\n Please delete it and re-run the previous cell: vsc.delete_index("{index_name}, {vs_endpoint_name}") \nIndex details: {idx}''')
  raise Exception(f"Timeout, your index isn't ready yet: {vsc.get_index(index_name, vs_endpoint_name)}")

# COMMAND ----------

# DBTITLE 1,Cell 16
def index_exists(vsc, endpoint_name, index_full_name):
  try:
      dict_vsindex = vsc.get_index(endpoint_name, index_full_name).describe()
      return dict_vsindex.get('status').get('ready', False)
  except Exception as e:
      if 'RESOURCE_DOES_NOT_EXIST' not in str(e) and 'NOT_FOUND' not in str(e) and 'does not exist' not in str(e):
          print(f'Unexpected error describing the index. This could be a permission issue.')
          raise e
  return False

# COMMAND ----------

index_exists(vsc, vs_endpoint_name, vs_index_fullname)

# COMMAND ----------

# create or sync the index
if not index_exists(vsc, vs_endpoint_name, vs_index_fullname):
  print(f"Creating index {vs_index_fullname} on endpoint {vs_endpoint_name}...")
  vsc.create_delta_sync_index(
    endpoint_name=vs_endpoint_name,
    index_name=vs_index_fullname,
    source_table_name=source_table_fullname,
    pipeline_type="TRIGGERED", #Sync needs to be manually triggered
    primary_key="id",
    embedding_dimension=1024, #Match your model embedding size (gte)
    embedding_vector_column="embedding"
  )
else:
  # trigger a sync to update our vs content with the new data saved in the table
  vsc.get_index(vs_endpoint_name, vs_index_fullname).sync()

# let's wait for the index to be ready and all our embeddings to be created and indexed
wait_for_index_to_be_ready(vsc, vs_endpoint_name, vs_index_fullname)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Search for Similar Content
# MAGIC
# MAGIC That's all we have to do. Databricks will automatically capture and synchronize new entries in your Delta Lake Table.
# MAGIC
# MAGIC Note that depending on your dataset size and model size, index creation can take a few seconds to start and index your embeddings.
# MAGIC
# MAGIC Let's give it a try and search for similar content.
# MAGIC
# MAGIC **📌 Note:** `similarity_search` also supports a filter parameter. This is useful to add a security layer to your RAG system: you can filter out some sensitive content based on who is doing the call (for example filter on a specific department based on the user preference).
# MAGIC

# COMMAND ----------

import mlflow.deployments

deploy_client = mlflow.deployments.get_deploy_client("databricks")
question = "What is the implication of large language models on the us labor market?"
response = deploy_client.predict(endpoint="databricks-gte-large-en", inputs={"input": [question]})
embeddings = [e["embedding"] for e in response.data]
print(embeddings)

# COMMAND ----------

from pprint import pprint

# get similar 5 documents.
results = vsc.get_index(vs_endpoint_name, vs_index_fullname).similarity_search(
  query_vector=embeddings[0],
  columns=["pdf_name", "content"],
  num_results=5)

# format result to align with reranker lib format. 
passages = []
for doc in results.get("result", {}).get("data_array", []):
    new_doc = {"file": doc[0], "text": doc[1]}
    passages.append(new_doc)
pprint(passages)



# COMMAND ----------

# MAGIC %md
# MAGIC ## Re-ranking Search Results
# MAGIC
# MAGIC For re-ranking the results, we will use a very light library. [**`flashrank`**](https://github.com/PrithivirajDamodaran/FlashRank) is an open-source reranking library based on SoTA cross-encoders. The library supports multiple models, and in this example we will use `rank-T5` model. 
# MAGIC
# MAGIC After re-ranking you can review the results to check if the order of the results has changed. 
# MAGIC
# MAGIC **💡Note:** Re-ranking order varies based on the model used!

# COMMAND ----------

import os
wd = os.getcwd()

print(wd)


# COMMAND ----------


from flashrank import Ranker, RerankRequest

# Ensure the model file exists at this path or update the path accordingly
cache_dir = f"{wd}/opt"

ranker = Ranker(model_name="rank-T5-flan", cache_dir=cache_dir)

rerankrequest = RerankRequest(query=question, passages=passages)
results = ranker.rerank(rerankrequest)
print(*results[:3], sep="\n\n")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Conclusion
# MAGIC
# MAGIC In this demo, the objective was to generate embeddings from documents and store them in Vector Search. The initial step involved creating a Vector Search index, which required the establishment of a compute endpoint and the creation of an index that is synchronized with a source Delta table. Following this, we conducted a search for the stored indexes using a sample input query.