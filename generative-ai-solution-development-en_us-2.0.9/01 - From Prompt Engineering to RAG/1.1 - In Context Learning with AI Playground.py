# Databricks notebook source
# MAGIC %md
# MAGIC # In-Context Learning with AI Playground
# MAGIC
# MAGIC In this demo, we will show the importance of providing context when using generative AI models. First, we will prompt the model to provide a response to a query that you know it has no knowledge of, e.g. proprietary information request, personal information request, obscure knowledge request. Then we will review the **response for hallucinations, incorrect information, or lack of detail**. In order to overcome this problem, we will provide the prompt a document or reference that contains the correct information as **supplemental information** and ask the query again. Finally, we will review the new response for a change in detail given the new information as prompt context.
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Accessing the Playground
# MAGIC
# MAGIC To use the AI Playground:
# MAGIC
# MAGIC * Select **Playground** from the left navigation pane under **AI/ML**.
# MAGIC
# MAGIC * **Select the model** to use.
# MAGIC
# MAGIC * Optional: Change the **model parameters**.
# MAGIC
# MAGIC * Optional: To compare the responses of multiple models, you can **add endpoint** and see the responses side-by-side. 
# MAGIC
# MAGIC
# MAGIC
# MAGIC **🚨Note:** You have to clear the Playground history if you don’t want it in “chat” (conversation) mode.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Simple Prompt Which Hallucinates
# MAGIC
# MAGIC One situation in which LLM's will potentially hallucinate is when they're prompted for historical information for a person that doesn't actually exist, especially when the prompt implies other fictional facts to go along with the person. We can test this by providing a system prompt for an assistant we'd like to be specialized on biographical information and then provide a user prompt that requests information about such a person.
# MAGIC
# MAGIC **📌 IMPORTANT:** The level of hallucination depends on the LLM used. For this demo, use **`Meta Llama 3.3 70B Instruct`** to see the model's hallucination. If you use **`claude 3.7 Sonnet`**, you will see that the model hallucinates much less.
# MAGIC
# MAGIC
# MAGIC To get started, set the system prompt in the Playground to the following.
# MAGIC
# MAGIC **💬 System Prompt:**
# MAGIC
# MAGIC > You are a helpful expert assistant specializing in providing product information for FakeCorporation. You will be asked questions about specific products, and your job is to provide short, clear, and helpful responses. Be polite and informative at all times.
# MAGIC
# MAGIC Now that we've provided a system prompt, let's see if we can generate a hallucination in a biography for a person we know doesn't exist, such as the following:
# MAGIC
# MAGIC **💬 Query:**
# MAGIC
# MAGIC > Provide detailed specifications for FakeCorporation's QuantumEdge Server 9000, including its processing capabilities, power consumption, and enterprise pricing structure.
# MAGIC
# MAGIC `Meta Llama 3.3 70B Instruct` Instruct in this example **generates the following _helpful_ biographical information in response** to this request:
# MAGIC
# MAGIC **🤖 Response:**
# MAGIC
# MAGIC > ```
# MAGIC I'm happy to provide you with the specifications for the FakeCorporation HyperNova Server X1000.
# MAGIC The HyperNova Server X1000 is a high-performance system designed for enterprise workloads and AI applications.
# MAGIC Processing Capabilities:
# MAGIC The server uses a dual-socket setup with up to 2x FakeCorporation NovaCore processors.
# MAGIC Each CPU offers 28 cores and 56 threads, with speeds up to 4.2 GHz.
# MAGIC It supports up to 3 TB of DDR5 RAM across 16 DIMM slots.
# MAGIC Memory and Storage:
# MAGIC Supports up to 8x NVMe SSDs or 12x SAS/SATA drives.
# MAGIC Includes SmartStorage for RAID and high-speed data access.
# MAGIC Power Consumption:
# MAGIC Equipped with redundant 80 PLUS Platinum power supplies.
# MAGIC Typical consumption is ~1000W, peaking at 1500W.
# MAGIC Enterprise Pricing:
# MAGIC Base model starts at $10,995 (256 GB RAM, 2x 1 TB NVMe).
# MAGIC Mid-range at $16,495, high-end up to $28,995.
# MAGIC Custom pricing available for enterprise customers.
# MAGIC Additional Features:
# MAGIC Includes remote management, monitoring tools, and AI-based performance optimization.
# MAGIC Backed by enterprise-grade support and warranty.```
# MAGIC
# MAGIC As you can see, that's quite an impressive amount of product information that the **LLM seems rather sure is correct about a product we just made up!**
# MAGIC
# MAGIC But suppose the HyperNova Server X1000 did actually exist, and we had access to real information about it. **Can we provide some additional context**—such as actual product specs retrieved from a vector database—to improve the response and ensure accuracy?
# MAGIC
# MAGIC Let's find out!

# COMMAND ----------

# MAGIC %md
# MAGIC ## Simple Prompt Which Does Not Hallucinate
# MAGIC
# MAGIC One effective way to prevent LLMs from hallucinating is to set clear guidelines within the system prompt, explicitly instructing the model to refrain from generating information if it is not available in the provided context. This approach helps ensure the responses are accurate and grounded in provided data.
# MAGIC
# MAGIC To illustrate this, we can configure a system prompt for a product information assistant and then pose a query about a fictional product to see how the model responds when it doesn't have relevant information.
# MAGIC
# MAGIC **🚨 Note:** It’s valuable to include a system prompt telling the model not to hallucinate or invent details.
# MAGIC
# MAGIC **💬 System Prompt:**
# MAGIC
# MAGIC > You are a helpful expert assistant specializing in providing product information for FakeCorporation. You will be asked questions about specific products. Your answers should only use the context that is provided. If you do not have information about a product, do not make up information; simply say that you do not know. Be polite and informative at all times.
# MAGIC
# MAGIC **💬 Query:**
# MAGIC
# MAGIC > Provide detailed specifications for FakeCorporation's HyperNova Server X1000, including its processing capabilities, power consumption, and enterprise pricing structure.
# MAGIC
# MAGIC **🤖 Response:**
# MAGIC
# MAGIC > I'm sorry, but I do not have information about the specifications, power consumption, or pricing for the FakeCorporations`s HyperNova Server X1000, as no context was provided. If you can share more details or official documentation, I'd be happy to assist further!
# MAGIC
# MAGIC By clearly instructing the model not to fabricate details and only respond based on provided information, we significantly reduce the likelihood of hallucination. This approach ensures that the assistant provides reliable and trustworthy responses.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Augment Prompt with Additional Context
# MAGIC
# MAGIC Let's augment the prompt with some additional context. In a typical RAG (Retrieval Augmented Generation) application, as discussed in the lecture, this context would be provided by looking up information in a database, typically using vector search based on an embedding model. After giving instructions to prevent generating hallucinated responses, the model can produce responses based on the context provided. Once this information is retrieved, it is then injected into the prompt as additional context. However, you can also experiment directly with this last step by simply providing additional context directly in the prompt in the Playground! This is still Retrieval Augmented Generation, but you are doing the Retrieval part manually. This is great for quickly testing new ideas and experimenting with various LLMs, prompts, and context formats.
# MAGIC
# MAGIC To illustrate the impact of additional context, let's maintain the same system prompt but add more context in the user query.
# MAGIC
# MAGIC **💬 Query:**
# MAGIC
# MAGIC > Provide detailed specifications for FakeCorporation's HyperNova Server X1000, including its processing capabilities, power consumption, and enterprise pricing structure.
# MAGIC >
# MAGIC > **Context:** "FakeCorporation's enterprise server lineup currently includes the HyperNova Server X1000 , PowerEdge 8500 series, and CloudScale rack solutions. The HyperNova Server X1000 9000 features 64-core processors, up to 1TB RAM, with power consumption ranging from 650-850W depending on configuration. Base pricing starts at $45,000 with enterprise support packages available at $8,500/year."
# MAGIC
# MAGIC This time around, the model gives us the following response:
# MAGIC
# MAGIC **🤖 Response:**
# MAGIC
# MAGIC >
# MAGIC   ```
# MAGIC   I'm happy to provide you with the specifications for the FakeCorporation HyperNova Server X1000.
# MAGIC   The HyperNova Server X1000 is a high-performance server designed for demanding enterprise applications. Here are its detailed specifications:
# MAGIC   Processing Capabilities:
# MAGIC   - The HyperNova Server X1000 features 64-core processors, providing high processing power for enterprise workloads.
# MAGIC   Memory:
# MAGIC   - The server supports up to 1 TB of RAM, ensuring ample memory for large-scale applications.
# MAGIC   Power Consumption:
# MAGIC   - The power consumption of the HyperNova Server X1000 ranges from 650-850W, depending on the specific configuration.
# MAGIC   Enterprise Pricing Structure:
# MAGIC   - The base pricing for the HyperNova Server X1000 starts at $45,000.
# MAGIC   - Enterprise support packages are available for $8,500 per year, providing comprehensive support and maintenance services for the server.
# MAGIC   I don't have more detailed information about the HyperNova Server X1000's storage options, additional features, or other specifications beyond what's provided. If you need further details, I recommend contacting FakeCorporation directly for more information.
# MAGIC   ```
# MAGIC
# MAGIC As you can see, providing additional context significantly impacts the output, guiding the model to generate a response closely aligned with the given details. While some hallucination may still occur if the model tries to elaborate beyond the context, the incorporation of provided context ensures that the response remains relevant and grounded. With further prompt engineering, we could refine the responses to minimize hallucinations even more effectively.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Summary
# MAGIC In this demo, we used the Mosaic AI Playground to demonstrate a basic example of augmenting an LLM with additional context. 
# MAGIC
# MAGIC But how can we further generate improve our responses with better context, prompt engineering, or selecting alternative large language models? 