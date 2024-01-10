# Chat-with-your-PDF-using-Azure-Cognitive-Search-and-LangChain.
Chat with your PDF data using Azure Cognitive Search and LangChain.

## <b> About ACS </b>
The Azure Cognitive Search LangChain integration, built in Python, provides the ability to chunk the documents, seamlessly connect an embedding model for document vectorization, store the vectorized contents in a predefined index, perform similarity search (pure vector), hybrid search and hybrid with semantic search. It also provides configurability to create your own index and apply scoring profiles to achieve better search accuracy. With LangChain, you can combine native workflows (indexing and querying) with non-native workflows (like chunking and embedding) to create an end-to-end similarity search solution.


## <b> Mandatory requirement credentials </b>

### Azure Cognitive Search credentials
1. AZURE_COGNITIVE_SEARCH_ENDPOINT = "YOUR_AZURE_SEARCH_ENDPOINT"
2. AZURE_COGNITIVE_SEARCH_KEY = "YOUR_AZURE_SEARCH_KEY"
3. AZURE_COGNITIVE_SEARCH_INDEX_NAME = "YOUR_AZURE_INDEX_NAME"

### Azure OpenAI Credentials
1. os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
2. os.environ['OPENAI_API_TYPE'] = 'azure'
3. os.environ['OPENAI_API_BASE'] = "YOUR_OPENAI_AZURE_BASE_URL"
4. os.environ['OPENAI_API_VERSION'] = "YOUR_OPENAI_API_VERSION"
'''
## System Setup:
Step 1. Create a virtual environment
  > conda create -p myenv python=3.9 -y

step 2. Activate the environment:
  > conda activate myenv/

step 3. Install all the requirements:
  > pip install -r requirements.txt
 
step 4. start writing the code, with standard file name as main.py

### <b> Steps: </b>
1. Upload the PDF Files
2. Extract the content from the PDF
3. Split content into Chunk
4. Do Embeddings of the data (Download embeddings from the Azure OpenAI)
5. Store Data into Vector Store (eg: ACS)
6. User can pass a Prompt
7. Based on User query, Similarity search will apply
8. Get the Output.

## <b> Output </b>
![image](https://github.com/Chandrakant817/Chat-with-PDF-using-LangChain/assets/69152112/e203a8a7-0388-418b-b11b-e8e2a18e9d74)

## <b> Reference </b>
https://techcommunity.microsoft.com/t5/ai-azure-ai-services-blog/azure-cognitive-search-and-langchain-a-seamless-integration-for/ba-p/3901448
