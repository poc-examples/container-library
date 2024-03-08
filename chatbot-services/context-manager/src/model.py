from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from models import MixtralLLM

from langchain.text_splitter import RecursiveCharacterTextSplitter

import logging
from collections import defaultdict
import json
import weaviate
import pymysql
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextRequest(BaseModel):
    input_text: str

DB_CONFIG = {
    'host': 'chat-mysql.chat-window.svc.cluster.local',
    'user': 'llm',
    'password': 'password',
    'database': 'forethought'
}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

weaviate_url = "http://weaviate.vectordb.svc.cluster.local"
logger.info(f"Weviate Url: {weaviate_url}")
weaviate_client = weaviate.Client(
    url = weaviate_url,
    auth_client_secret=weaviate.AuthApiKey("admin-api-key"),
)
weaviate_schema = weaviate_client.schema.get()
logger.info(weaviate_schema)

if len(weaviate_schema['classes']) == 0:
    weaviate_client.schema.create_class(
        {
            "class": "Documents",
            "description": "A class for storing documents",
            "vectorizer": "text2vec-transformers",
            "properties": [
                {
                    "name": "content",
                    "dataType": ["text"],
                    "description": "The content of the document",
                    "moduleConfig": {
                        "text2vec-transformers": {
                            "skip": False,
                            "vectorizePropertyName": False
                        }
                    }
                }
            ]
        }
    )

llm = MixtralLLM(
    model_url='http://mixtra7b-service.collective.svc.cluster.local/generate-text',
    headers={'Content-Type': 'application/json'}
)

def spr_smart_filter(spr_map: list[dict], prompt: str) -> list[dict]:
    instruct = """
        Select from the following list which one of the topics are most closely related
        to the inquiry. We will provide the data in the following format:

        "15345: Unique Topic 1 | 234: Unique Topic 2 | 7395: Unique Topic 3"

        Only respond with valid JSON. The structure should be:

        [15345, 7395]
    """

    sprs = ""
    for spr in spr_map:
        sprs += f"{spr['id']}: {spr['topic']} | "

    result = llm._generate(prompt=f"[INST] {instruct} {sprs} [/INST]")
    print(result)

    ids = json.loads(force_array(result))

    filtered_sbrs = []
    for spr in spr_map:
        if str(spr['id']) in ids:
            filtered_sbrs.append(spr)
    
    return filtered_sbrs


def generate_smart_vector(prompt: str):
    instruct = """
        Rewrite the following text so that it is better 
        suited for use in vectordb semantic search. 
        Reduce the number of words in the search text as much as possible.
        Only respond with the answer no description
    """

    ## Smart Query
    return llm._generate(prompt=f"[INST] {instruct} text={prompt} [/INST]")

def fetch_all_documents():
    connection = pymysql.connect(**DB_CONFIG)
    try:
        with connection.cursor() as cursor:
            sql = "SELECT * FROM `documents`"
            cursor.execute(sql)
            result = cursor.fetchall()  # Fetch all rows from the last executed statement
            return result
    except pymysql.MySQLError as e:
        logger.error(f"Error fetching from database: {e}")
        return []
    finally:
        if connection.open:
            connection.close()

def insert_sprs_to_db(sprs):
    # CREATE TABLE IF NOT EXISTS documents (
    #     id INT AUTO_INCREMENT PRIMARY KEY,
    #     topic TEXT,
    #     spr TEXT,
    #     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    # );
    connection = pymysql.connect(**DB_CONFIG)
    try:
        with connection.cursor() as cursor:
            for topic, spr in sprs.items():
                logger.info(f"Inserting: {topic}")
                sql = "INSERT INTO `documents` (topic, spr) VALUES (%s, %s)"
                cursor.execute(sql, (topic, json.dumps(spr)))
        logger.info("Commit")
        connection.commit()
    except Exception as e:
        logger.error(f"Error inserting into database: {e}")
    finally:
        if connection.open:
            connection.close()

def trim_text_around_braces(text):

    start_index = text.find('{')
    end_index = text.rfind('}')
    
    if start_index != -1 and end_index != -1 and end_index > start_index:
        return text[start_index:end_index+1]
    else:
        return text

def force_array(text):

    start_index = text.find('[')
    end_index = text.rfind(']')
    
    if start_index != -1 and end_index != -1 and end_index > start_index:
        return text[start_index:end_index+1]
    else:
        return text

def correct_json(payload):
    instruct = """
            Check the following JSON structure.
            Only correct the structure so it is valid JSON. No backticks or Notes.

            {"Topic One":["Sentence representation of spr 1","Sentence representation of spr 2"],"Topic Two":["Sentence representation of spr 1","Sentence representation of spr 2"],"Topic Three":["Sentence representation of spr 1","Sentence representation of spr 2"]}
    """
    return trim_text_around_braces(llm._generate(prompt=f'[INST] {instruct} {payload} [/INST]'))


def to_sprs(content: str, chunk_size: int) -> dict:
    logger.info(f"Running SPR w/ {chunk_size} chunk size.")
    sprs = []

    instruct = """
            Render the input as a distilled list of succinct statements, assertions, associations, 
            concepts, analogies, and metaphors. The idea is to capture as much, conceptually, as 
            possible but with as few words as possible.

            Write it in a way that makes sense to you, as the future audience will be another language 
            model, not a human. Use complete sentences.

            The output should be valid JSON structure:

            {"Topic One":["Sentence representation of spr 1","Sentence representation of spr 2"],"Topic Two":["Sentence representation of spr 1","Sentence representation of spr 2"]}
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )

    texts = text_splitter.create_documents([content])

    for text in texts:

        result = llm._generate(prompt=f'[INST] {instruct} text="{text}" [/INST]')
        
        while True:
            try:
                json_payload = json.loads(result)
                logger.info(f"{json.dumps(result)}")
                break
            except:
                logger.info(f"Attempting to correct JSON Structure...")
                logger.warning(result)
                result = correct_json(result)

        sprs.append(json_payload)

    merged_dict = defaultdict(list)
    for spr in sprs:
        for key, value in spr.items():
            merged_dict[key].extend(value)

    return merged_dict



@app.post("/document/index")
async def index_text(request: TextRequest):
    content = request.input_text

    sprs = to_sprs(content, 5000)
    logger.info(sprs)
    insert_sprs_to_db(sprs)

    #     sprs.append(output)
    
    # # Loop over the processed text chunks
    # for sbr in sprs:

    #     # Insert them into Weaviate Database under

    #     content_object = {
    #         "content": sbr
    #     }
    
    #     try:

    #         result = weaviate_client.data_object.create(
    #             data_object=content_object,
    #             class_name="Documents"
    #         )
    #         logger.info(f"Document indexed in Weaviate: {result}")
    #         return {"generated_text": f"Content Indexed Successfully: {result}"}
        
    #     except Exception as e:

    #         logger.error(f"Error indexing document in Weaviate: {e}")
    #         return JSONResponse(
    #             status_code=500, 
    #             content={
    #                 "message": "Failed to index document in Weaviate", "error": str(e)
    #             }
            # )

    return {"generated_text": f"<br>{json.dumps(sprs)}"}

@app.post("/v2/rag-text")
async def generate_text(request: TextRequest):
    prompt = request.input_text

    debug = ""

    topics = fetch_all_documents()

    # [{
    #     "id": "1234123",
    #     "topic": "some topic",
    #     "sprs": "sprs"
    # }]

    topic_map = []
    for topic in topics:
        topic_map.append({
            "id": topic[0],
            "topic": topic[1],
            "sprs": topic[3]
        })

    sprs = ""
    if len(topic_map) > 0:
        filtered_sprs = spr_smart_filter(topic_map, prompt)

        print(filtered_sprs)

        for spr in filtered_sprs:
            sprs += f"{spr['topic']}: {json.dumps(spr['sprs'])}"

        print(sprs)

    # ## Generate Smart Vectors for Querying
    # response_text = generate_smart_vector(prompt)

    # print(response_text)
    # debug += "<b>Refined Search Text</b>: " + response_text + "<br><br>"

    # response = (
    #     weaviate_client.query
    #     .get("Documents", ["content"])
    #     .with_near_text({
    #         "concepts": [response_text],
    #         "distance": 0.8
    #     })
    #     .with_additional(["distance"])
    #     .do()
    # )

    # print(response['data']['Get']['Documents'])

    instruct = ""
    # documents = response['data']['Get']['Documents']
    # for document in documents:
    #     instruct += document['content'] + ", "

    debug += "<b>Located</b>: " + instruct + "<br><br>"

    output = llm._generate(prompt=f"[INST] {instruct} {sprs} {prompt} [/INST]") 

    # Extract the relevant information from the output
    response_text = output

    debug += "<b>Result</b>: " + response_text

    return {"generated_text": f"<br>{response_text}"}

@app.post("/generate-text")
async def generate_text(request: TextRequest):
    prompt = request.input_text

    styling = """
        Output all responses in Markdown format.  
        Ensure clear and concise formatting.  
        Use headings for different sections.  
        Use bullet points or numbered lists for organized content.  
        Ensure that all text is aligned properly and easy to read.  
        Adhere to Markdown syntax standards for optimal readability.
    """

    output = llm._generate(prompt=f"[INST] {styling} {prompt} [/INST]") 

    print(output)

    # Extract the relevant information from the output
    # response_text = output.get("choices")[0]["text"]
    response_text = output

    return {"generated_text": response_text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
