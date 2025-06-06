from agents import Agent, ModelSettings, OpenAIChatCompletionsModel, function_tool
from pydantic import BaseModel, Field
from typing import Literal, List, Optional, Union
from dotenv import load_dotenv, find_dotenv
from datetime import datetime
from prompts import load_prompt
from client_factory import get_agent_open_ai_client, get_agent_model_name
from config import config
import logging
import requests
import json

load_dotenv(find_dotenv())

### SETTINGS ### 
general_purpose = OpenAIChatCompletionsModel(
    model=get_agent_model_name('general_purpose'),
    openai_client=get_agent_open_ai_client('general_purpose'),
)

critique = OpenAIChatCompletionsModel(
    model=get_agent_model_name('critique'),
    openai_client=get_agent_open_ai_client('critique'),
)

model_settings = ModelSettings(
    temperature=0,
    top_p=0.1,

    max_tokens=3900
    )

### DATA CLASSES FOR GENERAL PURPOSE ###  USUALLY INPUT AND OUTÛT OF THE APP BUT NOT TIED TO THE LLM API CALL ITSELF

class InputCriterion(BaseModel):
    criterion_id: str = Field(..., description="Unique identifier for the criterion.")
    text: str = Field(..., description="The full text description of the criterion.")
    nature: Literal["requirement", "prohibition", "descriptor"] = Field(description="The nature of the criterion: 'requirement' (must be met), 'prohibition' (must not be met), 'descriptor' (neutral information).")
    analyst_notes: List[str]  = Field(..., description="Provides analyst notes that help to assess a criteria.")

class EvidenceSnippet(BaseModel):
    text: str = Field(..., description="The text of the snippet.")

### JUDGE AGENT ###    (THIS IS STRUCTURED OUTPUT FOR AGENT; API CALL)
class JudgeOutput(BaseModel):
    verdict: Literal["yes", "no", "likely yes", "likely no", "unclear", "contradictory"] = Field(
        ...,
        description="""
        - 'yes' means the information memorandum clearly indicates the criterion is met.
        - 'no' means the information memorandum clearly indicates the criterion is NOT met
        - 'likely yes' means strong indication it meets the criteria, but not explicitly stated; inference made.
        - 'likely no' means strong indication it does not meet the criteria, but not explicitly stated; inference made.
        - 'unclear' means insufficient information, the memorandum does not provide enough direct or indirect information to make a determination
        - 'contradictory' means the memorandum contains conflicting information regarding this criterion
        """)
    reasoning: str = Field(..., description="The reasoning behind the verdict.")
    evidence_snippets: List[EvidenceSnippet] = Field(..., description="List of snippets of text that support the verdict.")
    confidence_score: float = Field(..., description="The confidence score of the verdict.")
    criterion_id: str = Field(..., description="The ID of the criterion being judged.")


logger = logging.getLogger("new.agents.main")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    
    log_file_name = "script_run.log"
    
    fh = logging.FileHandler(log_file_name, mode='w')
    fh.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(message)s')
    
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.propagate = False
else:
    logger.info("Logger already configured.")

@function_tool
def company_details_retriever(detailed_query: str) -> str:

    """
    Fetch the company detail that is missing. It aims to return very low level details about the company.

    Args:
        detailed_query: Very detailed and prescriptive query that helps to fetch very detailed information about the company. In case the query is about time related data it MUST specify the time period explicitly.
    """

    api_url = config.get('retriever.search_url')
    if not api_url:
        logger.error("Retriever search_url not configured in config.")
        return "Error: Retriever search_url not configured."

    payload = {
        "query": detailed_query,
        "mode": "local",
        "only_need_context": False,
        "only_need_prompt": False,
        "top_k": 5,
        "max_token_for_text_unit": 5000,
        "response_type": "Single Paragraph",
        "max_token_for_global_context": 5000,
        "max_token_for_local_context": 5000
    }
    try:
        response = requests.post(api_url, json=payload)
        response.raise_for_status()
        result = response.json()
        if "response" not in result:
            logger.error(f"Retriever Error: 'response' key not found. Query: '{detailed_query}'")
            return "Error: Retriever did not return a 'response'."
        return result["response"].replace("\u20ac", "EUR").replace("\u2248", "=").replace("≈", "=")
    except requests.exceptions.RequestException as e:
        logger.error(f"Retriever API request failed for query '{detailed_query}': {e}")
        return f"Error: Retriever API request failed - {e}"
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON response from retriever for query '{detailed_query}': {e}")
        return f"Error: Failed to decode JSON from retriever - {e}"

##################### AGENTS #####################

judge_agent = Agent(
    name="JudgeAgent",
    model=critique,
    model_settings=model_settings,
    output_type=JudgeOutput,
    instructions=load_prompt("criteria.judge"),
    tools=[company_details_retriever]
    )
