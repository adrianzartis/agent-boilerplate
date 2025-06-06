import logging
from .criteria_agents import criteria_extraction_agent, ExtractedCriteria
from agents import Runner
from prompts import format_prompt
from typing import Optional

import json
from typing import List
from pydantic import ValidationError
from .criteria_agents import InputCriterion
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "input"

def load_criteria_from_json(file_path: str, current_year: str) -> List[InputCriterion]:
    """Loads a list of investment criteria from a JSON file."""
    file_path = DATA_DIR / file_path
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("JSON root must be a list of criteria.")

        criteria_list = [InputCriterion(**item) for item in data]

        for criteria in criteria_list:
            for note in criteria.analyst_notes:
                note = note.replace("{{current_year}}", current_year)

        return criteria_list
    except FileNotFoundError:
        raise ValueError(f"Criteria file not found: {file_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Error decoding JSON from criteria file: {file_path}")
    except ValidationError as e:
        raise ValueError(f"Invalid criteria data structure in {file_path}: {e}")


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

async def extract_criteria_from_text(investment_criteria_text: str) -> Optional[ExtractedCriteria]:
    """
    Extracts structured criteria from a given text using an LLM agent.
    """
    logger.info("Starting criteria extraction from text.")
    
    variables = {
        "investment_criteria": investment_criteria_text
    }
    
    formatted_criteria_extraction_prompt = format_prompt("criteria.criteria_extraction", variables)
    
    if formatted_criteria_extraction_prompt is None:
        logger.error("Failed to load or format 'criteria.criteria_extraction' prompt.")
        return None
    
    criteria_extraction_agent.instructions = formatted_criteria_extraction_prompt
    
    try:
        run_result = await Runner.run(criteria_extraction_agent, input="")
    except Exception as e:
        logger.error(f"Error during Runner.run for criteria_extraction_agent: {e}", exc_info=True)
        raise ValueError(f"Agent run failed: {e}")

    if isinstance(run_result.final_output, ExtractedCriteria):
        logger.info("Successfully extracted structured criteria.")
        return run_result.final_output
    else:
        logger.error(f"CriteriaExtractionAgent returned unexpected output type: {type(run_result.final_output)}. Expected ExtractedCriteria.")
        raise ValueError(f"CriteriaExtractionAgent returned unexpected output type: {type(run_result.final_output)}")

async def get_extracted_criteria() -> Optional[ExtractedCriteria]:
    """
    Extracts structured criteria from a predefined internal text.
    """
    logger.info("Getting predefined extracted criteria.")
    investment_criteria_text_block = """
    1. Target Company Profile
    Our investment strategy targets mid-sized companies demonstrating significant growth potential, competitive advantages, and scalable operations.
    Revenue: $10M - $200M annually
    EBITDA: $2M - $30M annually
    Operational History: Minimum of 3 years
    2. Sector Focus
    Preferred sectors include those with sustainable growth trajectories and resilience:
    Technology (Enterprise software, cybersecurity, SaaS)
    Healthcare (HealthTech, medical devices, specialized care services)
    Renewable Energy & Sustainability
    Consumer Goods & Services (High-end, niche markets)
    Industrial Technology (Advanced manufacturing, robotics)
    3. Sectors to Avoid
    High-risk sectors with regulatory volatility (e.g., gambling, tobacco)
    Highly cyclical industries (commodities, construction without specialized niches)
    Industries with significant ethical and reputational risks
    4. Geographical Scope
    Primary: North America and Europe
    Secondary: APAC region (especially high-growth markets)
    5. ESG (Environmental, Social, Governance) Criteria
    Environmental: Active sustainability initiatives, improved energy efficiency, measurable carbon reduction efforts.
    Social: Positive workplace culture, diversity, community involvement, fair labor practices.
    Governance: Strong governance structures, ethical practices, clear accountability, comprehensive risk management.
    6. Industry Focus: Services vs. Manufacturing
    Services: Tech-enabled services, professional and business services, healthcare services, subscription-based models.
    Manufacturing: Advanced manufacturing, automation, precision engineering, sustainable practices.
    7. Management Criteria
    Proven leadership team
    Demonstrated operational excellence
    Openness to strategic collaboration
    Strong incentive alignment
    8. Due Diligence Focus Areas
    Financial health and historical performance
    Regulatory and legal compliance
    Technological and operational scalability
    Customer relationship quality and retention
    Market competitiveness and market share
    9. Key Performance Indicators (KPIs)
    Revenue Growth Rate
    EBITDA Margin
    Customer Retention Rate
    Market Share
    ESG Rating and Compliance Metrics
    10. Red Flags
    Significant regulatory or legal issues
    Poor ESG compliance history
    Excessive customer concentration
    Inconsistent financial reporting
    High employee turnover at senior management level
    """
    return await extract_criteria_from_text(investment_criteria_text_block)