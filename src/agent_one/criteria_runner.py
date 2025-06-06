import json
import logging
import requests
import asyncio

from .criteria_agents import (
    judge_agent, JudgeOutput,
    no_agent, NoCriterionAnalysisOutput,
    unclear_agent, UnclearCriterionAnalysisOutput,
    likely_agent, ProbableOutcomeAnalysisOutput,
    yes_agent, PositiveConfirmationAnalysisOutput,
    contradictory_agent, ContradictoryCriterionAnalysisOutput,
    InputCriterion,
    AllAnalysisOutputs
    )
from .criteria_services import load_criteria_from_json

from typing import List, Optional
from agents import Runner
from config import config
from prompts import format_prompt, load_prompt

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

def summary_retriever(query: str):
    api_url = config.get('retriever.search_url')
    if not api_url:
        logger.error("Retriever search_url not configured in config.")
        return "Error: Retriever search_url not configured."

    payload = {
        "query": query,
        "mode": "hybrid",
        "only_need_context": False,
        "only_need_prompt": False,
        "top_k": 30,
        "max_token_for_text_unit": 10000,
        "response_type": "Multiple Paragraph",
        "max_token_for_global_context": 10000,
        "max_token_for_local_context": 10000
    }
    try:
        response = requests.post(api_url, json=payload)
        response.raise_for_status()
        result = response.json()
        if "response" not in result:
            logger.error(f"Retriever Error: 'response' key not found. Query: '{query}'")
            return "Error: Retriever did not return a 'response'."
        return result["response"].replace("\u20ac", "EUR").replace("\u2248", "=").replace("≈", "=")
    except requests.exceptions.RequestException as e:
        logger.error(f"Retriever API request failed for query '{query}': {e}")
        return f"Error: Retriever API request failed - {e}"
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON response from retriever for query '{query}': {e}")
        return f"Error: Failed to decode JSON from retriever - {e}"

async def run_judge_agent(information_memorandum: str, criterion_text: str, criterion_notes: List[str], criterion_id: str) -> Optional[JudgeOutput]:
    
    judge_agent_variables = {
        "information_memorandum": information_memorandum,
        "criteria": criterion_text,
        "notes": "\n".join(["* " + note for note in criterion_notes])
        }

    formatted_judge_prompt = format_prompt("criteria.judge", judge_agent_variables)
    if formatted_judge_prompt is None:
        logger.error("Failed to load or format 'judge' prompt.")
        return None
    
    judge_agent.instructions = formatted_judge_prompt
    try:
        run_result = await Runner.run(judge_agent, input="")
    except Exception as e:
        logger.error(f"JudgeAgent failed for criterion ID {criterion_id}: {e}")
        return None
    
    if run_result and run_result.final_output and isinstance(run_result.final_output, JudgeOutput):
        run_result.final_output.criterion_id = criterion_id
        return run_result.final_output
    else:
        error_message = (
            f"JudgeAgent for criterion ID {criterion_id} returned unexpected output or no output. "
            f"Type: {type(run_result.final_output) if run_result and run_result.final_output else 'N/A'}. "
            f"Output: {run_result.final_output if run_result else 'N/A'}"
        )
        logger.error(error_message)
        raise ValueError(error_message)

async def run_yes_agent(information_memorandum: str, criteria: str, judge_output: JudgeOutput) -> Optional[PositiveConfirmationAnalysisOutput]:


    yes_agent_variables = {
        "criterion_text": criteria,
        "criterion_id": judge_output.criterion_id,
        "judge_reasoning_for_yes": judge_output.reasoning,
        "judge_evidence_snippets_yes": judge_output.evidence_snippets,
        "relevant_im_excerpts": information_memorandum
        }
    
    formatted_yes_prompt = format_prompt("criteria.yes_agent", yes_agent_variables)
    
    if formatted_yes_prompt is None:
        logger.error("Failed to load or format 'yes_agent' prompt.")
        return None
    
    yes_agent.instructions = formatted_yes_prompt
    
    run_result = await Runner.run(yes_agent, input="")
    
    if isinstance(run_result.final_output, PositiveConfirmationAnalysisOutput):
        return run_result.final_output
    else:
        raise ValueError(f"YesAgent returned unexpected output type: {type(run_result.final_output)}")

async def run_unclear_agent(information_memorandum: str, criteria: str, judge_output: JudgeOutput) -> Optional[UnclearCriterionAnalysisOutput]:

    unclear_agent_variables = {
        "criterion_id": judge_output.criterion_id,
        "criterion_text": criteria,
        "judge_agent_reasoning_for_unclear": judge_output.reasoning,
        "judge_agent_unclear_evidence_snippets": judge_output.evidence_snippets,
        "relevant_im_excerpts": information_memorandum
        }
    
    formatted_unclear_prompt = format_prompt("criteria.unclear", unclear_agent_variables)
    
    if formatted_unclear_prompt is None:
        logger.error("Failed to load or format 'unclear' prompt.")
        return None
    
    unclear_agent.instructions = formatted_unclear_prompt
    
    run_result = await Runner.run(unclear_agent, input="")
    
    if isinstance(run_result.final_output, UnclearCriterionAnalysisOutput):
        return run_result.final_output
    else:
        raise ValueError(f"UnclearAgent returned unexpected output type: {type(run_result.final_output)}")

async def run_no_agent(information_memorandum: str, criteria: str, judge_output: JudgeOutput) -> Optional[NoCriterionAnalysisOutput]:
    
    no_agent_variables = {
        "criterion_text": criteria,
        "criterion_id": judge_output.criterion_id,
        "judge_reasoning": judge_output.reasoning,
        "judge_evidence_snippets": judge_output.evidence_snippets,
        "relevant_im_excerpts": information_memorandum
        }

    formatted_no_prompt = format_prompt("criteria.no_agent", no_agent_variables)
    
    if formatted_no_prompt is None:
        logger.error("Failed to load or format 'no_agent' prompt.")
        return None
    
    no_agent.instructions = formatted_no_prompt
    
    run_result = await Runner.run(no_agent, input="")
    
    if isinstance(run_result.final_output, NoCriterionAnalysisOutput):
        return run_result.final_output
    else:
        raise ValueError(f"NoAgent returned unexpected output type: {type(run_result.final_output)}")

async def run_likely_agent(information_memorandum: str, criteria: str, judge_output: JudgeOutput) -> Optional[ProbableOutcomeAnalysisOutput]:
    

    likely_agent_variables = {
        "criterion_text": criteria,
        "criterion_id": judge_output.criterion_id,
        "judge_probable_verdict": judge_output.verdict,
        "judge_reasoning_for_probable": judge_output.reasoning,
        "judge_evidence_snippets_probable": judge_output.evidence_snippets,
        "relevant_im_excerpts": information_memorandum
        }
    
    formatted_likely_prompt = format_prompt("criteria.likely", likely_agent_variables)
    
    if formatted_likely_prompt is None:
        logger.error("Failed to load or format 'likely' prompt.")
        return None
    
    likely_agent.instructions = formatted_likely_prompt
    
    run_result = await Runner.run(likely_agent, input="")
    
    if isinstance(run_result.final_output, ProbableOutcomeAnalysisOutput):
        return run_result.final_output
    else:
        raise ValueError(f"LikelyAgent returned unexpected output type: {type(run_result.final_output)}")

async def run_contradictory_agent(information_memorandum: str, criteria: str, judge_output: JudgeOutput) -> ContradictoryCriterionAnalysisOutput:
   
    contradictory_agent_variables = {
        "criterion_id": judge_output.criterion_id,
        "criterion_text": criteria,
        "judge_agent_reasoning_for_contradictory": judge_output.reasoning,
        "judge_agent_contradictory_evidence_snippets": judge_output.evidence_snippets,
        "relevant_im_excerpts": information_memorandum
        }
    
    formatted_contradictory_prompt = format_prompt("criteria.contradictory", contradictory_agent_variables)
    
    if formatted_contradictory_prompt is None:
        logger.error("Failed to load or format 'contradictory' prompt.")
        return None
    
    contradictory_agent.instructions = formatted_contradictory_prompt
    
    run_result = await Runner.run(contradictory_agent, input="")
    
    if isinstance(run_result.final_output, ContradictoryCriterionAnalysisOutput):
        return run_result.final_output
    else:
        raise ValueError(f"ContradictoryAgent returned unexpected output type: {type(run_result.final_output)}")


# Core Orchestration for a single criterion
async def evaluate_single_criterion(
    criterion: 'InputCriterion',
    current_year: str,
    full_im_text: Optional[str] = None
) -> Optional[AllAnalysisOutputs]:
    """
    Evaluates a single criterion against an Information Memorandum.
    It retrieves context specific to the criterion and then runs the judge & specialized agents.
    """
    logger.info(f"Starting evaluation for criterion ID: {criterion.criterion_id} - \"{criterion.text[:50]}...\"")
    
    summary_variables = {
        "current_year": current_year
    }
    # Step 1: Retrieve context specific to this criterion
    retrieved_im_context = summary_retriever(format_prompt('criteria.summary_report', summary_variables)) 
    
    if isinstance(retrieved_im_context, str) and retrieved_im_context.startswith("Error:"):
        logger.error(f"Retriever failed for criterion ID {criterion.criterion_id}: {retrieved_im_context}")
        return None 

    # Step 2: Run Judge Agent
    judge_output = await run_judge_agent(retrieved_im_context, criterion.text, criterion.analyst_notes, criterion.criterion_id)
    
    if judge_output is None:
        logger.error(f"JudgeAgent failed for criterion ID {criterion.criterion_id}.")
        return None

    logger.info(f"JudgeAgent for criterion ID '{judge_output.criterion_id}' completed. Verdict: {judge_output.verdict}")

    # Step 3: Dispatch to specialized agent based on verdict
    dispatch_verdict = judge_output.verdict

    # Prohibition criteria are reversed TODO: check if the specialized agents handle this semantic inversion correctly
    if criterion.nature == "prohibition":
        if judge_output.verdict == "yes":       
            dispatch_verdict = "no"             
        elif judge_output.verdict == "no":      
            dispatch_verdict = "yes"            
        elif judge_output.verdict == "likely yes": 
            dispatch_verdict = "likely no"      
        elif judge_output.verdict == "likely no": 
            dispatch_verdict = "likely yes"     
    
    logger.info(f"Dispatching criterion ID '{judge_output.criterion_id}' (Nature: '{criterion.nature}', Judge Verdict: '{judge_output.verdict}') with effective dispatch verdict for agent routing: '{dispatch_verdict}'")

    analysis_output: Optional[AllAnalysisOutputs] = None

    if dispatch_verdict == "no":
        analysis_output = await run_no_agent(retrieved_im_context, criterion.text, judge_output)
        if analysis_output: logger.info(f"NoAgent for '{judge_output.criterion_id}'. Severity: {analysis_output.severity}")
    elif dispatch_verdict == "unclear":
        analysis_output = await run_unclear_agent(retrieved_im_context, criterion.text, judge_output)
        if analysis_output: logger.info(f"UnclearAgent for '{judge_output.criterion_id}'. Priority: {analysis_output.overall_clarification_priority}")
    elif dispatch_verdict in ["likely yes", "likely no"]:
        analysis_output = await run_likely_agent(retrieved_im_context, criterion.text, judge_output)
        if analysis_output: logger.info(f"LikelyAgent for '{judge_output.criterion_id}'. Original judge verdict: '{judge_output.verdict}', Stored as: {analysis_output.original_probable_verdict}")
    elif dispatch_verdict == "yes":
        analysis_output = await run_yes_agent(retrieved_im_context, criterion.text, judge_output)
        if analysis_output: logger.info(f"YesAgent for '{judge_output.criterion_id}'. Impact: {analysis_output.overall_positive_impact_summary[:50]}...")
    elif dispatch_verdict == "contradictory":
        analysis_output = await run_contradictory_agent(retrieved_im_context, criterion.text, judge_output)
        if analysis_output: logger.info(f"ContradictoryAgent for '{judge_output.criterion_id}'.")
    else:
        logger.error(f"Unknown dispatch_verdict '{dispatch_verdict}' for criterion ID '{judge_output.criterion_id}'. Returning JudgeOutput.")
        return judge_output

    if analysis_output is None:
        logger.warning(f"Specialized agent for dispatch_verdict '{dispatch_verdict}' returned None for criterion ID '{judge_output.criterion_id}'. Returning JudgeOutput as fallback.")
        return judge_output 

    return analysis_output

# Parallel Execution
async def _evaluate_single_criterion_with_semaphore(
    criterion: InputCriterion,
    current_year: str,
    semaphore: asyncio.Semaphore
) -> Optional[AllAnalysisOutputs]:
    async with semaphore:
        logger.debug(f"Semaphore acquired for criterion ID: {criterion.criterion_id}")
        result = await evaluate_single_criterion(criterion, current_year) 
        logger.debug(f"Semaphore released for criterion ID: {criterion.criterion_id}")
        return result

async def run_criteria_evaluations_parallel(
    criteria_list: List[InputCriterion],
    current_year: str,
    max_concurrent_tasks: int = 1
) -> List[Optional[AllAnalysisOutputs]]:
    if not criteria_list:
        logger.info("No criteria provided for parallel evaluation.")
        return []

    if max_concurrent_tasks < 1:
        logger.warning("max_concurrent_tasks cannot be less than 1. Setting to 1.")
        max_concurrent_tasks = 1

    logger.info(f"Starting parallel evaluation for {len(criteria_list)} criteria with max concurrency: {max_concurrent_tasks}")

    semaphore = asyncio.Semaphore(max_concurrent_tasks)

    tasks = [
        _evaluate_single_criterion_with_semaphore(criterion, current_year, semaphore)
        for criterion in criteria_list
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    processed_results: List[Optional[AllAnalysisOutputs]] = []
    for i, res in enumerate(results):
        criterion_id_for_log = criteria_list[i].criterion_id
        if isinstance(res, Exception):
            logger.error(f"Error evaluating criterion ID {criterion_id_for_log}: {res}", exc_info=res)
            processed_results.append(None) 
        elif res is None:
            logger.warning(f"Evaluation for criterion ID {criterion_id_for_log} resulted in None.")
            processed_results.append(None)
        else:
            processed_results.append(res)

    logger.info(f"Parallel evaluation completed for {len(criteria_list)} criteria.")
    return processed_results


async def run_full_evaluation_from_json(
    criteria_json_path: str = "src/im_criteria/investment_criteria.json",
    current_year: str = "2019",
    max_concurrent_tasks: int = 1
):
    """
    Main orchestration function:
    1. Loads criteria from a JSON file.
    2. Evaluates all criteria against an IM (retrieved per criterion) in parallel.
    3. Logs the results.
    """
    logger.info(f"Starting full evaluation using criteria from: {criteria_json_path}")
    
    try:
        criteria_list = load_criteria_from_json(criteria_json_path, current_year)
    except ValueError as e:
        logger.error(f"Failed to load criteria: {e}")
        return

    if not criteria_list:
        logger.warning("No criteria loaded. Aborting evaluation.")
        return
    
    all_results = await run_criteria_evaluations_parallel(
        criteria_list,
        current_year,
        max_concurrent_tasks=max_concurrent_tasks
    )

    logger.info("--- Full Evaluation Report ---")
    for i, result in enumerate(all_results):
        criterion = criteria_list[i]
        logger.info(f"\nCriterion ID: {criterion.criterion_id}")
        logger.info(f"Criterion Text: \"{criterion.text[:100]}...\"")
        if result:
            if hasattr(result, 'analysis_type'):
                logger.info(f"Analysis Type: {result.analysis_type}")
                if hasattr(result, 'verdict'):
                    logger.info(f"Verdict: {result.verdict}")
                    logger.info(f"Reasoning: {result.reasoning}")
                elif hasattr(result, 'severity'):
                    logger.info(f"Severity: {result.severity}")
                    logger.info(f"Implication: {result.implication_statement}")
                else:
                    logger.info(f"Details: {result.model_dump_json(indent=2)}")
            else:
                 logger.info(f"Result (raw): {result}")
        else:
            logger.warning("Evaluation for this criterion failed or returned no result.")
    logger.info("--- End of Full Evaluation Report ---")