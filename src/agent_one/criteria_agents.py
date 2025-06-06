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

### DATA CLASSES ###

class InputCriterion(BaseModel):
    criterion_id: str = Field(..., description="Unique identifier for the criterion.")
    text: str = Field(..., description="The full text description of the criterion.")
    nature: Literal["requirement", "prohibition", "descriptor"] = Field(description="The nature of the criterion: 'requirement' (must be met), 'prohibition' (must not be met), 'descriptor' (neutral information).")
    analyst_notes: List[str]  = Field(..., description="Provides analyst notes that help to assess a criteria.")

class EvidenceSnippet(BaseModel):
    text: str = Field(..., description="The text of the snippet.")

### JUDGE AGENT ###    
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

##################### YES AGENT #####################
class MetricExceedanceDetail(BaseModel):
    """Describes how a quantitative criterion was positively exceeded."""
    metric_description: str = Field(
        ...,
        description="Description of the metric from the criterion (e.g., 'Annual Revenue', 'Customer Retention Rate')."
    )
    criterion_target: str = Field(
        ...,
        description="The target value or condition specified in the criterion (e.g., '> $50M', 'Minimum 90%')."
    )
    actual_value_reported: str = Field(
        ...,
        description="The actual value reported in the IM (e.g., '$70M', '95%')."
    )
    degree_of_exceedance: str = Field(
        ...,
        description="A clear description of how much the target was exceeded (e.g., 'Exceeded by $20M (40%)', 'Surpassed target by 5 percentage points')."
    )

class RelatedStrengthOrOpportunity(BaseModel):
    """A related strength or opportunity highlighted in the IM."""
    description: str = Field(
        ...,
        description="Description of the associated strength, competitive advantage, or opportunity."
    )
    supporting_evidence: List[EvidenceSnippet] = Field(
        ...,
        description="Evidence snippet(s) from the IM supporting this."
    )

class NotedCaveat(BaseModel):
    """A minor caveat, nuance, or limitation noted in the IM, despite the criterion being met."""
    description: str = Field(
        ...,
        description="Description of the minor caveat or qualifying remark."
    )
    supporting_evidence: List[EvidenceSnippet] = Field(
        ...,
        description="Evidence snippet(s) from the IM supporting this."
    )

class PositiveConfirmationAnalysisOutput(BaseModel):
    """Detailed analysis for an investment criterion definitively marked as 'yes'."""
    analysis_type: Literal["positive_confirmation_analysis"] = Field(description="Type of analysis performed.")
    criterion_id: str = Field(
        ...,
        description="An identifier for the specific criterion being analyzed."
    )
    criterion_text: str = Field(
        ...,
        description="The full text of the investment criterion being analyzed."
    )
    summary_of_positive_finding: str = Field(
        ...,
        description="A concise summary confirming the criterion is met and highlighting the core reason, based on JudgeAgent's assessment."
    )
    conclusive_evidence: List[EvidenceSnippet] = Field(
        ...,
        description="Key evidence snippet(s) from the IM that definitively prove the criterion is met."
    )
    quantitative_target_exceedances: List[MetricExceedanceDetail] = Field(
        default_factory=list,
        description="Details of how any quantitative targets within the criterion were exceeded. Empty if not applicable or not exceeded."
    )
    identified_related_strengths_opportunities: List[RelatedStrengthOrOpportunity] = Field(
        default_factory=list,
        description="Associated strengths, competitive advantages, or opportunities mentioned in the IM that are linked to this met criterion."
    )
    observed_minor_caveats_nuances: List[NotedCaveat] = Field(
        default_factory=list,
        description="Any minor caveats, limitations, or qualifying remarks mentioned in the IM, despite the overall 'yes' assessment."
    )
    overall_positive_impact_summary: str = Field(
        ...,
        description="A brief concluding statement on the positive impact or significance of this criterion being successfully met."
    )

##################### NO AGENT #####################
class GapAnalysis(BaseModel):
    """Describes the quantitative or qualitative gap for a failed criterion."""
    description: str = Field(..., description="A clear description of what the gap represents (e.g., 'ARR Shortfall', 'Customer Concentration Overage', 'Feature Not Present').")
    criterion_target_value: str = Field(..., description="The target value or state defined by the investment criterion.")
    actual_value_in_memorandum: str = Field(..., description="The actual value or state found in the information memorandum.")
    gap_value: str = Field(..., description="The calculated or described difference (e.g., '$2M', '5%', 'Feature X is missing').")
    unit: Optional[str] = Field(None, description="The unit of measurement for the gap, if applicable (e.g., 'USD', '%').")

class PotentialMitigant(BaseModel):
    """Describes a potential mitigating factor mentioned in the memorandum for a failed criterion."""
    description: str = Field(..., description="Explanation of the potential mitigating factor.")
    evidence_snippets: List[EvidenceSnippet] = Field(..., description="Supporting evidence snippets from the memorandum for this mitigant.")

class NoCriterionAnalysisOutput(BaseModel):
    """Detailed analysis for an investment criterion that has been marked as 'No (Fail)'."""
    criterion_id: str = Field(..., description="An identifier for the specific criterion being analyzed (e.g., 'CRITERION_001_ARR').")
    severity: Literal["Hard No", "Soft No", "Undefined"] = Field(
        ...,
        description="Assessment of the failure's severity: 'Hard No' (potential deal-breaker), 'Soft No' (negotiable/addressable), or 'Undefined'."
    )
    severity_reasoning: str = Field(..., description="Justification for the assigned severity level.")
    gap_analysis: Optional[GapAnalysis] = Field(
        None,
        description="Quantitative or qualitative analysis of the gap between the criterion and the memorandum's content. Null if not applicable."
    )
    potential_mitigants: List[PotentialMitigant] = Field(
        default_factory=list,
        description="List of potential mitigating factors mentioned in the memorandum, if any."
    )
    implication_statement: str = Field(
        ...,
        description="A brief statement on the potential business implication of this specific failure."
    )
    suggested_follow_up_questions: List[str] = Field(
        default_factory=list,
        description="Specific questions to ask the seller or for further due diligence related to this failed criterion."
    )

##################### UNCLEAR AGENT #####################
class IdentifiedInformationGap(BaseModel):
    """Describes a specific piece of missing or insufficient information related to an unclear criterion."""
    gap_description: str = Field(
        ...,
        description="Detailed description of the specific information that is missing, vague, or insufficient to assess a part of the criterion."
    )
    reason_for_gap: str = Field(
        ...,
        description="Explanation of why the available information in the IM excerpts leads to this gap (e.g., 'Not mentioned', 'Data provided is out of scope/period', 'Description is too high-level')."
    )
    potential_information_sources: List[str] = Field(
        default_factory=list,
        description="Suggested typical sources where this information might be found (e.g., 'Management Q&A', 'Virtual Data Room: Financials Folder', 'Customer Contracts')."
    )

class UnclearCriterionAnalysisOutput(BaseModel):
    """Detailed analysis for an investment criterion that has been marked as 'Unclear' by the JudgeAgent."""
    criterion_id: str = Field(..., description="An identifier for the specific criterion being analyzed (e.g., 'CRITERION_005_ESG_Compliance').")
    summary_of_why_unclear: str = Field(
        ...,
        description="A concise summary, based on the JudgeAgent's initial assessment and IM excerpts, explaining the core reason for the 'Unclear' status."
    )
    identified_gaps: List[IdentifiedInformationGap] = Field(
        ...,
        description="A list of specific information gaps identified that prevent a clear assessment of this criterion."
    )
    potential_implications_of_gaps: str = Field(
        ...,
        description="Brief assessment of potential risks, hidden issues, or missed insights due to the identified information gaps remaining unresolved."
    )
    suggested_due_diligence_questions: List[str] = Field(
        ...,
        description="Specific, actionable questions to ask the seller or to guide further investigation to fill the identified information gaps."
    )
    overall_clarification_priority: Optional[Literal["High", "Medium", "Low"]] = Field(
        None,
        description="Optional: Estimated priority for clarifying this criterion based on its likely impact on the investment decision."
    )

##################### PROBABLE AGENT #####################
class InferentialEvidence(BaseModel):
    """Describes a piece of evidence and the inference drawn from it for a probable assessment."""
    evidence_snippets: List[EvidenceSnippet] = Field(
        ...,
        description="Key evidence snippet(s) from the IM that support this part of the inference."
    )
    inferred_conclusion_from_evidence: str = Field(
        ...,
        description="The specific point inferred from this evidence that contributes to the 'likely' verdict."
    )
    reasoning_for_inference_and_uncertainty: str = Field(
        ...,
        description="Explanation of how this evidence leads to the inference, and why it's not conclusive."
    )

class ProbableOutcomeAnalysisOutput(BaseModel):
    """Detailed analysis for an investment criterion marked as 'likely yes' or 'likely no'."""
    analysis_type: Literal["probable_outcome_analysis"] = Field(description="Type of analysis performed.")
    criterion_id: str = Field(
        ...,
        description="An identifier for the specific criterion being analyzed."
    )
    criterion_text: str = Field(
        ...,
        description="The full text of the investment criterion being analyzed."
    )
    original_probable_verdict: Literal["likely yes", "likely no"] = Field(
        ...,
        description="The original 'likely yes' or 'likely no' verdict from the JudgeAgent."
    )
    summary_of_probabilistic_assessment: str = Field(
        ...,
        description="A high-level summary explaining why the assessment is 'likely' rather than definitive, based on the JudgeAgent's reasoning."
    )
    detailed_inferential_evidence: List[InferentialEvidence] = Field(
        ...,
        description="A breakdown of the key pieces of evidence and the inferences drawn from them."
    )
    key_assumptions_made: List[str] = Field(
        default_factory=list,
        description="Any significant assumptions made to reach the probable verdict (e.g., 'Assuming 'industry-leading' implies top quartile performance')."
    )
    information_to_achieve_certainty: List[str] = Field(
        ...,
        description="Specific pieces of information or clarifications needed to convert the 'likely' status to a definitive 'yes' or 'no'."
    )
    suggested_verification_actions: List[str] = Field(
        ...,
        description="Targeted questions for management or due diligence actions to obtain the information needed for certainty."
    )
    potential_implications_upon_confirmation: str = Field(
        ...,
        description="The potential positive (if 'likely yes' confirmed as 'yes') or negative (if 'likely no' confirmed as 'no') implications for the investment."
    )

##################### CONTRADICTORY AGENT #####################
class ConflictingInformationPoint(BaseModel):
    """Represents one piece of information involved in a contradiction."""
    statement_summary: str = Field(
        ...,
        description="A summary of the claim or data point derived from the evidence."
    )
    supporting_evidence: List[EvidenceSnippet] = Field(
        ...,
        description="List of evidence snippets from the IM that support this statement."
    )
    interpretation_of_statement: str = Field(
        ...,
        description="How this statement, if true, would relate to or assess the investment criterion."
    )

class ContradictionDetail(BaseModel):
    """Describes a specific point of contradiction between two or more pieces of information."""
    conflicting_aspect: str = Field(
        ...,
        description="A brief description of the specific aspect or data point where the contradiction lies (e.g., 'Reported FY23 Revenue', 'Number of Employees', 'Market Share Percentage')."
    )
    information_A: ConflictingInformationPoint = Field(
        ...,
        description="The first piece of conflicting information."
    )
    information_B: ConflictingInformationPoint = Field(
        ...,
        description="The second piece of conflicting information (can be extended if more than two points conflict)."
    )
    explanation_of_contradiction: str = Field(
        ...,
        description="Detailed explanation of how Information A and Information B contradict each other in relation to the investment criterion."
    )

class ContradictoryCriterionAnalysisOutput(BaseModel):
    """Detailed analysis for an investment criterion that has been marked as 'Contradictory' by the JudgeAgent."""
    analysis_type: Literal["contradictory_analysis"] = Field(description="Type of analysis performed.")
    criterion_id: str = Field(
        ...,
        description="An identifier for the specific criterion being analyzed."
    )
    criterion_text: str = Field( # Added for context in the output
        ...,
        description="The full text of the investment criterion being analyzed."
    )
    overall_summary_of_issue: str = Field(
        ...,
        description="A high-level summary stating that contradictory information was found for this criterion and its general nature."
    )
    identified_contradictions: List[ContradictionDetail] = Field(
        ...,
        description="A list detailing each specific point of contradiction found relevant to this criterion."
    )
    potential_impact_if_unresolved: str = Field(
        ...,
        description="Assessment of the potential risks, misjudgments, or impact on decision-making if these contradictions are not resolved."
    )
    suggested_resolution_actions: List[str] = Field(
        ...,
        description="Specific questions for management or actions for the due diligence team to take to clarify the discrepancies and resolve the contradictions."
    )

    initial_hypothesis_on_correctness: Optional[str] = Field(
        None,
        description="A cautious hypothesis on which piece of information might be more reliable, if inferable, clearly stating its speculative nature."
    )

AllAnalysisOutputs = Union[JudgeOutput, NoCriterionAnalysisOutput, UnclearCriterionAnalysisOutput, ProbableOutcomeAnalysisOutput, ContradictoryCriterionAnalysisOutput, PositiveConfirmationAnalysisOutput]

## FOR EXPORT AND REPORTING
class ReportMetadata(BaseModel):
    report_generated_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of when the report was generated (UTC).")
    criteria_source_file: Optional[str] = Field(None, description="Name of the JSON criteria file used for this evaluation.")
    information_memorandum_source: Optional[str] = Field(None, description="Identifier for the Information Memorandum processed (e.g., filename, ID).")
    total_criteria_evaluated: int = Field(..., description="Total number of criteria attempted for evaluation.")
    evaluations_completed_successfully: int = Field(..., description="Number of criteria for which an analysis object was successfully generated.")
    evaluations_failed_or_incomplete: int = Field(..., description="Number of criteria for which evaluation failed or was incomplete.")

class CriterionEvaluationExportItem(BaseModel):
    criterion_id: str = Field(..., description="The unique identifier of the criterion.")
    criterion_text: str = Field(..., description="The full text of the criterion.")
    evaluation_result: Optional[AllAnalysisOutputs] = Field( # This is the Union type
        None, 
        description="The detailed analysis output for this criterion. Null if evaluation failed for this item."
    )

class FullReportExport(BaseModel):
    report_metadata: ReportMetadata = Field(..., description="Metadata about the evaluation report.")
    criterion_evaluations: List[CriterionEvaluationExportItem] = Field(
        ..., 
        description="A list containing the evaluation results for each criterion."
    )

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

no_agent = Agent(
    name="NoAgent",
    model=critique,
    model_settings=model_settings,
    output_type=NoCriterionAnalysisOutput,
    instructions=load_prompt("criteria.no_agent"),
    )

unclear_agent = Agent(
    name="UnclearAgent",
    model=critique,
    model_settings=model_settings,
    output_type=UnclearCriterionAnalysisOutput,
    instructions=load_prompt("criteria.unclear"),
    )

likely_agent = Agent(
    name="LikelyAgent",
    model=critique,
    model_settings=model_settings,
    output_type=ProbableOutcomeAnalysisOutput,
    instructions=load_prompt("criteria.likely"),
    )

yes_agent = Agent(
    name="YesAgent",
    model=critique,
    model_settings=model_settings,
    output_type=PositiveConfirmationAnalysisOutput,
    instructions=load_prompt("criteria.yes_agent"),
    )

contradictory_agent = Agent(
    name="ContradictoryAgent",
    model=critique,
    model_settings=model_settings,
    output_type=ContradictoryCriterionAnalysisOutput,
    instructions=load_prompt("criteria.contradictory"),
    )

### EXTRACTION AGENT ###
class ExtractedCriterion(BaseModel):
    criteria_id: str = Field(..., description="The ID of the extracted criterion.")
    criteria_text: str = Field(..., description="The text of the extracted criterion.")
    criteria_type: Literal["quantitative", "qualitative"] = Field(..., description="The type of the extracted criterion.")

class ExtractedCriteria(BaseModel):
    criteria: List[ExtractedCriterion] = Field(..., description="List of extracted criteria.")

criteria_extraction_agent = Agent(
    name="Criteria Extraction Agent",
    model=general_purpose,
    model_settings=model_settings,
    output_type=ExtractedCriteria,
    instructions=load_prompt("criteria.extract"),
    )