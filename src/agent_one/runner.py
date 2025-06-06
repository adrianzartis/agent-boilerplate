from agents import Runner

from .agent import summary_agent, followup_agent
from schemas import SummaryOutput, FollowupOutput
import logging

logger = logging.getLogger(__name__)


class MultiAgentOutput(FollowupOutput):
    summary: str


async def run(text: str) -> MultiAgentOutput:
    original_summary_instructions = summary_agent.instructions
    summary_agent.instructions = original_summary_instructions.replace("{{text}}", text)
    logger.debug("SummaryAgent instructions: %s", summary_agent.instructions)
    run_summary = await Runner.run(summary_agent, input="")
    logger.debug("SummaryAgent output: %s", run_summary.final_output.model_dump_json())
    summary_agent.instructions = original_summary_instructions

    summary_text = run_summary.final_output.summary

    original_followup_instructions = followup_agent.instructions
    followup_agent.instructions = original_followup_instructions.replace("{{summary}}", summary_text)
    logger.debug("FollowupAgent instructions: %s", followup_agent.instructions)
    run_followup = await Runner.run(followup_agent, input="")
    logger.debug("FollowupAgent output: %s", run_followup.final_output.model_dump_json())
    followup_agent.instructions = original_followup_instructions

    return MultiAgentOutput(
        question=run_followup.final_output.question,
        kind=run_followup.final_output.kind,
        summary=summary_text,
    )
