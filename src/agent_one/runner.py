from agents import Runner

from .agent import summary_agent, followup_agent
from schemas import SummaryOutput, FollowupOutput


class MultiAgentOutput(FollowupOutput):
    summary: str


async def run(text: str) -> MultiAgentOutput:
    original_summary_instructions = summary_agent.instructions
    summary_agent.instructions = original_summary_instructions.replace("{{text}}", text)
    run_summary = await Runner.run(summary_agent, input="")
    summary_agent.instructions = original_summary_instructions

    summary_text = run_summary.final_output.summary

    original_followup_instructions = followup_agent.instructions
    followup_agent.instructions = original_followup_instructions.replace("{{summary}}", summary_text)
    run_followup = await Runner.run(followup_agent, input="")
    followup_agent.instructions = original_followup_instructions

    return MultiAgentOutput(
        question=run_followup.final_output.question,
        kind=run_followup.final_output.kind,
        summary=summary_text,
    )
