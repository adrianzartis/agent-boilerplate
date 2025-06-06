import json
import os
from datetime import datetime

from fastapi import APIRouter, HTTPException, status, Query, Form
from im_criteria.criteria_agents import (
    ExtractedCriteria,
    AllAnalysisOutputs,
    ReportMetadata,
    CriterionEvaluationExportItem,
    FullReportExport
)
from pydantic import BaseModel
from typing import List, Optional

from im_criteria.criteria_runner import run_criteria_evaluations_parallel
from im_criteria.criteria_services import get_extracted_criteria, load_criteria_from_json

REPORTS_DIR = "evaluation_reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

router = APIRouter(
    prefix="/criteria",
    tags=["investment_criteria"],
)

class BatchEvaluationResponse(BaseModel):
    evaluations: List[Optional[AllAnalysisOutputs]]
    total_criteria_processed: int
    successful_evaluations: int
    failed_evaluations: int
    report_filename: Optional[str] = None
    message: Optional[str] = None


def register_endpoints():
    @router.post("/evaluate-batch", response_model=BatchEvaluationResponse)
    async def evaluate_memo_batch_endpoint(
        criteria_file_name: str = Form("investment_criteria.json", description="Name of the JSON file containing the criteria (must be accessible by the server)."),
        max_concurrent_evaluations: int = Query(1, ge=1, description="Maximum number of criteria to evaluate in parallel."),
        current_year: int = Query(description="Determines current year for the evaluation process")

    ):
        """
        Evaluates an Information Memorandum against a list of criteria loaded from a specified JSON file.
        Criteria are processed in parallel up to the specified concurrency limit.
        """
        report_filename = None
        try:
            criteria_list = load_criteria_from_json(criteria_file_name, str(current_year))
            if not criteria_list:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"No criteria found in '{criteria_file_name}' or file is empty."
                )

            results = await run_criteria_evaluations_parallel(
                criteria_list,
                str(current_year),
                max_concurrent_tasks=max_concurrent_evaluations
            )
            
            successful_count = sum(1 for r in results if r is not None)
            failed_count = len(results) - successful_count

            # JSON Export
            report_metadata = ReportMetadata(
                criteria_source_file=criteria_file_name,
                information_memorandum_source="Information Memorandum",
                total_criteria_evaluated=len(criteria_list),
                evaluations_completed_successfully=successful_count,
                evaluations_failed_or_incomplete=failed_count
            )

            criterion_eval_exports: List[CriterionEvaluationExportItem] = []
            for i, result_item in enumerate(results):
                criterion_input = criteria_list[i]
                criterion_eval_exports.append(
                    CriterionEvaluationExportItem(
                        criterion_id=criterion_input.criterion_id,
                        criterion_text=criterion_input.text,
                        evaluation_result=result_item
                    )
                )
            
            full_report = FullReportExport(
                report_metadata=report_metadata,
                criterion_evaluations=criterion_eval_exports
            )

            timestamp_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            base_report_name = os.path.splitext(os.path.basename(criteria_file_name))[0]
            report_filename = f"report_{base_report_name}_{timestamp_str}.json"
            report_filepath = os.path.join(REPORTS_DIR, report_filename)

            with open(report_filepath, 'w') as f:
                json.dump(full_report.model_dump(mode='json'), f, indent=2) 
            
            return BatchEvaluationResponse(
                evaluations=results,
                total_criteria_processed=len(criteria_list),
                successful_evaluations=successful_count,
                failed_evaluations=failed_count,
                message="Batch evaluation completed."
            )

        except ValueError as ve:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
        except FileNotFoundError:
             raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Criteria file '{criteria_file_name}' not found.")
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"An unexpected error occurred during batch evaluation: {str(e)}"
            )


    @router.post("/extract", response_model=ExtractedCriteria, status_code=status.HTTP_200_OK)
    async def extract_predefined_criteria_endpoint() -> ExtractedCriteria:
        """
        Extracts and structures investment criteria from a predefined internal text block.
        """
        try:
            result = await get_extracted_criteria()
            if result is None:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Criteria extraction process failed internally (e.g., prompt issue)."
                )
            return result
        except ValueError as ve:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Criteria extraction error: {str(ve)}")
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An unexpected error occurred during criteria extraction: {str(e)}")


    return router