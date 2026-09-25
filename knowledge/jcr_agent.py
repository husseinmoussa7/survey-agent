"""
JCR Journal Report Generation Agent

This script generates a research report based on OSF journal survey design and data.

Usage:
    python jcr_agent.py --paper_number 1
"""

import os
import argparse
import time
import glob
from pathlib import Path
from dotenv import load_dotenv
import openai
from openai import OpenAI
import json
import pandas as pd

# Load environment variables
load_dotenv("/Users/princess/Documents/RA/Field-Experiment-AI-Agent/.env")

# Configure OpenAI
client = OpenAI()

# Paths
BASE_DIR = Path("New Papers")

def find_data_files(directory):
    """Find all CSV and Excel files in a directory and its subdirectories."""
    csv_files = list(directory.glob("**/*.csv"))
    excel_files = list(directory.glob("**/*.xlsx")) + list(directory.glob("**/*.xls"))
    return csv_files, excel_files

def read_text_files(directory):
    """Find and read all text files in a directory and its subdirectories."""
    # Find text files (.txt, .md) and PDF files
    text_files = list(directory.glob("**/*.txt")) + list(directory.glob("**/*.md"))
    pdf_files = list(directory.glob("**/*.pdf"))
    
    survey_design = ""
    
    # Process text files (.txt and .md)
    for file_path in text_files:
        try:
            with open(file_path, "r", encoding='utf-8') as f:
                content = f.read()
                # If it seems to be a survey design document, use it
                if "survey" in file_path.name.lower() or "design" in file_path.name.lower():
                    survey_design += f"\n\n## Content from {file_path.name}:\n{content}"
                # Otherwise, just note that we found it
                else:
                    survey_design += f"\n\n## Found text file: {file_path.name}"
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    # Process PDF files
    for file_path in pdf_files:
        try:
            # Check if this looks like a methodology/survey design PDF
            if any(keyword in file_path.name.lower() for keyword in ["survey", "design", "method", "protocol", "procedure"]):
                # Try to extract text from PDF
                try:
                    import PyPDF2
                    with open(file_path, "rb") as f:
                        pdf_reader = PyPDF2.PdfReader(f)
                        pdf_text = ""
                        for page in pdf_reader.pages:
                            pdf_text += page.extract_text() + "\n"
                        
                        if pdf_text.strip():  # Only add if we extracted some text
                            survey_design += f"\n\n## Content from {file_path.name} (PDF):\n{pdf_text}"
                        else:
                            survey_design += f"\n\n## Found PDF file (text extraction failed): {file_path.name}"
                            
                except ImportError:
                    print(f"PyPDF2 not available. Install with: pip install PyPDF2")
                    survey_design += f"\n\n## Found PDF file (PyPDF2 not installed): {file_path.name}"
                except Exception as pdf_error:
                    print(f"Error extracting text from PDF {file_path}: {pdf_error}")
                    survey_design += f"\n\n## Found PDF file (text extraction failed): {file_path.name}"
            else:
                # Just note that we found a PDF file
                survey_design += f"\n\n## Found PDF file: {file_path.name}"
                
        except Exception as e:
            print(f"Error processing PDF {file_path}: {e}")
    
    return survey_design

def get_hypothesis():
    """Prompt user to input the research hypothesis."""
    print("\n=== Research Hypothesis Input ===")
    print("Please enter the research hypothesis for this paper.")
    print("This will help guide the analysis and report generation.")
    print("Enter your hypothesis below (type 'END' on a new line when finished):\n")
    
    lines = []
    while True:
        line = input()
        if line.strip().upper() == "END":
            break
        lines.append(line)
    
    hypothesis = "\n".join(lines)
    
    print("\nHypothesis received. This will be used to guide the report generation.")
    return hypothesis

def analyze_data(paper_number, hypothesis):
    """Analyze survey data."""
    paper_dir = BASE_DIR / f"Paper #{paper_number}"
    data_dir = paper_dir / "Data and Code"
    output_dir = paper_dir / "Output"
    
    # Create Output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Find all CSV and Excel files in the data directory and its subdirectories
    csv_files, excel_files = find_data_files(data_dir)
    
    # Read text files that might contain survey design information
    survey_design = read_text_files(data_dir)
    
    print(f"Found {len(csv_files)} CSV files and {len(excel_files)} Excel files")
    
    # Prepare data information for analysis
    data_info = []
    
    # Process CSV files
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            data_info.append({
                "file_name": file_path.name,
                "file_path": str(file_path.relative_to(paper_dir)),
                "columns": list(df.columns),
                "row_count": len(df),
                "preview": df.head(10).to_csv(index=False)
            })
        except Exception as e:
            print(f"Error reading CSV file {file_path}: {e}")
    
    # Process Excel files
    for file_path in excel_files:
        try:
            xl = pd.ExcelFile(file_path)
            sheets_info = []
            
            for sheet_name in xl.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                sheets_info.append({
                    "sheet_name": sheet_name,
                    "columns": list(df.columns),
                    "row_count": len(df),
                    "preview": df.head(10).to_csv(index=False)
                })
            
            data_info.append({
                "file_name": file_path.name,
                "file_path": str(file_path.relative_to(paper_dir)),
                "sheets": sheets_info
            })
        except Exception as e:
            print(f"Error reading Excel file {file_path}: {e}")
    
    # Create the prompt for data analysis
    data_description = ""
    for item in data_info:
        data_description += f"\n\n### File: {item['file_name']} ({item['file_path']})\n"
        if "sheets" in item:  # Excel file
            for sheet in item["sheets"]:
                data_description += f"\n#### Sheet: {sheet['sheet_name']}\n"
                data_description += f"Columns: {', '.join(sheet['columns'])}\n"
                data_description += f"Row count: {sheet['row_count']}\n"
                data_description += f"Preview (first 10 rows):\n```\n{sheet['preview']}\n```\n"
        else:  # CSV file
            data_description += f"Columns: {', '.join(item['columns'])}\n"
            data_description += f"Row count: {item['row_count']}\n"
            data_description += f"Preview (first 10 rows):\n```\n{item['preview']}\n```\n"
    
    prompt = f"""
    You are a Data Analyst examining OSF journal survey data for Paper #{paper_number}.
    
    Your task is to conduct a comprehensive analysis of the data files in the Paper #{paper_number} directory.
    
    IMPORTANT RESEARCH HYPOTHESIS:
    ```
    {hypothesis}
    ```
    
    IMPORTANT: Your analysis should focus on testing and exploring this hypothesis using the available data.
    
    Data files available for analysis:
    
    {data_description}
    
    Survey design information:
    ```
    {survey_design}
    ```
    
    Provide a complete and comprehensive data analysis including:
    
    1. Descriptive Statistics:
       - Calculate mean, median, standard deviation for all numeric variables
       - Frequency distributions for categorical variables
       - Create tables showing these statistics
    
    2. Correlation Analysis:
       - Calculate correlations between appropriate variables (e.g., speed perception, abstraction level, decision-making variables)
       - Identify significant relationships with correlation coefficients
    
    3. Group Comparisons:
       - Compare differences between experimental conditions (e.g., fast vs. slow movement)
       - Use specific statistics for these comparisons (t-tests, ANOVA, etc.)
    
    4. Experiment Analysis:
       - Compare outcomes between control and treatment groups
       - Calculate effect sizes for each experiment mentioned in the hypothesis
       - Test whether the data supports the speed-abstraction effect described in the hypothesis
    
    5. Key Findings and Patterns:
       - Identify the top 3-5 most significant findings with specific numbers
       - Evaluate whether the findings support the stated hypothesis
       - Note any unexpected or surprising patterns in the data
    
    Your analysis should include specific numerical results (means, standard deviations, correlation coefficients, p-values where appropriate). Provide tables of results where helpful. This is not a theoretical exercise - actually analyze the data provided to test the stated hypothesis.
    """
    
    print(f"Performing data analysis for Paper #{paper_number}...")
    
    # Call GPT to analyze the data
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a skilled data analyst specializing in research data. Always provide specific numerical results and detailed analysis of actual data, not theoretical instructions."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    
    analysis = response.choices[0].message.content
    
    # Save the analysis in the Output directory
    output_path = output_dir / "data_analysis.md"
    with open(output_path, "w") as f:
        f.write(analysis)
    
    # Save the hypothesis separately
    hypothesis_path = output_dir / "research_hypothesis.md"
    with open(hypothesis_path, "w") as f:
        f.write(f"# Research Hypothesis for Paper #{paper_number}\n\n{hypothesis}")
    
    print(f"Data analysis saved to {output_path}")
    print(f"Research hypothesis saved to {hypothesis_path}")
    return analysis, hypothesis

def review_methodology(paper_number, analysis, hypothesis):
    """Review research methodology."""
    paper_dir = BASE_DIR / f"Paper #{paper_number}"
    data_dir = paper_dir / "Data and Code"
    output_dir = paper_dir / "Output"
    
    # Read text files to find survey design information
    survey_design = read_text_files(data_dir)
    
    # Create the prompt for methodology review
    prompt = f"""
    You are a Methodology Expert reviewing the research design for Paper #{paper_number}.
    
    RESEARCH HYPOTHESIS:
    ```
    {hypothesis}
    ```
    
    Review the following survey design:
    ```
    {survey_design}
    ```
    
    And the following data analysis:
    ```
    {analysis}
    ```
    
    Your task is to conduct a comprehensive review of the methodology, focusing on concrete, specific aspects:
    
    1. Research Questions and Hypotheses:
       - Evaluate if the research hypothesis is clearly articulated and appropriately focused 
       - Assess if the hypothesis is testable with the collected data
       - Evaluate how well the seven experiments mentioned in the hypothesis are designed to test the speed-abstraction effect
       - Recommend specific improvements to the research questions
    
    2. Sampling Strategy:
       - Analyze the specific sampling method used (e.g., stratified random, convenience)
       - Calculate potential sampling errors and confidence intervals where possible
       - Evaluate sample size adequacy for the statistical tests conducted
       - Assess potential sampling biases and their impact on validity
    
    3. Experimental Design:
       - Evaluate the design of each experiment mentioned in the hypothesis
       - Assess the manipulation of speed (virtual vs. physical)
       - Evaluate the measurement of abstraction levels
       - Identify potential confounding variables
    
    4. Survey Instrument Validity and Reliability:
       - Evaluate construct validity - do the questions measure what they claim to?
       - Assess content validity - do the questions cover all relevant aspects of the topic?
       - Calculate (or estimate) reliability coefficients (e.g., Cronbach's alpha) for scale items
       - Identify specific items that may have validity or reliability issues
    
    5. Data Collection Procedures:
       - Evaluate the time frame and potential temporal biases
       - Assess the response rate and potential non-response bias
       - Evaluate the data collection method (online, in-person, etc.) and associated biases
    
    6. Data Analysis Methods:
       - Evaluate the appropriateness of statistical tests used to test the speed-abstraction effect
       - Assess if assumptions for these tests were met
       - Identify any missing or additional analyses that should be conducted
       - Suggest specific alternative analyses if applicable
    
    7. Ethical Considerations:
       - Evaluate informed consent procedures
       - Assess data privacy and confidentiality measures
       - Identify any ethical concerns with the methodology
    
    For each area, provide:
    1. A specific assessment (not vague generalities)
    2. Concrete examples from the survey design and data analysis
    3. Specific recommendations for improvement
    4. A severity rating (Low, Medium, High) for each issue identified
    
    Your review should be thorough, specific, and actionable - focusing on the actual methodology used rather than general principles.
    """
    
    print(f"Reviewing methodology for Paper #{paper_number}...")
    
    # Call GPT to review the methodology
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a research methodologist with expertise in survey design and experimental methods. Provide concrete, specific methodological assessments based on actual research designs, not general principles."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    
    review = response.choices[0].message.content
    
    # Save the review in the Output directory
    output_path = output_dir / "methodology_review.md"
    with open(output_path, "w") as f:
        f.write(review)
    
    print(f"Methodology review saved to {output_path}")
    return review

def create_report_outline(paper_number, analysis, review, hypothesis):
    """Create an outline for the research report."""
    paper_dir = BASE_DIR / f"Paper #{paper_number}"
    output_dir = paper_dir / "Output"
    
    # Create the prompt for report outline
    prompt = f"""
    You are a Research Writer creating a detailed outline for a research report based on OSF journal survey data.
    
    RESEARCH HYPOTHESIS:
    ```
    {hypothesis}
    ```
    
    You have access to:
    
    1. Data analysis:
    ```
    {analysis}
    ```
    
    2. Methodology review:
    ```
    {review}
    ```
    
    Your task is to create a comprehensive, detailed outline for a scholarly research report that tests the speed-abstraction effect described in the hypothesis. The outline should:
    
    1. Follow standard academic paper structure with all major sections and subsections
    2. Include specific bullet points under each section detailing exactly what content will be covered
    3. Highlight key statistical findings that should be reported in each section
    4. Note which methodological considerations should be addressed
    5. Specify where tables, figures, or visualizations should be included
    
    The outline must include these sections:
    
    1. Title (suggest a specific, descriptive title related to the speed-abstraction effect)
    
    2. Abstract (bullet points for what should be included)
    
    3. Introduction
       - Background literature review (specific topics related to movement speed and abstract thinking)
       - Problem statement
       - Research questions/hypotheses (be specific about each question)
       - Significance of the study
    
    4. Theoretical Framework
       - Speed-abstraction schema
       - Relationship between virtual and physical movement
       - Connection to decision-making processes
       - Alternative explanations (psychological distance, affect, fluency, spatial orientation)
    
    5. Methods
       - Participants (demographics, sampling approach)
       - Materials/Instruments (survey details, scales used)
       - Procedure (data collection, timeline, ethical considerations)
       - Data Analysis Approach (specific statistical tests)
    
    6. Results
       - Descriptive Statistics (which specific statistics to report)
       - Inferential Statistics (which analyses and findings to present)
       - Tables and Figures (describe what each table/figure should show)
       - Separate sections for each of the seven experiments mentioned in the hypothesis
    
    7. Discussion
       - Interpretation of Key Findings (for each major finding)
       - Evaluation of the speed-abstraction effect
       - Comparison with Existing Literature
       - Limitations (methodological issues identified in the review)
       - Implications (theoretical and practical)
    
    8. Conclusion
       - Summary of Findings
       - Recommendations
       - Future Research Directions
    
    9. References (note types of sources to include)
    
    10. Appendices (what should be included)
    
    For each section, include:
    - At least 3-5 detailed bullet points specifying content
    - Notes on what statistical data should be presented
    - References to specific findings from the data analysis
    - Methodological considerations from the review that should be addressed
    
    This outline will serve as the comprehensive blueprint for writing the actual paper, so be thorough and specific.
    """
    
    print(f"Creating report outline for Paper #{paper_number}...")
    
    # Call GPT to create the outline
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are an experienced academic writer who specializes in creating detailed, comprehensive research paper outlines. Your outlines are specific, thorough, and provide clear direction for paper writing."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=3000
    )
    
    outline = response.choices[0].message.content
    
    # Save the outline in the Output directory
    output_path = output_dir / "report_outline.md"
    with open(output_path, "w") as f:
        f.write(outline)
    
    print(f"Report outline saved to {output_path}")
    return outline

def write_report_draft(paper_number, outline, analysis, review, hypothesis):
    """Write a draft of the research report."""
    paper_dir = BASE_DIR / f"Paper #{paper_number}"
    output_dir = paper_dir / "Output"
    
    # Create the prompt for report draft
    prompt = f"""
    You are a Research Writer drafting a detailed research report based on OSF journal survey data.
    
    RESEARCH HYPOTHESIS:
    ```
    {hypothesis}
    ```
    
    You have:
    
    1. Report outline:
    ```
    {outline}
    ```
    
    2. Data analysis:
    ```
    {analysis}
    ```
    
    3. Methodology review:
    ```
    {review}
    ```
    
    Your task is to write a comprehensive, detailed academic research report that tests the speed-abstraction effect described in the hypothesis. The report should:
    
    1. Follows the structure in the outline precisely
    2. Incorporates ALL specific numerical findings from the data analysis
    3. Addresses ALL methodological considerations from the review
    4. Cites relevant literature throughout (create appropriate citations where needed)
    5. Includes detailed tables and describes specific visualizations where relevant
    
    Specific requirements:
    
    - Introduction: Include at least 3 paragraphs with relevant background literature on speed perception and abstract thinking, and clearly articulated research questions
    - Theoretical Framework: Explain the speed-abstraction schema in detail
    - Methods: Provide detailed subsections on participants, materials, and procedures for each of the seven experiments mentioned in the hypothesis (at least 500 words total)
    - Results: Report specific statistical findings with exact numbers, p-values, effect sizes, and confidence intervals where appropriate for each experiment
    - Discussion: Include at least 4 paragraphs interpreting results, discussing limitations, and suggesting implications
    - Conclusion: Summarize key findings and suggest at least 3 specific future research directions
    - Include at least a dozen relevant academic citations throughout
    
    The report should be thorough, detailed, and meet high standards for academic publication. The final product should be at least 2500 words and suitable for submission to a peer-reviewed journal.
    """
    
    print(f"Writing report draft for Paper #{paper_number}...")
    
    # Call GPT to write the draft
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are an academic writer with experience in publishing journal articles. Produce detailed, comprehensive reports with specific statistical findings and thorough academic content."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=4000
    )
    
    draft = response.choices[0].message.content
    
    # Save the draft in the Output directory
    output_path = output_dir / "report_draft.md"
    with open(output_path, "w") as f:
        f.write(draft)
    
    print(f"Report draft saved to {output_path}")
    return draft

def review_report(paper_number, draft, hypothesis):
    """Review the report draft."""
    paper_dir = BASE_DIR / f"Paper #{paper_number}"
    output_dir = paper_dir / "Output"
    
    # Load the original data analysis and methodology review for context
    analysis_path = output_dir / "data_analysis.md"
    review_path = output_dir / "methodology_review.md"
    
    with open(analysis_path, "r") as f:
        analysis = f.read()
    
    with open(review_path, "r") as f:
        methodology_review = f.read()
    
    # Create the prompt for report review
    prompt = f"""
    You are a Methodology Expert reviewing a draft research report based on OSF journal survey data.
    
    RESEARCH HYPOTHESIS:
    ```
    {hypothesis}
    ```
    
    The draft report:
    ```
    {draft}
    ```
    
    Original data analysis:
    ```
    {analysis}
    ```
    
    Original methodology review:
    ```
    {methodology_review}
    ```
    
    Your task is to conduct a thorough, critical review of this report draft focusing on methodological rigor, accurate reporting of findings, and scholarly writing. Provide detailed feedback on:
    
    1. Overall Quality and Structure (organization, coherence, flow)
       - Evaluate each major section (intro, methods, results, discussion, conclusion)
       - Assess how well the paper addresses the speed-abstraction effect described in the hypothesis
       - Identify specific areas where structure could be improved
    
    2. Methodological Accuracy and Completeness
       - Verify that all methodological details for the seven experiments are correctly reported
       - Identify any missing methodological information
       - Assess if statistical analyses are appropriately described and interpreted
       - Check if appropriate statistical tests were used and reported correctly
    
    3. Results Reporting
       - Verify that all major findings from the data analysis are included
       - Evaluate if the results properly test the speed-abstraction effect
       - Check for accuracy of reported statistics, p-values, effect sizes
       - Evaluate completeness of results reporting
       - Assess if tables/figures are properly described and necessary
    
    4. Discussion Quality
       - Evaluate if interpretations are justified by the actual results
       - Assess if alternative explanations are adequately addressed
       - Assess if limitations are thoroughly addressed
       - Check if implications are reasonable and not overstated
    
    5. Citation and Academic Writing
       - Evaluate if appropriate literature is cited
       - Check for academic writing style and terminology usage
       - Identify any areas of imprecise language
    
    6. Ethical Considerations
       - Check if ethical procedures are adequately described
       - Evaluate if privacy and confidentiality issues are addressed
    
    For each area, provide:
    - Specific examples from the text (with quotes where appropriate)
    - Concrete suggestions for improvement
    - A priority level for each issue (High/Medium/Low)
    
    Focus on substantive issues rather than minor stylistic concerns. Be specific, detailed, and actionable in your feedback.
    """
    
    print(f"Reviewing report for Paper #{paper_number}...")
    
    # Call GPT to review the report
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a research methodologist and academic editor with expertise in reviewing scholarly papers. Be thorough, specific, and critical in your review, focusing on substantive methodological and reporting issues."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=3000
    )
    
    feedback = response.choices[0].message.content
    
    # Save the feedback in the Output directory
    output_path = output_dir / "report_feedback.md"
    with open(output_path, "w") as f:
        f.write(feedback)
    
    print(f"Report feedback saved to {output_path}")
    return feedback

def finalize_report(paper_number, draft, feedback, hypothesis):
    """Finalize the research report."""
    paper_dir = BASE_DIR / f"Paper #{paper_number}"
    output_dir = paper_dir / "Output"
    
    # Get the data analysis and methodology review for additional context
    analysis_path = output_dir / "data_analysis.md"
    review_path = output_dir / "methodology_review.md"
    
    with open(analysis_path, "r") as f:
        analysis = f.read()
    
    with open(review_path, "r") as f:
        review = f.read()
    
    # Create the prompt for finalizing the report
    prompt = f"""
    You are a Research Writer finalizing a comprehensive research report based on OSF journal survey data.
    
    RESEARCH HYPOTHESIS:
    ```
    {hypothesis}
    ```
    
    You have:
    
    1. Draft report:
    ```
    {draft}
    ```
    
    2. Review feedback:
    ```
    {feedback}
    ```
    
    3. Original data analysis:
    ```
    {analysis}
    ```
    
    4. Original methodology review:
    ```
    {review}
    ```
    
    Your task is to produce an exceptional, publication-ready final report that thoroughly tests and explores the speed-abstraction effect described in the hypothesis. The report should:
    
    1. Addresses ALL feedback and suggestions from the review in detail
    2. Significantly enhances the draft by adding more depth, rigor, and detail
    3. Incorporates ALL key statistical findings from the data analysis with precise reporting
    4. Features a comprehensive methods section that addresses methodological concerns for all seven experiments
    5. Includes detailed tables for reporting results 
    6. Describes visualizations that would enhance understanding (as if they were included)
    7. Contains a sophisticated discussion that thoroughly examines implications
    8. Provides a detailed conclusion with specific recommendations
    
    Specific requirements for the final report:
    
    - Length: At least 3500 words
    - Citations: Include at least 15-20 relevant academic references
    - Results section: Report all statistical findings with precise values, confidence intervals, effect sizes, and p-values
    - Methods section: Include detailed subsections on participants, materials, procedures, and data analysis approach for each experiment
    - Discussion: At least 1000 words with thorough examination of findings, limitations, and implications
    - Use formal academic language and structure throughout
    - Include detailed demographic information about participants
    - Address ethical considerations thoroughly
    
    This should be an exemplary academic paper suitable for submission to a high-quality peer-reviewed journal.
    """
    
    print(f"Finalizing report for Paper #{paper_number}...")
    
    # Call GPT to finalize the report
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are an academic writer with experience producing thorough, detailed, and rigorous research reports for prestigious journals. Your reports are comprehensive, precise, and meticulously detailed."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=4000  # Reduced from 4500 to stay within limits
    )
    
    final_report = response.choices[0].message.content
    
    # Save the final report in the Output directory
    output_path = output_dir / "final_report.md"
    with open(output_path, "w") as f:
        f.write(final_report)
    
    print(f"Final report saved to {output_path}")
    return final_report

def main(paper_number: int):
    """Main function to run the journal report generation process."""
    paper_dir = BASE_DIR / f"Paper #{paper_number}"
    output_dir = paper_dir / "Output"
    
    # Create Paper directory if it doesn't exist
    paper_dir.mkdir(parents=True, exist_ok=True)
    # Create Output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n=== Starting report generation process for Paper #{paper_number} ===\n")
    
    # Get the research hypothesis from the user
    hypothesis = get_hypothesis()
    
    # Run the pipeline
    try:
        # Step 1: Perform data analysis
        print("\n--- Step 1: Data Analysis ---")
        analysis, hypothesis = analyze_data(paper_number, hypothesis)
        time.sleep(2)  # Pause to avoid rate limiting
        
        # Step 2: Review methodology
        print("\n--- Step 2: Methodology Review ---")
        review = review_methodology(paper_number, analysis, hypothesis)
        time.sleep(2)
        
        # Step 3: Create report outline
        print("\n--- Step 3: Report Outline Creation ---")
        outline = create_report_outline(paper_number, analysis, review, hypothesis)
        time.sleep(2)
        
        # Step 4: Write report draft
        print("\n--- Step 4: Report Draft Writing ---")
        draft = write_report_draft(paper_number, outline, analysis, review, hypothesis)
        time.sleep(2)
        
        # Step 5: Review report
        print("\n--- Step 5: Report Review ---")
        feedback = review_report(paper_number, draft, hypothesis)
        time.sleep(2)
        
        # Step 6: Finalize report
        print("\n--- Step 6: Report Finalization ---")
        final_report = finalize_report(paper_number, draft, feedback, hypothesis)
        
        print(f"\n=== Report generation complete for Paper #{paper_number}! ===")
        print(f"All outputs saved in: {output_dir}\n")
        print(f"Files generated:")
        print(f"- {output_dir}/research_hypothesis.md (Initial Input)")
        print(f"- {output_dir}/data_analysis.md (Step 1)")
        print(f"- {output_dir}/methodology_review.md (Step 2)")
        print(f"- {output_dir}/report_outline.md (Step 3)")
        print(f"- {output_dir}/report_draft.md (Step 4)")
        print(f"- {output_dir}/report_feedback.md (Step 5)")
        print(f"- {output_dir}/final_report.md (Final Output)")
        
    except Exception as e:
        print(f"Error in report generation process: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='JCR Journal Report Agent')
    parser.add_argument('--paper_number', type=int, required=True, help='Paper number to analyze')
    args = parser.parse_args()
    
    main(args.paper_number)