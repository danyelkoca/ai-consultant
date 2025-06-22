import os
import json
import sys
import signal
import docker
import pandas as pd
import logging
import uuid
from io import StringIO
from docker import errors as docker_errors
from openai import OpenAI
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from docker.models.containers import Container
from docker.models.images import Image


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def signal_handler(signum: int, frame: Any) -> None:
    """Handle interrupt signals gracefully"""
    logging.info("\nReceived interrupt signal, cleaning up...")
    # The cleanup will be handled by __del__
    sys.exit(0)


class ConsultantAgent:
    def __init__(self, data_path: str):
        # Set up signal handler for graceful shutdown
        signal.signal(signal.SIGINT, signal_handler)
        logging.info("Initializing ConsultantAgent")

        # Load API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            from dotenv import load_dotenv

            load_dotenv()
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment or .env file")

        self.openai_client = OpenAI(api_key=api_key)
        self.data_path = Path(data_path)

        # Use default environment settings with Unix socket on macOS
        self.docker_client = docker.DockerClient(base_url="unix://var/run/docker.sock")
        self.container_name = "sandbox"  # Use fixed container name
        self.container: Optional[Container] = None

        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        # Ensure sandbox image is built and create container
        self._build_sandbox()
        self._create_container()
        logging.info("ConsultantAgent initialized successfully")

    def _build_sandbox(self):
        """Build the sandbox Docker image if it doesn't exist or if force_rebuild is True."""
        try:
            try:
                self.docker_client.images.get("sandbox")
                logging.info("Using existing sandbox image")
                return
            except docker_errors.ImageNotFound:
                logging.info("Sandbox image not found, building...")

            sandbox_path = Path(__file__).parent / "sandbox"
            self.docker_client.images.build(
                path=str(sandbox_path), tag="sandbox", rm=True
            )
            logging.info("Sandbox image built successfully")
        except Exception as e:
            logging.error(f"Failed to build sandbox image: {e}")
            raise

    def _create_container(self):
        """Create a reusable sandbox container."""
        try:
            # Try to remove any existing container with the same name
            try:
                old_container = self.docker_client.containers.get(self.container_name)
                logging.info(
                    f"Found existing container {self.container_name}, removing it"
                )
                old_container.remove(force=True)
            except docker_errors.NotFound:
                pass  # Container doesn't exist, which is fine

            # Debug logging for volume mounting
            logging.info(f"Data path: {self.data_path}")
            logging.info(f"Data parent path: {self.data_path.parent}")
            logging.info(f"Data file exists: {self.data_path.exists()}")

            volume_config = {
                str(self.data_path.parent): {"bind": "/app/data", "mode": "ro"}
            }
            logging.info(f"Volume config: {volume_config}")

            logging.info(f"Creating container {self.container_name}")
            self.container = self.docker_client.containers.run(
                "sandbox",
                name=self.container_name,
                stdin_open=True,
                detach=True,
                remove=True,
                volumes=volume_config,
                command="tail -f /dev/null",  # Keep container running
            )
            if not self.container:  # Defensive check
                raise RuntimeError("Failed to create container")

            logging.info(f"Container {self.container_name} created successfully")
        except Exception as e:
            logging.error(f"Failed to create container: {e}")
            raise

    def __del__(self):
        """Cleanup the container when the agent is destroyed."""
        try:
            if hasattr(self, "container") and self.container:
                try:
                    logging.info(f"Stopping container {self.container_name}")
                    self.container.stop(timeout=1)  # Reduced timeout for faster cleanup
                    logging.info(f"Container {self.container_name} stopped")
                except:
                    pass  # Ignore errors during normal stop

                try:
                    # Always try to force remove
                    self.container.remove(force=True)
                    logging.info(f"Container {self.container_name} removed")
                except:
                    pass  # Ignore errors during force remove
        except:
            pass  # Ignore all errors during cleanup

    def _print_hypothesis_cycle(
        self,
        cycle: int,
        max_cycles: int,
        hypothesis_data: Dict[str, Any],
        results: Dict[str, Any],
    ):
        """Print a clean, well-formatted hypothesis cycle output."""
        print("\n" + "=" * 80)
        print(f"HYPOTHESIS CYCLE {cycle}/{max_cycles}")
        print("=" * 80)

        print("\nHYPOTHESIS:")
        print("-" * 80)
        print(hypothesis_data["hypothesis"])

        print("\nTEST CODE:")
        print("-" * 80)
        print(hypothesis_data["test_code"])

        print("\nRESULTS:")
        print("-" * 80)
        if results.get("success") == True:  # Explicit check for True
            if results.get("output"):
                print("Execution Log:")
                print(results["output"])
            if results.get("result"):
                print("\nAnalysis Results:")
                print(json.dumps(results["result"], indent=2))
            else:
                print("(No results returned)")
        else:
            error_msg = results.get("error", "Unknown error")
            print("Error:", error_msg)
            if results.get("output"):
                print("\nExecution Log:")
                print(results["output"])

        print("=" * 80 + "\n")

    def _execute_code(self, code: str) -> Dict[str, Any]:
        """Execute code in sandbox container and return results."""
        try:
            if not self.container:
                return {"success": False, "error": "No sandbox container available"}

            # Clean the code:
            # 1. Split into lines
            # 2. Remove empty or whitespace-only lines
            # 3. Replace tabs with spaces (4 spaces per tab)
            # 4. Keep proper indentation
            cleaned_lines = []
            for line in code.splitlines():
                # Skip empty or whitespace-only lines
                if not line.strip():
                    continue
                # Replace tabs with spaces (4 spaces per tab)
                cleaned_line = line.replace("\t", "    ")
                cleaned_lines.append(cleaned_line.rstrip())

            cleaned_code = "\n".join(cleaned_lines)

            # Write code to a temporary file in the container
            code_file = "/tmp/analysis.py"

            # Escape both single quotes and format strings for printf
            escaped_code = cleaned_code.replace("'", "'\\''")  # Escape single quotes
            escaped_code = escaped_code.replace("%", "%%")  # Escape % for printf
            write_cmd = f"printf '{escaped_code}' > {code_file}"

            # Write the code file
            write_result = self.container.exec_run(["sh", "-c", write_cmd])
            if write_result.exit_code != 0:
                return {
                    "success": False,
                    "error": f"Failed to write code file: {write_result.output.decode()}",
                }

            # Execute the code file
            cmd = f"python /app/execute.py {code_file} /app/data/sales.csv"
            result = self.container.exec_run(cmd)
            output = result.output.decode("utf-8").strip()

            try:
                result_json = json.loads(output)
                if not isinstance(result_json, dict):
                    return {"success": False, "error": "Output is not a dictionary"}

                # If we have a success key, return as is
                if "success" in result_json:
                    return result_json

                # If no explicit success flag but we have output or result, consider it success
                if "output" in result_json or "result" in result_json:
                    result_json["success"] = True
                    return result_json

                # Otherwise, something went wrong
                return {"success": False, "error": "Invalid output format from sandbox"}

            except json.JSONDecodeError as e:
                return {
                    "success": False,
                    "error": f"Failed to parse output as JSON: {str(e)}\nOutput was: {output}",
                }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _get_data_info(self) -> str:
        """Get information about the data schema and sample rows."""
        try:
            df = pd.read_csv(self.data_path)
            info_capture = StringIO()
            df.info(buf=info_capture)

            # Include more comprehensive data info
            info = f"""
Data Schema:
{info_capture.getvalue()}

First 5 rows:
{df.head().to_string()}

Data Summary:
- Total rows: {len(df)}
- Numeric columns stats:
{df.describe().to_string()}
"""
            return info
        except Exception as e:
            logging.error(f"Error reading data: {e}")
            return f"Error reading data: {str(e)}"

    def _print_llm_response(self, question: str, hypothesis_data: Dict[str, Any]):
        """Format and print LLM response for debugging."""
        logging.info("\nLLM Response for question: " + question)
        logging.info("=" * 80)
        logging.info("Hypothesis: " + hypothesis_data["hypothesis"])
        logging.info("=" * 80)

    def _generate_hypothesis(
        self, question: str, previous_findings: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Generate a hypothesis about how to test the question."""
        try:
            # First, analyze the data structure
            data_structure = self._analyze_data_structure()
            if "error" in data_structure:
                raise ValueError(f"Failed to analyze data: {data_structure['error']}")

            # Format the data structure information in a clear way
            column_details = []
            for col, info in data_structure.get("column_info", {}).items():
                detail = f"- {col} (type: {info['dtype']}):\n"
                detail += f"  * {info['unique_count']} unique values\n"
                detail += f"  * Sample values: {', '.join(str(x) for x in info['sample_values'][:3])}\n"
                if info.get("common_values"):
                    detail += f"  * Most common values: {dict(list(info['common_values'].items())[:3])}\n"
                column_details.append(detail)

            data_info = f"""Data Overview:
- Total rows: {data_structure.get('total_rows', 'unknown')}
- Total columns: {data_structure.get('total_columns', 'unknown')}

Columns:
{''.join(column_details)}

{self._format_findings_history(previous_findings) if previous_findings else ''}
"""
            print("Generating hypothesis with LLM")

            response = self.openai_client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a data science consultant that forms hypotheses and writes code to test them.
                        
                        IMPORTANT ANALYSIS STRATEGY: 
                        1. Start with high-level analysis of each factor
                        2. When a factor shows any significance:
                           - Test its interaction with other factors
                           - Look for specific combinations of factors
                           - Validate the pattern holds under different conditions
                        3. For each hypothesis:
                           - Test clear, specific relationships
                           - Report effect sizes and statistical significance
                           - Check if patterns are consistent
                        4. Build upon previous findings:
                           - If a factor is significant, test its interactions
                           - If a combination shows an effect, validate against other factors
                           - Rule out confounding variables
                        
                        CODE REQUIREMENTS:
                        1. The DataFrame is already loaded as 'df':
                           - Available as 'df' variable
                           - DO NOT try to read any CSV file
                        
                        2. Required code structure:
                           ```python
                           import pandas as pd
                           import numpy as np
                           import json
                           from scipy import stats
                           
                           try:
                               # The DataFrame is already loaded as 'df'
                               # Your analysis code here
                               result = {{
                                   "analysis": "results",
                                   "statistics": "your_stats_here"
                               }}
                               print(json.dumps({{"success": True, "result": result}}))
                           except Exception as e:
                               print(json.dumps({{"success": False, "error": str(e)}}))
                           ```
                        
                        Here is the information about the data and previous findings:
                        {data_info}""".format(data_info=data_info),
                    },
                    {
                        "role": "user",
                        "content": f"Based on this data and previous findings, help me investigate: {question}",
                    },
                ],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "create_hypothesis",
                            "description": "Create a hypothesis and test code",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "hypothesis": {
                                        "type": "string",
                                        "description": "The hypothesis to test",
                                    },
                                    "test_code": {
                                        "type": "string",
                                        "description": "Python code to test the hypothesis. Must follow the template with proper JSON output handling.",
                                    },
                                },
                                "required": ["hypothesis", "test_code"],
                            },
                        },
                    }
                ],
                tool_choice={
                    "type": "function",
                    "function": {"name": "create_hypothesis"},
                },
            )

            tool_call = response.choices[0].message.tool_calls[0]
            if tool_call.function.name == "create_hypothesis":
                args = tool_call.function.arguments.strip()
                try:
                    args = args.replace("\\n", "\n").replace("\n", "\\n")
                    hypothesis_data = json.loads(args)
                    logging.info("Successfully parsed hypothesis data")
                    self._print_llm_response(question, hypothesis_data)
                    return hypothesis_data
                except json.JSONDecodeError as e:
                    logging.error(f"Failed to parse hypothesis JSON. Error: {e}")
                    logging.error(f"Raw JSON string: {repr(args)}")
                    return {
                        "hypothesis": "Error generating hypothesis",
                        "test_code": """
import json
print(json.dumps({"success": True, "result": {"error": "Failed to generate valid test code"}}))
""",
                    }
            raise ValueError("Unexpected tool call response")

        except Exception as e:
            logging.error(f"Error in hypothesis generation: {str(e)}")
            return {
                "hypothesis": "Error generating hypothesis",
                "test_code": """
import json
print(json.dumps({"success": True, "result": {"error": "Failed to generate valid test code"}}))
""",
            }

    def _evaluate_findings(
        self, question: str, findings: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Evaluate if more hypotheses are needed and synthesize current findings."""
        try:
            findings_text = ""
            for idx, finding in enumerate(findings, 1):
                findings_text += f"\nHypothesis {idx}:\n"
                findings_text += f"Statement: {finding['hypothesis']}\n"
                findings_text += (
                    f"Results: {json.dumps(finding['results'], indent=2)}\n"
                )

            response = self.openai_client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {
                        "role": "system",
                        "content": """You are analyzing data to understand causation. Previous hypotheses and their results are provided.
                        
                        Important guidelines:
                        1. When a factor shows correlation, investigate its interaction with other factors before concluding
                        2. For each significant factor found, ensure follow-up hypotheses explore:
                           - Interaction with other variables
                           - Ruling out confounding factors
                           - Validating the relationship holds across different conditions
                        3. Don't stop analysis until:
                           - All relevant factors have been investigated
                           - Their interactions have been explored
                           - Other potential factors have been ruled out
                        4. Avoid similar broad hypotheses - each new hypothesis should test a specific factor or interaction
                        """,
                    },
                    {
                        "role": "user",
                        "content": f"""Question: {question}

Previous findings:
{json.dumps(findings, indent=2)}

Evaluate if more hypotheses are needed and why.""",
                    },
                ],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "evaluation_result",
                            "description": "Provide evaluation results",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "need_more_hypotheses": {
                                        "type": "boolean",
                                        "description": "Whether more hypotheses are needed",
                                    },
                                    "reason": {
                                        "type": "string",
                                        "description": "Explanation of why more hypotheses are needed or not",
                                    },
                                    "synthesis": {
                                        "type": "string",
                                        "description": "Synthesis of findings so far",
                                    },
                                },
                                "required": [
                                    "need_more_hypotheses",
                                    "reason",
                                    "synthesis",
                                ],
                            },
                        },
                    }
                ],
                tool_choice={
                    "type": "function",
                    "function": {"name": "evaluation_result"},
                },
            )

            tool_call = response.choices[0].message.tool_calls[0]
            if tool_call.function.name == "evaluation_result":
                return json.loads(tool_call.function.arguments)

            raise ValueError("Unexpected tool call response")
        except Exception as e:
            logging.error(f"Error in findings evaluation: {str(e)}")
            return {
                "need_more_hypotheses": False,
                "reason": f"Error in evaluation: {str(e)}",
                "synthesis": "Error synthesizing findings",
            }

    def _fix_code_error(self, error_msg: str, code: str) -> str:
        """Ask LLM to fix code based on the error message."""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a Python debugging expert. Fix the code based on the error message.
                        Important:
                        1. Return only the fixed code, no explanations
                        2. Maintain the same analysis logic
                        3. Ensure proper indentation
                        4. The DataFrame 'df' is already loaded
                        5. Return results must be in dictionary format
                        """,
                    },
                    {
                        "role": "user",
                        "content": f"""Error message:
{error_msg}

Code to fix:
{code}""",
                    },
                ],
            )

            fixed_code = response.choices[0].message.content.strip()
            logging.info("Code fixed by LLM")
            return fixed_code
        except Exception as e:
            logging.error(f"Error in code fixing: {str(e)}")
            return code  # Return original code if fixing fails

    def analyze(self, question: str, max_cycles: int = 10, max_retries: int = 3) -> str:
        """Run dynamic hypothesis cycles to answer the question."""
        findings = []
        cycle = 0
        conclusive_answer = False

        while cycle < max_cycles and not conclusive_answer:
            cycle += 1
            try:
                # Generate and test hypothesis with previous findings
                hypothesis = self._generate_hypothesis(question, findings)

                # Initialize results
                results = {"success": False, "error": "Not executed yet"}

                # Try executing the code with retries for errors
                code_success = False
                retry_count = 0
                current_code = hypothesis["test_code"]

                while not code_success and retry_count < max_retries:
                    results = self._execute_code(current_code)

                    if results.get("success"):
                        code_success = True
                    else:
                        retry_count += 1
                        logging.info(
                            f"Code execution failed (attempt {retry_count}/{max_retries})"
                        )
                        logging.info(f"Error: {results.get('error')}")

                        if retry_count < max_retries:
                            # Try to fix the code
                            current_code = self._fix_code_error(
                                results.get("error", "Unknown error"), current_code
                            )
                            logging.info("Retrying with fixed code")
                        else:
                            logging.error(
                                "Max retries reached, moving to next hypothesis"
                            )
                            break  # Skip storing this failed attempt

                # Log the hypothesis and results
                logging.info(f"\nHypothesis {cycle}:")
                logging.info(f"Statement: {hypothesis['hypothesis']}")
                logging.info("Results:")
                logging.info(json.dumps(results, indent=2))

                # Store findings only if successful and not debugging
                if (
                    code_success
                    and results.get("result")
                    and not results.get("result", {}).get("error")
                ):
                    findings.append(
                        {
                            "hypothesis": hypothesis["hypothesis"],
                            "results": results.get("result", {}),
                        }
                    )

                # Print the current cycle results
                self._print_hypothesis_cycle(cycle, max_cycles, hypothesis, results)

                # Evaluate if we need more hypotheses
                evaluation = self._evaluate_findings(question, findings)

                logging.info(f"\nEvaluation after cycle {cycle}:")
                logging.info(
                    f"Need more hypotheses: {evaluation['need_more_hypotheses']}"
                )
                logging.info(f"Reason: {evaluation['reason']}")

                if not evaluation["need_more_hypotheses"]:
                    conclusive_answer = True
                    logging.info("Analysis complete - synthesizing final results")
                    return evaluation["synthesis"]

            except Exception as e:
                logging.error(f"Error in analysis cycle {cycle}: {e}")
                if not findings:
                    return f"Error: Analysis failed - {str(e)}"

        # If we hit max cycles, synthesize what we have
        final_evaluation = self._evaluate_findings(question, findings)
        if not findings:
            return "No conclusive results found after maximum analysis cycles."
        return final_evaluation["synthesis"]

    def _analyze_data_structure(self) -> Dict[str, Any]:
        """Analyze the input data to understand its structure and content."""
        try:
            # Execute code to analyze the data
            analysis_code = """
import pandas as pd
import numpy as np
import json
from io import StringIO

try:
    # Get basic information about the data
    info_capture = StringIO()
    df.info(buf=info_capture)
    df_info = info_capture.getvalue()

    # Get column types and sample values
    columns_info = {}
    for col in df.columns:
        unique_values = df[col].nunique()
        sample_values = df[col].dropna().sample(min(3, len(df))).tolist()
        dtype = str(df[col].dtype)
        
        # For categorical-like columns (low unique values)
        if unique_values <= 10:
            value_counts = df[col].value_counts().head(5).to_dict()
        else:
            value_counts = None
        
        # Convert sample values to strings to ensure JSON serialization
        str_sample_values = [str(x) for x in sample_values]
        
        columns_info[col] = {
            "dtype": dtype,
            "unique_count": int(unique_values),
            "sample_values": str_sample_values,
            "common_values": value_counts,
            "null_count": int(df[col].isna().sum())
        }

    # Get basic statistics for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_stats = {}
    if len(numeric_cols) > 0:
        stats = df[numeric_cols].describe()
        for col in numeric_cols:
            numeric_stats[col] = {k: float(v) for k, v in stats[col].items()}

    result = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "column_info": columns_info,
        "numeric_stats": numeric_stats,
        "df_info": df_info
    }
    print(json.dumps({"success": True, "result": result}))
except Exception as e:
    print(json.dumps({"success": False, "error": str(e)}))
"""

            results = self._execute_code(analysis_code)
            if not results.get("success"):
                return {
                    "error": results.get("error", "Failed to analyze data structure")
                }

            return results.get("result", {})
        except Exception as e:
            logging.error(f"Error analyzing data structure: {e}")
            return {"error": str(e)}

    def _format_findings_history(self, findings: List[Dict[str, Any]]) -> str:
        """Format the history of successful findings for the LLM."""
        if not findings:
            return "No previous findings yet."

        formatted = "Previous successful findings:\n"
        for idx, finding in enumerate(findings, 1):
            formatted += f"\nFinding {idx}:\n"
            formatted += f"Hypothesis: {finding['hypothesis']}\n"
            if finding.get("results"):
                formatted += f"Results: {json.dumps(finding['results'], indent=2)}\n"
            formatted += "-" * 40 + "\n"
        return formatted
