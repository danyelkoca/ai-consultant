import os
import sys
import json
import pandas as pd
import numpy as np
from scipy import stats
from io import StringIO


def convert_to_json_safe(obj):
    """Convert Python objects to JSON-safe format."""
    try:
        if isinstance(obj, (pd.DataFrame, pd.Series)):
            return obj.to_dict()
        elif isinstance(obj, (np.ndarray, list, tuple, set)):
            return [convert_to_json_safe(x) for x in obj]
        elif isinstance(obj, dict):
            return {str(k): convert_to_json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, (np.number, np.bool_)):
            return obj.item()
        elif isinstance(obj, pd.Timestamp):
            return obj.strftime("%Y-%m-%d %H:%M:%S")
        elif hasattr(obj, "to_dict"):
            return obj.to_dict()
        return obj
    except:
        return str(obj)


def execute_code_from_file(code_file: str, data_path: str) -> dict:
    """
    Execute code from a file in the sandbox environment.
    Returns the execution results and any generated figures/plots.
    """
    stdout_capture = None
    try:
        # Create output capture
        stdout_capture = StringIO()
        sys.stdout = stdout_capture

        # Create namespace with DataFrame and required libraries
        local_ns = {"pd": pd, "np": np, "stats": stats}

        # Load DataFrame with proper date parsing
        df = pd.read_csv(data_path)
        # Convert month column to datetime if it exists
        if "month" in df.columns:
            df["month"] = pd.to_datetime(df["month"])
        local_ns["df"] = df
        # Also provide the data path
        local_ns["DATA_PATH"] = data_path

        # Read the code file
        with open(code_file, "r") as f:
            code = f.read()

        # Execute the code
        exec(code, local_ns)

        # Get captured output and restore stdout
        sys.stdout = sys.__stdout__
        captured_output = stdout_capture.getvalue()

        # Try to get the last value from either the last line or a 'result' variable
        result = None
        try:
            last_line = code.strip().split("\n")[-1]
            if not last_line.startswith("#"):  # Ignore comments
                result = eval(last_line, local_ns)
        except:
            result = local_ns.get("result", None)

        # Convert result to JSON-safe format
        result = convert_to_json_safe(result)

        # Return success with output and result
        return {
            "success": True,
            "output": captured_output.strip() if captured_output else None,
            "result": result,
        }

    except Exception as e:
        # Ensure stdout is restored
        if sys.stdout != sys.__stdout__:
            sys.stdout = sys.__stdout__

        captured = stdout_capture.getvalue() if stdout_capture else None
        return {
            "success": False,
            "error": str(e),
            "output": captured.strip() if captured else None,
        }


if __name__ == "__main__":
    try:
        if len(sys.argv) != 3:
            result = {
                "success": False,
                "error": "Expected code file path and data path",
            }
        else:
            code_file, data_path = sys.argv[1:3]
            if not os.path.exists(code_file):
                result = {
                    "success": False,
                    "error": f"Code file not found: {code_file}",
                }
            elif not os.path.exists(data_path):
                result = {
                    "success": False,
                    "error": f"Data file not found: {data_path}",
                }
            else:
                result = execute_code_from_file(code_file, data_path)

        print(json.dumps(result, ensure_ascii=False))
    except Exception as e:
        error_result = {"success": False, "error": f"Unhandled error: {str(e)}"}
        print(json.dumps(error_result, ensure_ascii=False))
