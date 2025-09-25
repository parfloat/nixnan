#! /usr/bin/env python3
import os
import subprocess
import argparse
import sys

def report_str(result, marker_index):
    res = ""
    # Extract content from the marker to the end of stderr
    report_content = result.stderr[marker_index:]
    res += report_content
    # Ensure there's a newline before the exit code
    if not report_content.endswith('\n'):
        res += '\n'
    res += f"Exit Code: {result.returncode}\n"
    return res

def record_results(result, report_marker, filepath):
    # Search for the report marker in stderr
    marker_index = result.stderr.find(report_marker)

    if marker_index != -1:
        

        # Define the output file name
        output_filename = f"{filepath}.expect"
        print(f"  -> Found report. Writing to {output_filename}")

        # Write the report and exit code to the output file
        with open(output_filename, 'w') as f:
            f.write(report_str(result, marker_index))
    else:
        print(f"  -> No report marker found in stderr.")



def find_and_run_executables(ld_preload_lib, root_dir='.'):
    """
    Finds and runs all executable files in a directory hierarchy,
    capturing their output and exit codes to generate report files.
    """
    report_marker = "#nixnan: ------------ nixnan Report -----------"
    env = os.environ.copy()
    env['LD_PRELOAD'] = ld_preload_lib

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)

            # Skip this script itself
            if os.path.samefile(filepath, __file__):
                continue

            # Check if the file is executable by the current user
            if os.path.isfile(filepath) and os.access(filepath, os.X_OK):
                print(f"Running: {filepath}")
                try:
                    # Run the executable file with LD_PRELOAD
                    result = subprocess.run(
                        [filepath],
                        capture_output=True,
                        text=True,
                        check=False,  # Do not raise exception for non-zero exit codes
                        env=env
                    )
                    record_results(result, report_marker, filepath)

                except Exception as e:
                    print(f"  -> Error running {filepath}: {e}")

if __name__ == "__main__":
    print("Starting executable runner...")
    parser = argparse.ArgumentParser(
        description="Find and run executables with a preloaded library to generate reports."
    )
    parser.add_argument(
        "ld_preload_lib",
        help="Path to the .so library to be preloaded via LD_PRELOAD."
    )
    args = parser.parse_args()

    lib_path = args.ld_preload_lib
    if not os.path.exists(lib_path) or not os.path.isfile(lib_path):
        print(f"Error: Library file not found at '{lib_path}'", file=sys.stderr)
        sys.exit(1)
    if not lib_path.endswith('.so'):
        print(f"Error: Library file must be a .so file, but got '{lib_path}'", file=sys.stderr)
        sys.exit(1)

    find_and_run_executables(lib_path)
    print("Finished.")
    sys.exit(0)
    find_and_run_executables()
    print("Finished.")