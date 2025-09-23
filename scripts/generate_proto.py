#!/usr/bin/env python3
"""Generate Python code from protobuf definitions."""

import os
import subprocess
import sys
from pathlib import Path

def generate_proto():
    """Generate Python code from .proto files."""
    proto_dir = Path("proto")
    output_dir = Path("eidolon/proto")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all .proto files
    proto_files = list(proto_dir.glob("*.proto"))
    
    if not proto_files:
        print("No .proto files found in proto/ directory")
        return
    
    # Generate Python code for each proto file
    for proto_file in proto_files:
        print(f"Generating code for {proto_file}")
        
        # Generate Python code
        subprocess.run([
            "python", "-m", "grpc_tools.protoc",
            f"--proto_path={proto_dir}",
            f"--python_out={output_dir}",
            f"--grpc_python_out={output_dir}",
            str(proto_file)
        ], check=True)
    
    # Create __init__.py file
    init_file = output_dir / "__init__.py"
    init_file.write_text('"""Generated protobuf code."""\n')
    
    print("Protobuf code generation completed")

if __name__ == "__main__":
    generate_proto()
