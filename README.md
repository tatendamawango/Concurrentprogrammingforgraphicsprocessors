# CUDA Student Grading with Concurrent Programming

This project demonstrates concurrent programming with CUDA to process and evaluate student data based on specific criteria. It performs parallel computation on student records using GPU threads, extracting and formatting results based on name and grade.

## Features

- **CUDA Kernel Execution**: Uses GPU threads to process student data concurrently.
- **Condition-Based Filtering**: Selects students whose names start with letters after 'P' in the alphabet.
- **Grade Classification**: Converts numerical grades to letter grades (A to F).
- **Formatted Output**: Names are capitalized and combined with year and grade for output.
- **Result Management**: Ensures no empty results are included by using available positions dynamically.

## Example

Given input data:
```json
{
  "students": [
    {"name": "Antanas", "year": 1, "grade": 6.95},
    {"name": "Sonata", "year": 3, "grade": 9.13},
    {"name": "Zigmas", "year": 3, "grade": 9.13}
  ]
}
```

The output result would be:
```
SONATA-3A
ZIGMAS-3A
```

## File Structure

- **main.cu**: Core implementation of the CUDA kernel and main logic.
- **data1.txt**: Input file containing student data.
- **results.txt**: Output file storing filtered and formatted student information.

## Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit installed
- C++ compiler with CUDA support

## Running the Project

1. Compile the CUDA code:
   ```bash
   nvcc main.cu -o student_grading
   ```

2. Run the compiled binary:
   ```bash
   ./student_grading
   ```

3. Check `results.txt` for the output.

## Additional Info

- Threads are assigned data chunks dynamically ensuring efficient utilization of GPU.
- Atomic operations and device functions are used to manage concurrent access.
