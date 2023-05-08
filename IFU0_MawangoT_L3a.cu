#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <stdio.h>
#include <cctype>
#include "cuda_runtime.h"
#include <cuda.h>
#include <device_launch_parameters.h>
#include <atomic>
#include <device_atomic_functions.hpp>

struct Student {
    std::string name;
    int year;
    double grade;
};

std::istream& operator>>(std::istream& is, Student& student) {
    return is >> student.name >> student.year >> student.grade;
}

struct Result {
    std::string name;
    std::string year_grade;
};


__global__ void computeResult(const Student* students, Result* results, int num_students, int* num_results) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_students) {

        const Student& student = students[tid];

        char first_letter = std::toupper(student.name[0]);
        if (first_letter > 'P') {
            char grade = getGrade(student);
            
            std::string upper_name = student.name;
            for (char& c : upper_name) {
                c = std::toupper(c);
            }

            int result_index = -1;
            for (int i = 0; i < num_students; ++i) {
                if (results[i].name.empty()) {
                    result_index = i;
                    break;
                }
            }

            if (result_index >= 0) {
                results[result_index].name = upper_name;
                results[result_index].year_grade = std::to_string(student.year) + grade;
            }
            num_results = atomicAdd(&num_results, 1);
        }
    }
}

__device__ char getGrade(Student student) {
    char grade;
    if (student.grade >= 90) {
        grade = 'A';
    }
    else if (student.grade >= 80) {
        grade = 'B';
    }
    else if (student.grade >= 70) {
        grade = 'C';
    }
    else if (student.grade >= 60) {
        grade = 'D';
    }
    else {
        grade = 'F';
    }
    return grade;
}


void PrintStudentsToFile(Result* studentArray, int numStudents, const std::string& filename) {
    std::ofstream file(filename);

    for (int i = 0; i < numStudents; i++) {
        file << studentArray[i].name << "-" << studentArray[i].year_grade << std::endl;
    }
}


int main() {
    std::ifstream input_file("data1.txt");
    int num_students = 0;
    int num_results = 0;
    std::string line;
    while (std::getline(input_file, line)) {
        ++num_students;
    }
    input_file.clear();
    input_file.seekg(0, std::ios::beg);
    Student* students = new Student[num_students];
    for (int i = 0; i < num_students; ++i) {
        input_file >> students[i];
    }
    input_file.close();

    int num_threads = 64;
    int block_size = 32;
    int num_blocks = (num_students + num_threads - 1) / num_threads;


    Student* d_students;
    cudaMalloc(&d_students, num_students * sizeof(Student));
    Result* d_results;
    cudaMalloc(&d_results, num_students * sizeof(Result));

    cudaMemcpy(d_students, students, num_students * sizeof(Student), cudaMemcpyHostToDevice);

    computeResult<<<num_blocks, block_size >>>(d_students, d_results, num_students, num_results);

    Result* results = new Result[num_students];
    cudaMemcpy(results, d_results, num_results * sizeof(Result), cudaMemcpyDeviceToHost);


    for (int i = 0; i < num_students; ++i) {
        std::cout << results[i].name << "-" << results[i].year_grade << std::endl;
    }

    PrintStudentsToFile(results, num_students, "results.txt");

    delete[] students;
    delete[] results;
    cudaFree(d_students);
    cudaFree(d_results);

    return 0;
}
