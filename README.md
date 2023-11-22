# Real-time Mandelbrot Set Visualization using CUDA

Welcome to the Real-time Mandelbrot Set Visualization project! This project utilizes CUDA technology in Visual Studio 2019 to compute and visualize the Mandelbrot set in real-time. Dive into the details below to understand the mathematics behind the Mandelbrot set and how to explore it through GPU acceleration.

## Table of Contents

- [Mandelbrot Set](#mandelbrot-set)
  - [Introduction](#introduction)
  - [Mathematics](#mathematics)
- [Project Overview](#project-overview)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Mandelbrot Set

### Introduction

The Mandelbrot set is a famous fractal in mathematics. It is defined by iterating a simple mathematical formula and examining the behavior of the resulting values. The set is named after Benoît B. Mandelbrot, who introduced the concept.

### Mathematics

The Mandelbrot set is defined by the iterative process:

\[ Z_{n+1} = Z_{n}^2 + C \]

Where:
- \(Z_{n}\) is a complex number representing the current iteration.
- \(C\) is a complex constant representing the initial value.
- The iteration continues until \(|Z_{n}| > 2\) or a predefined maximum number of iterations is reached.

The set consists of all complex values of \(C\) for which the iteration does not diverge to infinity.

## Project Overview

This project leverages CUDA, a parallel computing platform and programming model developed by NVIDIA, to perform high-performance parallel computations on the GPU. The Visual Studio 2019 solution includes CUDA kernels for computing the Mandelbrot set in parallel, allowing for real-time visualization.

## Getting Started

### Prerequisites

Make sure you have the following installed:
- Visual Studio 2019
- CUDA Toolkit

### Installation

1. Clone the repository.
2. Open the Visual Studio solution file.
3. Build the solution to ensure CUDA kernels are compiled correctly.

## Usage

Run the compiled executable to experience real-time visualization of the Mandelbrot set using GPU acceleration. Use mouse interactions to zoom in and explore the intricate details of the set.

## Contributing

If you'd like to contribute, please fork the repository and create a pull request. Feel free to open issues for feature requests or bug reports.

Explore the beauty of the Mandelbrot set in real-time with GPU acceleration. Happy coding!
