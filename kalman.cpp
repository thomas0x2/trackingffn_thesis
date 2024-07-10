#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>

class KalmanFilter{
}

PYBIND11_MODULE(kalman, m) {
    py::class_<KalmanFilter>(m, "KalmanFilter")
        .def(py::init<>());
} 

