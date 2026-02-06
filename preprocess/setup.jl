using Pkg
Pkg.add([
    "Plots",
    "LinearAlgebra",
    "Rotations",
    "PythonCall",
    "Statistics",
    "JLD2",
    "PlotlyBase",
    "CondaPkg"
])

# Configure Python environment
using CondaPkg
CondaPkg.add("python", version="=3.12")
CondaPkg.add("numpy", version="<2")
CondaPkg.add_pip("opencv-python")
CondaPkg.status()