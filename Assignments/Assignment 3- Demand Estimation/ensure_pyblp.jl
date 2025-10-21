#!/usr/bin/env julia

# This script wires Julia to the Python package `pyblp` through PyCall.
# Ensure that PyCall is built against the Python you used to install pyblp.

using PyCall

"""
    ensure_pyblp(; install_if_missing::Bool = true, extra_pip_packages = String[])

Import the Python module `pyblp`. When `install_if_missing` is true and the module
is not found, install it (and any `extra_pip_packages`) via `pip` in the same Python
environment that PyCall is bound to, then retry the import.
"""
function ensure_pyblp(; install_if_missing::Bool = true, extra_pip_packages::Vector{String} = String[])
    try
        return pyimport("pyblp")
    catch err
        if err isa PyCall.PyError && occursin("No module named 'pyblp'", sprint(showerror, err))
            install_if_missing || rethrow(err)

            python_exe = PyCall.python
            println("pyBLP not found. Installing into $(python_exe)...")

            run(`$python_exe -m pip install --upgrade pip`)
            run(`$python_exe -m pip install pyblp $(extra_pip_packages...)`)

            return pyimport("pyblp")
        else
            rethrow(err)
        end
    end
end

const pyblp = ensure_pyblp()
const pd = pyimport("pandas")
const np = pyimport("numpy")


println("Loaded pyBLP version $(pyblp.__version__)")
