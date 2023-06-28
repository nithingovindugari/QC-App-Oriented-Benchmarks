import importlib
import fire 
import os
import sys
import inspect

benchmark_algorithms = [
    "amplitude-estimation",
    "bernstein-vazirani",
    "deutsch-jozsa",
    "grovers",
    "hamiltonian-simulation",
    "hidden-shift",
    "maxcut",
    "monte-carlo",
    "phase-estimation",
    "quantum-fourier-transform",
    "shors",
    "vqe",
]

# Add algorithms to path:
for algorithm in benchmark_algorithms:
    sys.path.insert(1, os.path.join(f"{algorithm}", "qiskit"))

import ae_benchmark
import bv_benchmark
import dj_benchmark
import grovers_benchmark
import hamiltonian_simulation_benchmark
import hs_benchmark
import maxcut_benchmark
import mc_benchmark
import pe_benchmark
import qft_benchmark
import shors_benchmark
import vqe_benchmark
from custom.custom_qiskit_noise_model import my_noise_model
def run_benchmark(
    algorithm="quantum-fourier-transform",
    min_qubits=2,
    max_qubits=8,
    max_circuits=3,
    num_shots=100,
    backend_id="qasm_simulator",
    provider_backend=None,
    hub="ibm-q",
    group="open",
    project="main",
    provider_module_name=None,
    provider_class_name=None,
    noise_model=None,
    exec_options={},
    epsilon=0.05,
    degree=2,
    use_mcx_shim=False,
    use_XX_YY_ZZ=False,
    num_state_qubits=1,
    method=1,
    rounds=1,
    alpha=0.1,
    thetas_array=None,
    parameterized=False,
    do_fidelities=True,
    max_iter=30,
    score_metric="fidelity",
    x_metric="cumulative_exec_time",
    y_metric="num_qubits",
    fixed_metrics={},
    num_x_bins=15,
    x_size=None,
    y_size=None,
    use_fixed_angles=False,
    objective_func_type='approx_ratio',
    plot_results=True,
    save_res_to_file=False,
    save_final_counts=False,
    detailed_save_names=False,
    comfort=False,
    eta=0.5,
    _instance=None,
):
    # # For Inserting the Noise model default into exec option as it is function call
    # if noise_model is not None:
    #     if noise_model == 'None':
    #         exec_options["noise_model"] = None
    #     else:
    #         module, method = noise_model.split(".")
    #         module = globals()[module]
    #         method = method.split("(")[0]
    #         custom_noise = getattr(module, method)
    #         noise = custom_noise()
    #         exec_options["noise_model"] = noise
    # global noise
    # def call_func():
    if noise_model is not None:
        function_ref = globals().get(noise_model)
        # noise = my_noise_model()
        # exec_options["noise_model"] = noise
        if function_ref is not None and inspect.isfunction(function_ref):
            # Call the function and assign the returned value to the noise variable
            noise = function_ref()
            exec_options["noise_model"] = noise
    else:
        raise ValueError("Invalid noise model specified.")    

    
    # if noise_model is not None:
    #     if noise_model == "custom_qiskit_noise_model":
    #         exec_options["noise_model"] = custom_qiskit_noise_model()
    #     else:
    #          raise ValueError("Invalid noise model specified.")

    # Provider detail update using provider module name and class name
    if provider_module_name is not None and provider_class_name is not None:
        provider_class = getattr(importlib.import_module(provider_module_name), provider_class_name)
        provider = provider_class.get_backend(backend_id)
        provider_backend = provider
        
        
    # Parsing universal arguments.
    universal_args = {
        "min_qubits": min_qubits,
        "max_qubits": max_qubits,
        "num_shots": num_shots,
        "backend_id": backend_id,
        "provider_backend": provider_backend,
        "hub": hub,
        "group": group,
        "project": project,
        "exec_options": exec_options,
    }

    # Parsing additional arguments used in some algorithms.
    additional_args = {
        "epsilon": epsilon,
        "degree": degree,
        "use_mcx_shim": use_mcx_shim,
        "use_XX_YY_ZZ": use_XX_YY_ZZ,
        "num_state_qubits": num_state_qubits,
        "method": method,
    }

    # Parsing arguments for MaxCut
    maxcut_args = {
        "rounds": rounds,
        "alpha": alpha,
        "thetas_array": thetas_array,
        "parameterized": parameterized,
        "do_fidelities": do_fidelities,
        "max_iter": max_iter,
        "score_metric": score_metric,
        "x_metric": x_metric,
        "y_metric": y_metric,
        "fixed_metrics": fixed_metrics,
        "num_x_bins": num_x_bins,
        "x_size": x_size,
        "y_size": y_size,
        "use_fixed_angles": use_fixed_angles,
        "objective_func_type": objective_func_type,
        "plot_results": plot_results,
        "save_res_to_file": save_res_to_file,
        "save_final_counts": save_final_counts,
        "detailed_save_names": detailed_save_names,
        "comfort": comfort,
        "eta": eta,
        "_instance": _instance,
    }
    


    if algorithm == "amplitude-estimation":
        ae_benchmark.run(**universal_args, num_state_qubits=num_state_qubits,)

    elif algorithm == "bernstein-vazirani":
        bv_benchmark.run(**universal_args, method = method)

    elif algorithm == "deutsch-jozsa":
        dj_benchmark.run(**universal_args)

    elif algorithm == "grovers":
        universal_args["use_mcx_shim"] = additional_args["use_mcx_shim"]
        grovers_benchmark.run(**universal_args, use_mcx_shim = use_mcx_shim)

    elif algorithm == "hamiltonian-simulation":
        hamiltonian_simulation_benchmark.run(**universal_args , use_XX_YY_ZZ = use_XX_YY_ZZ)

    elif algorithm == "hidden-shift":
        hs_benchmark.run(**universal_args)

    elif algorithm == "maxcut":
        maxcut_args = {}
        maxcut_args.update(universal_args)
        maxcut_args.update(maxcut_args)
        maxcut_benchmark.run(**maxcut_args, method = method, degree = degree)

    elif algorithm == "monte-carlo":
        mc_benchmark.run(**universal_args, epsilon = epsilon, method = method, degree = degree)

    elif algorithm == "phase-estimation":
        pe_benchmark.run(**universal_args)

    elif algorithm == "quantum-fourier-transform":
        qft_benchmark.run(**universal_args, method = method)

    elif algorithm == "shors":
        shors_benchmark.run(**universal_args, method = method)

    elif algorithm == "vqe":
        vqe_benchmark.run(**universal_args,method = method)

    else:
        raise ValueError(f"Algorithm {algorithm} not supported.")

if __name__ == "__main__":
    # Instantiate the benchmark object
    benchmark = fire.Fire(run_benchmark)