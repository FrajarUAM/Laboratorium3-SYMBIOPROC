import numpy as np
import matplotlib.pyplot as plt
import os

# Podstawowy model – bez wrażliwości
def base_model(y, config):
    p53, mdmcyto, mdmn, pten = y
    siRNA = config.get("siRNA", False)
    pten_off = config.get("PTEN_off", False)
    DNA_damage = config.get("DNA_damage", True)
    params = config["params"]

    p1 = params['p1']
    d1 = params['d1']
    dp53 = p1 - d1 * p53 * (mdmn ** 2)

    siRNA_factor = 0.02 if siRNA else 1.0
    p2 = params['p2'] * siRNA_factor
    d2 = params['d2'] * (1.0 if DNA_damage else 0.1)
    k1 = params['k1']
    k2 = params['k2']
    k3 = params['k3']
    term1 = p2 * (p53 ** 4) / (p53 ** 4 + k2 ** 4) if p53 > 0 else 0.0
    term2 = k1 * (k3 ** 2) / (k3 ** 2 + pten ** 2) * mdmcyto if pten > 0 else 0.0
    dMDMcyto = term1 - term2 - d2 * mdmcyto

    dMDMn = term2 - d2 * mdmn
    p3 = 0.0 if pten_off else params['p3']
    d3 = params['d3']
    term3 = p3 * (p53 ** 4) / (p53 ** 4 + k2 ** 4) if p53 > 0 else 0.0
    dPTEN = term3 - d3 * pten

    return np.array([dp53, dMDMcyto, dMDMn, dPTEN])

# Solver RK4 dla układu z wrażliwościami
def rk4_sensitivity(f, y0, S0, t0, tf, h, config, param_keys):
    times = [t0]
    y_list = [y0]
    S_list = [S0.copy()]
    y = y0.copy()
    S = S0.copy()
    t = t0

    while t < tf:
        k1y, k1S = f(t, y, S, config, param_keys)
        k2y, k2S = f(t + h/2, y + h/2 * k1y, S + h/2 * k1S, config, param_keys)
        k3y, k3S = f(t + h/2, y + h/2 * k2y, S + h/2 * k2S, config, param_keys)
        k4y, k4S = f(t + h, y + h * k3y, S + h * k3S, config, param_keys)

        y = y + h/6 * (k1y + 2*k2y + 2*k3y + k4y)
        S = S + h/6 * (k1S + 2*k2S + 2*k3S + k4S)
        t += h

        times.append(t)
        y_list.append(y)
        S_list.append(S.copy())

    return np.array(times), np.array(y_list), np.array(S_list)

def model_with_sensitivity(t, y, S, config, param_keys):
    dydt = base_model(y, config)

    # Numeryczna macierz Jacobiego df/dy
    Jy = compute_jacobian_y(y, config, param_keys)

    dSdt = np.zeros_like(S)
    for i, key in enumerate(param_keys):
        dfdtheta = compute_dF_dtheta(y, config, key, param_keys)
        dSdt[i] = Jy @ S[i] + dfdtheta

    return dydt, dSdt

def compute_jacobian_y(y, config, param_keys):
    eps = 1e-6
    n = len(y)
    J = np.zeros((n, n))
    f0 = base_model(y, config)
    for i in range(n):
        y_pert = y.copy()
        y_pert[i] += eps
        f_pert = base_model(y_pert, config)
        J[:, i] = (f_pert - f0) / eps
    return J

def compute_dF_dtheta(y, config, key, param_keys):
    p = config["params"]
    eps = p[key] * 1e-4 if p[key] != 0 else 1e-4
    p_pert = p.copy()
    p_pert[key] += eps
    config_pert = config.copy()
    config_pert["params"] = p_pert
    f0 = base_model(y, config)
    f_pert = base_model(y, config_pert)
    return (f_pert - f0) / eps


# Analiza wrażliwości
def run_sensitivity_analysis(params_nominal, scenario):
    y0 = np.array([26854, 11173, 17245, 154378])
    S0 = np.zeros((len(params_nominal), len(y0)))

    config = {
        "siRNA": scenario[0],
        "PTEN_off": scenario[1],
        "DNA_damage": not scenario[2],
        "params": params_nominal
    }
    param_keys = list(params_nominal.keys())
    
    t, y, S = rk4_sensitivity(model_with_sensitivity, y0, S0, 0, 48*60, h=1.0, config=config, param_keys=param_keys)
    sensitivities = {}
    for i, key in enumerate(param_keys):
        dy_dtheta = S[:, i, 0]
        theta = params_nominal[key]
        S_norm = (theta / y[:, 0]) * dy_dtheta
        sensitivities[key] = S_norm
    return t, sensitivities, y[:, 0]

def compute_rankings(sensitivities, t_array):
    RS, OS = {}, {}
    for key, S in sensitivities.items():
        absS = np.abs(S)
        RS[key] = np.mean(absS)
        OS[key] = absS[-1]
    ranking_RS = sorted(RS.items(), key=lambda x: x[1], reverse=True)
    ranking_OS = sorted(OS.items(), key=lambda x: x[1], reverse=True)
    return RS, OS, ranking_RS, ranking_OS

def plot_sensitivity_curves(t, sensitivities, ranking_RS, scenario_name, output_dir):
    best_param = ranking_RS[0][0]
    worst_param = ranking_RS[-1][0]

    for name, param in [("best", best_param), ("worst", worst_param)]:
        S_vec = sensitivities[param]
        plt.figure(figsize=(9, 5))
        plt.plot(t, S_vec, label=f"{param}")
        plt.axhline(0, color='gray', linestyle='--')
        plt.title(f"Wrażliwość {param} ({name}) – {scenario_name}")
        plt.xlabel("Czas [min]")
        plt.ylabel("S_norm")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"S_{param}_{name}_{scenario_name}.png"))
        plt.close()

    plt.figure(figsize=(10, 6))
    for param, S_vec in sensitivities.items():
        plt.plot(t, S_vec, label=param)
    plt.axhline(0, color='gray', linestyle='--')
    plt.title(f"Wszystkie S_norm – {scenario_name}")
    plt.xlabel("Czas [min]")
    plt.ylabel("S_norm")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"S_all_{scenario_name}.png"), dpi=300)
    plt.close()

def plot_rankings_bar(RS_dict, OS_dict, scenario_name, output_dir):
    keys = list(RS_dict.keys())
    x = np.arange(len(keys))
    rs_vals = [RS_dict[k] for k in keys]
    os_vals = [OS_dict[k] for k in keys]

    plt.figure(figsize=(10, 5))
    plt.bar(x, rs_vals, tick_label=keys)
    plt.title(f"Ranking RS – {scenario_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"ranking_RS_{scenario_name}.png"))
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(x, os_vals, tick_label=keys, color='orange')
    plt.title(f"Ranking OS – {scenario_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"ranking_OS_{scenario_name}.png"))
    plt.close()

def plot_param_shift_effects(params_nominal, best_param, worst_param, scenarios, output_dir):
    for param in [best_param, worst_param]:
        theta_nom = params_nominal[param]
        variants = {
            'nominal': theta_nom,
            'plus20': theta_nom * 1.2,
            'minus20': theta_nom * 0.8
        }
        for scen_name, scen in scenarios.items():
            plt.figure(figsize=(8, 5))
            for label, val in variants.items():
                p_copy = params_nominal.copy()
                p_copy[param] = val
                config = {
                    "siRNA": scen[0],
                    "PTEN_off": scen[1],
                    "DNA_damage": not scen[2],
                    "params": p_copy
                }
                y0 = np.array([26854, 11173, 17245, 154378])
                t_vals, y_vals, _ = rk4_sensitivity(model_with_sensitivity, y0, np.zeros((len(p_copy), len(y0))), 0, 48*60, h=1.0, config=config, param_keys=list(p_copy.keys()))
                plt.plot(t_vals, y_vals[:, 0], label=label)
            plt.title(f"p53 – {param} ±20% – {scen_name}")
            plt.xlabel("Czas [min]")
            plt.ylabel("p53 [nM]")
            plt.yscale("log")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"p53_shift_{param}_{scen_name}.png"))
            plt.close()

def main():
    scenarios = {
        "Basic": (False, False, True),
        "Tumor": (False, True, False)
    }
    params_nominal = {
        'p1': 8.8, 'p2': 440.0, 'p3': 100.0,
        'd1': 1.375e-14, 'd2': 1.375e-4, 'd3': 3e-5,
        'k1': 1.925e-4, 'k2': 1e5, 'k3': 1.5e5
    }
    output_dir = "results_lab3_sensitivity"
    os.makedirs(output_dir, exist_ok=True)

    all_rankings = {}

    for scen_name, scen_tuple in scenarios.items():
        print(f"\n--- Scenariusz: {scen_name} ---")
        t_vals, sens_dict, p53_vals = run_sensitivity_analysis(params_nominal, scen_tuple)
        RS, OS, ranking_RS, ranking_OS = compute_rankings(sens_dict, t_vals)

        all_rankings[scen_name] = {
            'RS': RS, 'OS': OS, 'ranking_RS': ranking_RS, 'ranking_OS': ranking_OS
        }

        plot_rankings_bar(RS, OS, scen_name, output_dir)
        plot_sensitivity_curves(t_vals, sens_dict, ranking_RS, scen_name, output_dir)

        print("Ranking RS (średni wpływ):")
        for param, val in ranking_RS:
            print(f"  {param:4}: RS = {val:.6f}")
        print("Ranking OS (wpływ końcowy):")
        for param, val in ranking_OS:
            print(f"  {param:4}: OS = {val:.6f}")

    best = all_rankings["Basic"]["ranking_RS"][0][0]
    worst = all_rankings["Basic"]["ranking_RS"][-1][0]
    plot_param_shift_effects(params_nominal, best, worst, scenarios, output_dir)
    print("\nZapisano wyniki do katalogu:", output_dir)

if __name__ == "__main__":
    main()
