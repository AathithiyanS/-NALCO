import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


def calculate_cooling_water_temp(casting_temp):
    return max(20, min(0.5 * casting_temp + 2, 35))

def calculate_casting_speed(casting_temp):
    return max(6, min(0.8 * casting_temp + 1000, 12))

def calculate_entry_temp(casting_temp):
    return max(400, min(450, 500))

def calculate_emulsion_temp(casting_temp):
    return max(40, min(50, 60))

def calculate_emulsion_pressure(casting_temp):
    return max(1, min(2, 3))

def calculate_emulsion_concentration(casting_temp):
    return max(2, min(3, 5))

def calculate_quench_pressure(casting_temp):
    return max(1, min(1.5, 2))

# Function to suggest adjustments
def suggest_adjustments(parameters, min_range, max_range):
    suggestions = {}
    function_mapping = {
        'cooling_temp': 'calculate_cooling_water_temp',
        'casting_speed': 'calculate_casting_speed',
        'entry_temp': 'calculate_entry_temp',
        'emulsion_temp': 'calculate_emulsion_temp',
        'emulsion_pressure': 'calculate_emulsion_pressure',
        'emulsion_concentration': 'calculate_emulsion_concentration',
        'quench_pressure': 'calculate_quench_pressure',
    }

    for param, value in parameters.items():
        if param == 'casting_temperature':
            continue

        func_name = function_mapping.get(param)
        if func_name:
            if func_name in globals():
                optimal_value = globals()[func_name](parameters['casting_temperature'])
                optimal_value = max(min_range[param], min(optimal_value, max_range[param]))
                percentage_change = ((optimal_value - value) / value) * 100
                suggestions[param] = {
                    'optimal_value': optimal_value,
                    'percentage_change': percentage_change
                }
            else:
                print(f"Function {func_name} not found in globals()")
    return suggestions


min_range = {'casting_temperature': 680, 'cooling_temp': 20, 'casting_speed': 6, 'entry_temp': 400,
             'emulsion_temp': 40, 'emulsion_pressure': 1, 'emulsion_concentration': 2, 'quench_pressure': 1}
max_range = {'casting_temperature': 750, 'cooling_temp': 35, 'casting_speed': 12, 'entry_temp': 500,
             'emulsion_temp': 60, 'emulsion_pressure': 3, 'emulsion_concentration': 5, 'quench_pressure': 2}

parameters = {
    'casting_temperature': 700,
    'cooling_temp': 25,
    'casting_speed': 10,
    'entry_temp': 450,
    'emulsion_temp': 50,
    'emulsion_pressure': 2,
    'emulsion_concentration': 3,
    'quench_pressure': 1.5
}

suggestions = suggest_adjustments(parameters, min_range, max_range)
print(suggestions)
