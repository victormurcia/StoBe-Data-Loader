import os
import re
import pandas as pd
import streamlit as st
from natsort import natsorted
import time
from concurrent.futures import ProcessPoolExecutor

def parse_basis_line(line):
    parts = line.split(':', 1)
    if len(parts) == 2:
        atom = parts[0].split()[-1]
        basis = parts[1].strip()
        return atom, basis
    return None, None

def extract_all_information(file_path, originating_atom):
    """
    Extracts orbital data, basis sets, and energy information from the given file.
    
    Parameters:
    file_path (str): The path to the file containing the calculation results.
    originating_atom (str): The atom from which the data is extracted.
    
    Returns:
    tuple: A tuple containing six DataFrames: df_alpha, df_beta, df_auxiliary, df_orbital, df_model, df_energies.
    """
    data_alpha = []
    data_beta = []
    auxiliary_basis = []
    orbital_basis = []
    model_potential = []
    energies = {
        "Total energy (H)": None,
        "Nuc-nuc energy (H)": None,
        "El-nuc energy (H)": None,
        "Kinetic energy (H)": None,
        "Coulomb energy (H)": None,
        "Ex-cor energy (H)": None,
        "Orbital energy core hole (H)": None,
        "Orbital energy core hole (eV)": None,
        "Rigid spectral shift (eV)": None,
        "Ionization potential (eV)": None,
        "Atom": originating_atom,
        "Calculation Type": "TP" if "_tp.out" in file_path else "GND" if "gnd.out" in file_path else "EXC" if "exc.out" in file_path else None
    }

    with open(file_path, 'r') as file:
        lines = file.readlines()

    start_index = None
    end_index = None
    for i, line in enumerate(lines):
        if "         Spin alpha                              Spin beta" in line:
            start_index = i + 2
        elif " IV)" in line:
            end_index = i
            break

    if start_index is not None and end_index is not None:
        for line in lines[start_index:end_index]:
            if line.strip() == "":
                continue
            components = [x for x in line.split() if x.strip()]
            if len(components) >= 9:
                mo_index_alpha, occup_alpha, energy_alpha, sym_alpha = components[:4]
                mo_index_beta, occup_beta, energy_beta, sym_beta = components[5:9]
                mo_index_alpha = mo_index_alpha.strip(')')
                mo_index_beta = mo_index_beta.strip(')')
                data_alpha.append({"MO_Index": mo_index_alpha, "Occup.": occup_alpha, "Energy(eV)": energy_alpha, "Sym.": sym_alpha})
                data_beta.append({"MO_Index": mo_index_beta, "Occup.": occup_beta, "Energy(eV)": energy_beta, "Sym.": sym_beta})

    current_section = None
    for line in lines:
        if "I)  AUXILIARY BASIS SETS" in line:
            current_section = "auxiliary"
        elif "II)  ORBITAL BASIS SETS" in line:
            current_section = "orbital"
        elif "III)  MODEL POTENTIALS" in line:
            current_section = "model"
        elif "BASIS DIMENSIONS" in line or "WARNING! Electron count may be inconsistent:" in line or "(NEW) SYMMETRIZATION INFORMATION" in line:
            current_section = None
        elif current_section in ["auxiliary", "orbital", "model"] and line.strip():
            atom, basis = parse_basis_line(line)
            if atom and basis:
                if current_section == "auxiliary":
                    auxiliary_basis.append([atom, basis])
                elif current_section == "orbital":
                    orbital_basis.append([atom, basis])
                elif current_section == "model":
                    model_potential.append([atom, basis])

        if "Total energy   (H)" in line:
            match = re.search(r"Total energy   \(H\) =\s*([-+]?[0-9]*\.?[0-9]+)", line)
            if match:
                energies["Total energy (H)"] = float(match.group(1))
        elif "Nuc-nuc energy (H)" in line:
            energies["Nuc-nuc energy (H)"] = float(line.split('=')[-1].strip())
        elif "El-nuc energy  (H)" in line:
            energies["El-nuc energy (H)"] = float(line.split('=')[-1].strip())
        elif "Kinetic energy (H)" in line:
            energies["Kinetic energy (H)"] = float(line.split('=')[-1].strip())
        elif "Coulomb energy (H)" in line:
            energies["Coulomb energy (H)"] = float(line.split('=')[-1].strip())
        elif "Ex-cor energy  (H)" in line:
            energies["Ex-cor energy (H)"] = float(line.split('=')[-1].strip())
        elif "Orbital energy core hole" in line:
            match = re.search(r"Orbital energy core hole\s*=\s*([-+]?[0-9]*\.?[0-9]+)\s*H\s*\(\s*([-+]?[0-9]*\.?[0-9]+)\s*eV\s*\)", line)
            if match:
                energies["Orbital energy core hole (H)"] = float(match.group(1))
                energies["Orbital energy core hole (eV)"] = float(match.group(2))
        elif "Rigid spectral shift" in line:
            energies["Rigid spectral shift (eV)"] = float(line.split('=')[-1].strip().split()[0])
        elif "Ionization potential" in line:
            energies["Ionization potential (eV)"] = float(line.split('=')[-1].strip().split()[0])

    df_alpha = pd.DataFrame(data_alpha)
    df_beta = pd.DataFrame(data_beta)
    df_auxiliary = pd.DataFrame(auxiliary_basis, columns=['Atom', 'Auxiliary Basis'])
    df_orbital = pd.DataFrame(orbital_basis, columns=['Atom', 'Orbital Basis'])
    df_model = pd.DataFrame(model_potential, columns=['Atom', 'Model Potential'])
    df_energies = pd.DataFrame([energies])

    # Debug print to check for duplicate entries
    #print("Debug: Energies DataFrame for file", file_path)
    #print(df_energies.head)

    return df_alpha, df_beta, df_auxiliary, df_orbital, df_model, df_energies

def sort_dataframe_naturally(df, column):
    df[column] = df[column].astype(str)
    sorted_index = natsorted(df[column].tolist())
    df = df.set_index(column).loc[sorted_index].reset_index()
    return df

def process_file(file_info):
    entry, file_path = file_info
    originating_atom = entry
    df_alpha, df_beta, df_auxiliary, df_orbital, df_model, df_energies = extract_all_information(file_path, originating_atom)

    file_name = os.path.basename(file_path)
    df_auxiliary['Originating File'] = file_name
    df_orbital['Originating File'] = file_name
    df_model['Originating File'] = file_name
    df_alpha['Originating File'] = file_name
    df_beta['Originating File'] = file_name
    df_energies['Originating File'] = file_name

    df_combined = df_auxiliary.merge(df_orbital, on=['Atom', 'Originating File'], how='outer').merge(df_model, on=['Atom', 'Originating File'], how='outer')

    return df_combined, df_energies, df_alpha, df_beta

def process_directory(directory, progress_bar, progress_text):
    combined_results_list = []
    energy_results_list = []
    orbital_alpha_list = []
    orbital_beta_list = []

    pattern = re.compile(r'^[A-Za-z]+\d+$')
    entries = [entry for entry in os.listdir(directory) if os.path.isdir(os.path.join(directory, entry)) and pattern.match(entry)]

    # Collect file paths in a single pass
    file_paths = []
    for entry in entries:
        entry_path = os.path.join(directory, entry)
        for file in os.listdir(entry_path):
            if file.endswith(('_tp.out', 'gnd.out', 'exc.out')):
                file_paths.append((entry, os.path.join(entry_path, file)))

    total_files = len(file_paths)
    st.write(f'Total files to process: {total_files}')

    processed_files = 0

    with ProcessPoolExecutor() as executor:
        for df_combined, df_energies, df_alpha, df_beta in executor.map(process_file, file_paths):
            combined_results_list.append(df_combined)
            energy_results_list.append(df_energies)
            orbital_alpha_list.append(df_alpha)
            orbital_beta_list.append(df_beta)

            processed_files += 1
            percentage_complete = min((processed_files / total_files), 1.0)
            progress_bar.progress(percentage_complete)
            progress_text.text(f'Processing: {percentage_complete*100:.2f}% completed.')

    combined_results = pd.concat(combined_results_list, ignore_index=True)
    energy_results = pd.concat(energy_results_list, ignore_index=True).drop_duplicates()
    orbital_alpha = pd.concat(orbital_alpha_list, ignore_index=True)
    orbital_beta = pd.concat(orbital_beta_list, ignore_index=True)

    energy_results = sort_dataframe_naturally(energy_results, 'Atom')
    
    # Remove duplicates from energy_results
    energy_results = energy_results.drop_duplicates()
    
    return combined_results, energy_results, orbital_alpha, orbital_beta

st.title('X-ray Absorption Spectrum Analysis')

directory = st.text_input('Enter the directory containing the output folders:')
if directory:
    if os.path.isdir(directory):
        st.write('Processing directory:', directory)
        
        progress_bar = st.progress(0)
        progress_text = st.empty()
        
        start_time = time.time()
        combined_results, energy_results, orbital_alpha, orbital_beta = process_directory(directory, progress_bar, progress_text)
        end_time = time.time()
        
        st.write('### Combined Basis Sets Results')
        st.dataframe(combined_results)
        st.write('### Combined Energy Results')
        st.dataframe(energy_results)
        st.write('### Orbital Alpha Data')
        st.dataframe(orbital_alpha)
        st.write('### Orbital Beta Data')
        st.dataframe(orbital_beta)
        st.write(f'Processing completed in {end_time - start_time:.2f} seconds.')
    else:
        st.write('Invalid directory. Please enter a valid directory path.')
