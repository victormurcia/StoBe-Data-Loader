import os
import re
import pandas as pd
import streamlit as st
from natsort import natsorted
import time
from concurrent.futures import ProcessPoolExecutor
import py3Dmol
from stmol import showmol
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import seaborn as sns
import plotly.graph_objects as go
from scipy.integrate import quad
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from joblib import Parallel, delayed
import plotly.figure_factory as ff
import plotly.colors
from zipfile import ZipFile
from io import BytesIO
import plotly.express as px
            
def parse_basis_line(line):
    """
    Parses a line of text to extract the atom and basis set information.

    Args:
        line (str): A string containing the atom and basis set information in the format "Atom: Basis".

    Returns:
        tuple: A tuple containing the atom (str) and the basis (str). 
               If the line does not contain both atom and basis set information, returns (None, None).
    """
    parts = line.split(':', 1)
    if len(parts) == 2:
        atom = parts[0].split()[-1]
        basis = parts[1].strip()
        return atom, basis
    return None, None

def extract_energy_information(lines):
    """
    Extracts energy information from the given lines and returns it as a dictionary.
    
    Parameters:
    lines (list): The lines of the file containing the calculation results.
    
    Returns:
    dict: A dictionary containing the extracted energy information.
    """
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
    }

    for line in lines:
        if "Total energy   (H)" in line:
            match = re.search(r"Total energy\s+\(H\)\s*=\s*([-+]?[0-9]*\.?[0-9]+)", line)
            if match:
                energies["Total energy (H)"] = float(match.group(1))
        elif "Nuc-nuc energy (H)" in line:
            match = re.search(r"Nuc-nuc energy\s+\(H\)\s*=\s*([-+]?[0-9]*\.?[0-9]+)", line)
            if match:
                energies["Nuc-nuc energy (H)"] = float(match.group(1))
        elif "El-nuc energy  (H)" in line:
            match = re.search(r"El-nuc energy\s+\(H\)\s*=\s*([-+]?[0-9]*\.?[0-9]+)", line)
            if match:
                energies["El-nuc energy (H)"] = float(match.group(1))
        elif "Kinetic energy (H)" in line:
            match = re.search(r"Kinetic energy\s+\(H\)\s*=\s*([-+]?[0-9]*\.?[0-9]+)", line)
            if match:
                energies["Kinetic energy (H)"] = float(match.group(1))
        elif "Coulomb energy (H)" in line:
            match = re.search(r"Coulomb energy\s+\(H\)\s*=\s*([-+]?[0-9]*\.?[0-9]+)", line)
            if match:
                energies["Coulomb energy (H)"] = float(match.group(1))
        elif "Ex-cor energy  (H)" in line:
            match = re.search(r"Ex-cor energy\s+\(H\)\s*=\s*([-+]?[0-9]*\.?[0-9]+)", line)
            if match:
                energies["Ex-cor energy (H)"] = float(match.group(1))
        elif "Orbital energy core hole" in line:
            match = re.search(r"Orbital energy core hole\s*=\s*([-+]?[0-9]*\.?[0-9]+)\s*H\s*\(\s*([-+]?[0-9]*\.?[0-9]+)\s*eV\s*\)", line)
            if match:
                energies["Orbital energy core hole (H)"] = float(match.group(1))
                energies["Orbital energy core hole (eV)"] = float(match.group(2))
        elif "Rigid spectral shift" in line:
            match = re.search(r"Rigid spectral shift\s*=\s*([-+]?[0-9]*\.?[0-9]+)\s*eV", line)
            if match:
                energies["Rigid spectral shift (eV)"] = float(match.group(1))
        elif "Ionization potential" in line:
            match = re.search(r"Ionization potential\s*=\s*([-+]?[0-9]*\.?[0-9]+)\s*eV", line)
            if match:
                energies["Ionization potential (eV)"] = float(match.group(1))

    return energies

def extract_all_information(file_path, originating_atom):
    """
    Extracts orbital data, basis sets, energy information, x-ray transition data, and atomic coordinates from the given file.
    
    Parameters:
    file_path (str): The path to the file containing the calculation results.
    originating_atom (str): The atom from which the data is extracted.
    
    Returns:
    tuple: A tuple containing DataFrames: df_alpha, df_beta, df_auxiliary, df_orbital, df_model, df_energies, df_xray_transitions, df_atomic_coordinates.
    """
    data_alpha = []
    data_beta = []
    auxiliary_basis = []
    orbital_basis = []
    model_potential = []
    xray_transitions = []
    atomic_coordinates = []

    with open(file_path, 'r') as file:
        lines = file.readlines()
        
    energies = extract_energy_information(lines)
    energies["Atom"] = originating_atom
    energies["Calculation Type"] = "TP" if "_tp.out" in file_path else "GND" if "gnd.out" in file_path else "EXC" if "exc.out" in file_path else None

    start_index = None
    end_index = None
    xray_start = False
    atomic_start_index = None
    atomic_end_index = None
    current_section = None

    for i, line in enumerate(lines):
        if "         Spin alpha                              Spin beta" in line:
            start_index = i + 2
        elif " IV)" in line:
            end_index = i
        elif "I)  AUXILIARY BASIS SETS" in line:
            current_section = "auxiliary"
        elif "II)  ORBITAL BASIS SETS" in line:
            current_section = "orbital"
        elif "III)  MODEL POTENTIALS" in line:
            current_section = "model"
        elif "           E (eV)   OSCL       oslx       osly       oslz         osc(r2)       <r2>" in line:
            xray_start = True
        elif xray_start and "-----" in line:
            continue
        elif xray_start and line.strip() and line.startswith(" #"):
            try:
                index = int(line[2:6].strip())
                e_ev = float(line[7:17].strip())
                oscl = float(line[18:28].strip())
                oslx = float(line[29:39].strip())
                osly = float(line[40:50].strip())
                oslz = float(line[51:61].strip())
                osc_r2 = float(line[62:72].strip())
                r2 = float(line[73:].strip())
                xray_transitions.append({
                    "Index": index,
                    "E": e_ev,
                    "OS": oscl,
                    "osx": oslx,
                    "osy": osly,
                    "osz": oslz,
                    "osc(r2)": osc_r2,
                    "<r2>": r2
                })
            except ValueError as e:
                print(f"Error parsing line: {line}\nError: {e}")
        elif "Single image calculation (Angstrom):" in line:
            atomic_start_index = i + 3
        elif 'Smallest atom distance' in line:
            atomic_end_index = i
        elif current_section in ["auxiliary", "orbital", "model"] and line.strip():
            atom, basis = parse_basis_line(line)
            if atom and basis:
                if current_section == "auxiliary":
                    auxiliary_basis.append([atom, basis])
                elif current_section == "orbital":
                    orbital_basis.append([atom, basis])
                elif current_section == "model":
                    model_potential.append([atom, basis])

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

    if atomic_start_index is not None and atomic_end_index is not None:
        atomic_coordinates_lines = lines[atomic_start_index:atomic_end_index]
        for line in atomic_coordinates_lines:
            if line.strip() and not any(col in line for col in ['Atom', 'x', 'y', 'z', 'q', 'nuc', 'mass', 'neq', 'grid', 'grp']):  # Skip empty lines and the header
                split_line = line.split()
                if len(split_line) >= 11:
                    atom_info = split_line[1]  # Use the atom type and number
                    atomic_coordinates.append([atom_info] + split_line[2:11])

    df_alpha = pd.DataFrame(data_alpha)
    df_beta = pd.DataFrame(data_beta)
    df_auxiliary = pd.DataFrame(auxiliary_basis, columns=['Atom', 'Auxiliary Basis'])
    df_orbital = pd.DataFrame(orbital_basis, columns=['Atom', 'Orbital Basis'])
    df_model = pd.DataFrame(model_potential, columns=['Atom', 'Model Potential'])
    df_energies = pd.DataFrame([energies])
    df_xray_transitions = pd.DataFrame(xray_transitions)
    df_atomic_coordinates = pd.DataFrame(atomic_coordinates, columns=['Atom', 'x', 'y', 'z', 'q', 'nuc', 'mass', 'neq', 'grid', 'grp'])

    numeric_columns = ['x', 'y', 'z', 'q', 'nuc', 'mass', 'neq', 'grid', 'grp']
    df_atomic_coordinates[numeric_columns] = df_atomic_coordinates[numeric_columns].apply(pd.to_numeric, errors='coerce')
    
    print(df_energies)
    
    return df_alpha, df_beta, df_auxiliary, df_orbital, df_model, df_energies, df_xray_transitions, df_atomic_coordinates


def sort_dataframe_naturally(df, column):
    """
    Sorts a DataFrame naturally by the specified column.

    Natural sorting orders numbers in the way humans expect, e.g., 1, 2, 10 instead of 1, 10, 2.

    Args:
        df (pd.DataFrame): The DataFrame to be sorted.
        column (str): The column name to sort by.

    Returns:
        pd.DataFrame: The sorted DataFrame.
    """
    df[column] = df[column].astype(str)
    sorted_index = natsorted(df[column].tolist())
    df = df.set_index(column).loc[sorted_index].reset_index()
    return df

def process_file(file_info):
    """
    Processes a file to extract various pieces of information and combine them into DataFrames.

    Args:
        file_info (tuple): A tuple containing an entry (str) and a file path (str).

    Returns:
        tuple: A tuple containing the following DataFrames:
            - df_combined (pd.DataFrame): Combined DataFrame of auxiliary, orbital, and model information.
            - df_energies (pd.DataFrame): DataFrame containing energy information.
            - df_alpha (pd.DataFrame): DataFrame containing alpha information.
            - df_beta (pd.DataFrame): DataFrame containing beta information.
            - df_xray_transitions (pd.DataFrame): DataFrame containing x-ray transitions information.
            - df_atomic_coordinates (pd.DataFrame): DataFrame containing atomic coordinates.
        If an error occurs during processing, returns None.
    """
    try:
        entry, file_path = file_info
        originating_atom = entry
        df_alpha, df_beta, df_auxiliary, df_orbital, df_model, df_energies, df_xray_transitions, df_atomic_coordinates = extract_all_information(file_path, originating_atom)
    
        file_name = os.path.basename(file_path)
        df_auxiliary['Originating File'] = file_name
        df_orbital['Originating File'] = file_name
        df_model['Originating File'] = file_name
        df_alpha['Originating File'] = file_name
        df_beta['Originating File'] = file_name
        df_energies['Originating File'] = file_name
        df_xray_transitions['Originating File'] = file_name
        df_atomic_coordinates['Originating File'] = file_name
        df_combined = df_auxiliary.merge(df_orbital, on=['Atom', 'Originating File'], how='outer').merge(df_model, on=['Atom', 'Originating File'], how='outer')
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

    return df_combined, df_energies, df_alpha, df_beta, df_xray_transitions, df_atomic_coordinates

def process_directory(directory, progress_bar, progress_text, width1, width2, ewid1, ewid2):
    """
    Processes all files in a directory to extract and combine various pieces of information into DataFrames.

    Args:
        directory (str): The path to the directory containing subdirectories with files to process.
        progress_bar (st.progress): A Streamlit progress bar widget to display progress.
        progress_text (st.text): A Streamlit text widget to display progress text.
        width1 (float): The width value used in the broadening function for energy values below ewid1.
        width2 (float): The width value used in the broadening function for energy values above ewid2.
        ewid1 (float): The lower threshold energy value for broadening.
        ewid2 (float): The upper threshold energy value for broadening.

    Returns:
        tuple: A tuple containing the following DataFrames:
            - combined_results (pd.DataFrame): Combined DataFrame of auxiliary, orbital, and model information.
            - energy_results (pd.DataFrame): DataFrame containing energy information.
            - orbital_alpha (pd.DataFrame): DataFrame containing alpha orbital information.
            - orbital_beta (pd.DataFrame): DataFrame containing beta orbital information.
            - xray_transitions (pd.DataFrame): DataFrame containing x-ray transitions information.
            - atomic_coordinates (pd.DataFrame): DataFrame containing atomic coordinates.
    """
    combined_results_list = []
    energy_results_list = []
    orbital_alpha_list = []
    orbital_beta_list = []
    xray_transitions_list = []
    atomic_coordinates_list = []
    
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
        for result in executor.map(process_file, file_paths):
            df_combined, df_energies, df_alpha, df_beta, df_xray_transitions, df_atomic_coordinates = result
            combined_results_list.append(df_combined)
            energy_results_list.append(df_energies)
            orbital_alpha_list.append(df_alpha)
            orbital_beta_list.append(df_beta)
            xray_transitions_list.append(df_xray_transitions)
            atomic_coordinates_list.append(df_atomic_coordinates)

            processed_files += 1
            percentage_complete = min((processed_files / total_files), 1.0)
            progress_bar.progress(percentage_complete)
            progress_text.text(f'Processing: {percentage_complete*100:.2f}% completed.')

    combined_results = pd.concat(combined_results_list, ignore_index=True)
    energy_results = pd.concat(energy_results_list, ignore_index=True)
    orbital_alpha = pd.concat(orbital_alpha_list, ignore_index=True)
    orbital_beta = pd.concat(orbital_beta_list, ignore_index=True)
    xray_transitions = pd.concat(xray_transitions_list, ignore_index=True)
    atomic_coordinates = pd.concat(atomic_coordinates_list, ignore_index=True)

    energy_results = sort_dataframe_naturally(energy_results, 'Atom')
    
    # Ensure that the Atom column in energy_results is capitalized
    energy_results['Atom'] = energy_results['Atom'].str.upper()
    
    energy_results = energy_results.drop_duplicates()
    energy_results = energy_results.reset_index(drop=True)

    # Add the Atom column and move it to the first position in orbital_alpha, orbital_beta, and xray_transitions
    for df_name, df in [('orbital_alpha', orbital_alpha), ('orbital_beta', orbital_beta), ('xray_transitions', xray_transitions)]:
        df['Atom'] = df['Originating File'].str.replace(r'(_tp\.out|gnd\.out|exc\.out)', '', regex=True).str.upper()
        columns = ['Atom'] + [col for col in df.columns if col != 'Atom']
        df = df[columns]
        # Reassign the DataFrame back to the variable
        if df_name == 'orbital_alpha':
            orbital_alpha = df
        elif df_name == 'orbital_beta':
            orbital_beta = df
        elif df_name == 'xray_transitions':
            xray_transitions = df
    
    def broad(E):
        """
        Determines the broadening width based on the energy value.

        Args:
            E (float): The energy value.

        Returns:
            float: The calculated width based on the energy value.
        """
        if E < ewid1:
            return width1
        elif E > ewid2:
            return width2
        else:
            return width1 + (width2 - width1) * (E - ewid1) / (ewid2 - ewid1)
    
    xray_transitions['width'] = xray_transitions['E'].apply(broad)
    
    # Calculate the magnitude of the vector
    magnitude = np.sqrt(xray_transitions['osx']**2 + xray_transitions['osy']**2 + xray_transitions['osz']**2)
    
    # Create new columns with the normalized components
    xray_transitions['normalized_osx'] = xray_transitions['osx'] / magnitude
    xray_transitions['normalized_osy'] = xray_transitions['osy'] / magnitude
    xray_transitions['normalized_osz'] = xray_transitions['osz'] / magnitude
    
    # Calculate the magnitude of the normalized vector
    normalized_magnitude = np.sqrt(xray_transitions['normalized_osx']**2 + xray_transitions['normalized_osy']**2 + xray_transitions['normalized_osz']**2)
    
    # Add the magnitude of the normalized vector to the dataframe
    xray_transitions['normalized_magnitude'] = normalized_magnitude
    
    # Vector (0, 0, 1)
    reference_vector = np.array([0, 0, 1])
    
    # Calculate the dot product and the angle theta
    dot_product = (
        xray_transitions['normalized_osx'] * reference_vector[0] +
        xray_transitions['normalized_osy'] * reference_vector[1] +
        xray_transitions['normalized_osz'] * reference_vector[2]
    )
    theta = np.arccos(dot_product)
    
    # Convert theta from radians to degrees
    theta_degrees = np.degrees(theta)
    
    # Ensure the angle is between 0 and 90 degrees
    theta_degrees = np.where(theta_degrees > 90, 180 - theta_degrees, theta_degrees)
    
    # Add theta to the dataframe
    xray_transitions['theta'] = theta_degrees

    return combined_results, energy_results, orbital_alpha, orbital_beta, xray_transitions, atomic_coordinates

def dataframe_to_xyz(df, file_name="molecule.xyz"):
    """
    Convert a DataFrame to XYZ format file.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing atomic coordinates.
    file_name (str): The name of the output XYZ file.
    """
    lines = []
    # First line: total number of atoms
    lines.append(str(len(df)))
    # Second line: molecule name or comment
    lines.append("Generated by dataframe_to_xyz")
    # All other lines: element symbol or atomic number, x, y, and z coordinates
    for idx, row in df.iterrows():
        atom_symbol = ''.join([i for i in row['Atom'] if not i.isdigit()])  # Strip numbers from atom label
        lines.append(f"{atom_symbol} {row['x']} {row['y']} {row['z']}")
    
    # Write to file
    with open(file_name, 'w') as f:
        f.write("\n".join(lines))
    print(f"XYZ file '{file_name}' created successfully.")

def visualize_xyz_with_stmol(df, file_name, label_size=14, bond_width=0.1, atom_scale=0.3):
    """
    Visualize an XYZ file using Stmol and py3Dmol.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing atomic coordinates and labels.
    file_name (str): The name of the XYZ file to read.
    label_size (int): The font size for the labels.
    """
    # Read the XYZ file
    with open(file_name, 'r') as f:
        xyz_data = f.read()
    
    # Create a 3Dmol.js viewer
    view = py3Dmol.view(width=800, height=600)
    view.addModel(xyz_data, 'xyz')
    view.setStyle({'stick': {'radius': bond_width}, 'sphere': {'scale': atom_scale}})
    view.setBackgroundColor('black')  # Set background color to black
    
    # Add labels based on the Atom column with atom numbers
    atom_counters = {}
    for i, row in df.iterrows():
        atom_symbol = ''.join([i for i in row['Atom'] if not i.isdigit()])  # Strip numbers from atom label
        if atom_symbol not in atom_counters:
            atom_counters[atom_symbol] = 0
        atom_counters[atom_symbol] += 1
        label = f"{atom_symbol}{atom_counters[atom_symbol]}"
        x, y, z = row['x'], row['y'], row['z']
        view.addLabel(label, {'position': {'x': x, 'y': y, 'z': z}, 'backgroundColor': 'black', 'fontColor': 'white', 'fontSize': label_size})
    
    view.zoomTo()
    return view

def plot_individual_spectra(xray_transitions, E_max):
    """
    Generates and plots spectra for each unique atom in the xray_transitions DataFrame.

    Parameters:
    xray_transitions (pd.DataFrame): DataFrame containing the x-ray transitions with 'Atom', 'E (eV)', 'width', and 'OSCL' columns.
    E_max (float): Maximum energy value for the x-axis range.
    """
    unique_atoms = natsorted(xray_transitions['Atom'].unique())  # Sort atoms in natural order
    
    num_atoms = len(unique_atoms)
    num_cols = 4  # Define the number of columns for the grid
    num_rows = (num_atoms + num_cols - 1) // num_cols  # Calculate the number of rows needed

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 10))
    axs = axs.flatten()  # Flatten the 2D array of axes for easier iteration

    for i, atom in enumerate(unique_atoms):
        filtered_df = xray_transitions[xray_transitions['Atom'] == atom]
        
        # Define the energy range for the plot
        E_min = filtered_df['E'].min() - 10
        E_range = np.linspace(E_min, E_max, 2000)

        # Initialize the spectrum
        spectrum = np.zeros_like(E_range)

        E_values = filtered_df['E'].values
        width_values = filtered_df['width'].values
        amplitude_values = filtered_df['OS'].values
        
        # Create a 2D array for E_range to enable broadcasting
        E_range_2D = E_range[:, np.newaxis]
        
        # Compute the Gaussians for all rows at once
        gaussians = (amplitude_values / (width_values * np.sqrt(2 * np.pi))) * \
                    np.exp(-((E_range_2D - E_values) ** 2) / (2 * width_values ** 2))
        
        # Sum the Gaussians along the second axis (columns)
        spectrum = np.sum(gaussians, axis=1)

        # Plot the spectrum
        ax1 = axs[i]
        ax1.plot(E_range, spectrum, label='Spectrum')
        ax1.set_xlabel('Energy (eV)')
        ax1.set_ylabel('Intensity')
        ax1.set_xlim([E_min, E_max])
        ax1.set_title(f'Spectrum for {atom}')
        
        # Create a secondary y-axis for the OSCL vs E (eV) plot
        ax2 = ax1.twinx()
        # Plot the vertical lines using a single call to vlines
        ax2.vlines(x=E_values, ymin=0, ymax=amplitude_values, color='r')
        ax2.set_ylabel('OS')
        ax2.set_xlim([E_min, E_max])
        
        # Create a single custom legend entry for the sticks
        if not filtered_df.empty:
            custom_line = Line2D([0], [0], color='r', lw=2, label='OS')
        
        # Title and legend
        ax1.legend(loc='upper left')
        if not filtered_df.empty:
            ax2.legend(handles=[custom_line], loc='upper right')

    # Remove any empty subplots
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    fig.tight_layout()
    # Display the plot in Streamlit
    st.pyplot(fig)

def plot_total_spectra(xray_transitions, E_max, molName, os_col, os_ylim=1.0):
    """
    Generates and plots the total spectra from the sum of all the Gaussians for each unique atom in the xray_transitions DataFrame.

    Parameters:
    xray_transitions (pd.DataFrame): DataFrame containing the x-ray transitions with 'Atom', 'E (eV)', 'width', and 'OSCL' columns.
    E_max (float): Maximum energy value for the x-axis range.
    """
    
    # Define the energy range for the plot
    E_min = xray_transitions['E'].min() - 1
    E_range = np.linspace(E_min, E_max, 2000)
    
    # Initialize the total spectrum
    total_spectrum = np.zeros_like(E_range)

    unique_atoms = xray_transitions['Atom'].unique()

    for atom in unique_atoms:
        filtered_df = xray_transitions[xray_transitions['Atom'] == atom]
        
        E_values = filtered_df['E'].values
        width_values = filtered_df['width'].values
        amplitude_values = filtered_df[os_col].values
        
        # Create a 2D array for E_range to enable broadcasting
        E_range_2D = E_range[:, np.newaxis]
        
        # Compute the Gaussians for all rows at once
        gaussians = (amplitude_values / (width_values * np.sqrt(2 * np.pi))) * \
                    np.exp(-((E_range_2D - E_values) ** 2) / (2 * width_values ** 2))
        
        # Sum the Gaussians along the second axis (columns)
        total_spectrum += np.sum(gaussians, axis=1)

    # Plot the total spectrum
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(E_range, total_spectrum, label='Total Spectrum')
    ax1.set_xlabel('Energy (eV)')
    ax1.set_ylabel('Intensity')
    ax1.set_xlim([E_min, E_max])
    ax1.set_title(f'Total Spectrum from {molName}')
    
    # Create a secondary y-axis for the OSCL vs E (eV) plot
    ax2 = ax1.twinx()
    
    # Plot the vertical lines for each transition in red
    for atom in unique_atoms:
        filtered_df = xray_transitions[xray_transitions['Atom'] == atom]
        E_values = filtered_df['E'].values
        amplitude_values = filtered_df['OS'].values
        ax2.vlines(x=E_values, ymin=0, ymax=amplitude_values, color='r')

    ax2.set_ylabel('OS')
    ax2.set_xlim([E_min, E_max])
    if os_col == 'normalized_os':
        ax2.set_ylim([0, os_ylim])
    
    # Ensure the 0 of both y-axes matches up
    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)
    
    # Create a single custom legend entry for the sticks
    if not xray_transitions.empty:
        custom_line = Line2D([0], [0], color='r', lw=2, label='OS')
        ax2.legend(handles=[custom_line], loc='upper right')

    ax1.legend(loc='upper left')
    
    fig.tight_layout()
    # Display the plot in Streamlit
    st.pyplot(fig)
    
def find_core_hole_homo_lumo(orbital_alpha):
    """
    Identifies the Core Hole, Highest Occupied Molecular Orbital (HOMO), and Lowest Unoccupied Molecular Orbital (LUMO)
    for each unique Atom and Originating File in the given DataFrame.

    Args:
        orbital_alpha (pd.DataFrame): A DataFrame containing orbital information with columns 'Atom', 'Originating File',
                                      'Occup.', and 'MO_Index'.

    Returns:
        pd.DataFrame: A DataFrame with columns 'Atom', 'File', 'Core Hole', 'HOMO', and 'LUMO', containing the 
                      identified core hole, HOMO, and LUMO for each unique atom and originating file.
    """
    results = []
    
    # Convert Occup. column to float
    orbital_alpha['Occup.'] = orbital_alpha['Occup.'].astype(float)
    

    # Group by unique Atom and Originating File
    grouped = orbital_alpha.groupby(['Atom', 'Originating File'])
    
    for (atom, file), group in grouped:
        core_hole, homo, lumo = None, None, None
        
        core_hole_row = group[group['Occup.'] == 0.5]
        if not core_hole_row.empty:
            core_hole = core_hole_row.iloc[0]['MO_Index']
    
        homo_rows = group[group['Occup.'] == 1.0]
        if not homo_rows.empty:
            homo_index = homo_rows.index[-1]
            homo = homo_rows.iloc[-1]['MO_Index']
            lumo_index = homo_index + 1
            if lumo_index in group.index:
                lumo = group.loc[lumo_index]['MO_Index']
        
        results.append({
            'Atom': atom,
            'File': file,
            'Core Hole': core_hole,
            'HOMO': homo,
            'LUMO': lumo
        })
    
    return pd.DataFrame(results)

def plot_density_spectra(xray_transitions,E_max,os_col):
    """
    Generates a 2D density plot for the xray_transitions DataFrame.

    Parameters:
    xray_transitions (pd.DataFrame): DataFrame containing the x-ray transitions with 'E (eV)', 'theta', and 'OSCL' columns.
    """
    
    filtered_transitions = xray_transitions[xray_transitions['E'] < E_max]
    
    # Create the 2D KDE plot using seaborn
    fig, ax = plt.subplots(figsize=(10, 6))

    # KDE plot with OSCL as weights
    sns.kdeplot(
        x=filtered_transitions['E'],
        y=filtered_transitions['theta'],
        weights=filtered_transitions[os_col],
        fill=True,
        cmap="viridis",
        ax=ax
    )
    
    ax.set_xlim([filtered_transitions['E'].min()-10, E_max+10])
    ax.set_ylim([-50, 125])
    ax.set_xlabel('Energy (eV)')
    ax.set_ylabel('Theta (degrees)')
    ax.set_title('2D KDE Plot of X-ray Transitions')

    fig.tight_layout()
    # Display the plot in Streamlit
    st.pyplot(fig)

def filter_and_normalize_xray(df, maxE, OST):
    """
    Filters the dataframe based on the maxE and OST parameters, and adds a normalized_os column.

    Parameters:
    df (pd.DataFrame): The input dataframe with columns E, OS, and width.
    maxE (float): The maximum value for the E column.
    OST (float): The threshold as a percentage for the OS column.

    Returns:
    pd.DataFrame: The filtered and modified dataframe.
    """
    # Filter out rows where E is greater than maxE
    df_filtered = df[df['E'] <= maxE]

    # Normalize the OS column
    max_os = df_filtered['OS'].max()
    df_filtered['normalized_os'] = df_filtered['OS'] / max_os

    # Filter out rows where normalized_os is less than OST and sort by ascending energy
    df_filtered = df_filtered[df_filtered['normalized_os'] >= OST / 100]
    df_filtered = df_filtered.sort_values(by='E')
    df_filtered = df_filtered.reset_index(drop=True)
    
    return df_filtered

def gaussian(x, mu, sigma, amplitude):
    """
    Calculates the value of a Gaussian function for given input parameters.

    Args:
        x (float or np.ndarray): The input value(s) where the Gaussian function is evaluated.
        mu (float): The mean (center) of the Gaussian function.
        sigma (float): The standard deviation (width) of the Gaussian function.
        amplitude (float): The amplitude (height) of the Gaussian function.

    Returns:
        float or np.ndarray: The value(s) of the Gaussian function at the given input x.
    """
    return amplitude * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

def gaussian_area(mu, sigma, amplitude):
    """
    Calculates the area under a Gaussian function.

    Args:
        mu (float): The mean (center) of the Gaussian function (not used in the calculation).
        sigma (float): The standard deviation (width) of the Gaussian function.
        amplitude (float): The amplitude (height) of the Gaussian function.

    Returns:
        float: The area under the Gaussian function.
    """
    return amplitude * sigma * np.sqrt(2 * np.pi)

def overlap_area(mu1, sigma1, amplitude1, mu2, sigma2, amplitude2):
    """
    Calculates the overlap area between two Gaussian peaks.

    Args:
        mu1 (float): The mean (center) of the first Gaussian function.
        sigma1 (float): The standard deviation (width) of the first Gaussian function.
        amplitude1 (float): The amplitude (height) of the first Gaussian function.
        mu2 (float): The mean (center) of the second Gaussian function.
        sigma2 (float): The standard deviation (width) of the second Gaussian function.
        amplitude2 (float): The amplitude (height) of the second Gaussian function.

    Returns:
        float: The overlap area between the two Gaussian functions.
    """
    integrand = lambda x: np.minimum(gaussian(x, mu1, sigma1, amplitude1), gaussian(x, mu2, sigma2, amplitude2))
    # Integrate over a range wide enough to cover both peaks
    lower_bound = min(mu1 - 3 * sigma1, mu2 - 3 * sigma2)
    upper_bound = max(mu1 + 3 * sigma1, mu2 + 3 * sigma2)
    overlap, _ = quad(integrand, lower_bound, upper_bound)
    return overlap

def calculate_percent_overlap(df, n_jobs=-1):
    """
    Calculates the percent overlap matrix for peaks in the DataFrame.

    Args:
        df (pd.DataFrame): A DataFrame containing the columns 'E' (energy), 'width', and 'OS' (oscillator strength).
        n_jobs (int): The number of parallel jobs to run. Defaults to -1 (use all available processors).

    Returns:
        pd.DataFrame: A DataFrame representing the percent overlap matrix, where each element (i, j) is the percent overlap
                      between the peaks at index i and index j.
    """
    df = df.sort_values(by='E').reset_index(drop=True)
    n = len(df)
    overlap_matrix = np.zeros((n, n))

    progress_bar = st.progress(0)
    progress_text = st.empty()
    timer_text = st.empty()

    mu = df['E'].values
    sigma = df['width'].values
    amplitude = df['OS'].values

    total_areas = gaussian_area(mu, sigma, amplitude)

    start_time = time.time()

    def compute_overlap_row(i):
        overlaps = np.array([overlap_area(mu[i], sigma[i], amplitude[i], mu[j], sigma[j], amplitude[j]) for j in range(i, n)])
        percent_overlaps = (overlaps / np.minimum(total_areas[i], total_areas[i:])) * 100
        return i, percent_overlaps

    tasks = [delayed(compute_overlap_row)(i) for i in range(n)]
    results = Parallel(n_jobs=n_jobs)(tasks)

    for i, percent_overlaps in results:
        overlap_matrix[i, i:] = percent_overlaps
        overlap_matrix[i:, i] = percent_overlaps  # Symmetric matrix
        overlap_matrix[i, i] = 100.0

        if i % 10 == 0 or i == n - 1:
            progress = (i + 1) / n
            progress_bar.progress(progress)
            elapsed_time = time.time() - start_time
            progress_text.text(f'Calculating overlap: {i + 1}/{n} rows completed')
            timer_text.text(f'Time elapsed: {elapsed_time:.2f} seconds')

    return pd.DataFrame(overlap_matrix, index=df.index, columns=df.index)

def convert_to_distance_matrix(overlap_matrix):
    """
    Converts an overlap matrix to a distance matrix.

    Args:
        overlap_matrix (pd.DataFrame or np.ndarray): A square matrix representing the overlap percentages.

    Returns:
        pd.DataFrame or np.ndarray: A square matrix representing the distance, calculated as 100 - overlap.
    """
    distance_matrix = 100 - overlap_matrix
    return distance_matrix

def hierarchical_clustering(overlap_matrix, threshold):
    """
    Performs hierarchical clustering based on the overlap matrix.

    Args:
        overlap_matrix (pd.DataFrame or np.ndarray): A square matrix representing the overlap percentages.
        threshold (float): The distance threshold for forming clusters. Clusters are formed by cutting the dendrogram at this threshold.

    Returns:
        np.ndarray: An array of cluster labels for each element.
        np.ndarray: The hierarchical clustering encoded as a linkage matrix.
    """
    distance_matrix = convert_to_distance_matrix(overlap_matrix)
    condensed_distance_matrix = squareform(distance_matrix)
    Z = linkage(condensed_distance_matrix, method='ward')
    
    # Create clusters by cutting the dendrogram at a threshold
    clusters = fcluster(Z, threshold, criterion='distance')
    return clusters, Z

def combine_transitions(df):
    """
    Combines transitions within each cluster to form a representative Gaussian using weighted averages for E and a custom formula for width.

    Args:
        df (pd.DataFrame): A DataFrame containing transition data with columns 'cluster', 'E', 'width', and 'OS'.

    Returns:
        pd.DataFrame: A DataFrame with combined transitions, containing columns 'cluster', 'E', 'width', and 'OS'.
    """
    def weighted_average(group, avg_name, weight_name):
        """
        Calculate the weighted average.

        Args:
            group (pd.DataFrame): The group of rows for which to calculate the weighted average.
            avg_name (str): The name of the column to average.
            weight_name (str): The name of the column to use as weights.

        Returns:
            float: The weighted average.
        """
        d = group[avg_name]
        w = group[weight_name]
        return (d * w).sum() / w.sum()
    
    def weighted_width(group, E_weighted):
        """
        Calculate the weighted width.

        Args:
            group (pd.DataFrame): The group of rows for which to calculate the weighted width.
            E_weighted (float): The weighted average of the energy values.

        Returns:
            float: The weighted width.
        """
        width_contrib = ((group['width'] + group['E']) * group['OS']).sum()
        return (width_contrib / group['OS'].sum()) - E_weighted
    
    combined_df = df.groupby('cluster').apply(
        lambda x: pd.Series({
            'E': weighted_average(x, 'E', 'OS'),
            'width': weighted_width(x, weighted_average(x, 'E', 'OS')),
            'OS': x['OS'].max()
        })
    ).reset_index()
    
    # Sort by the 'E' column
    combined_df = combined_df.sort_values(by='E').reset_index(drop=True)
    
    # Reassign clusters in sequential order
    combined_df['cluster'] = np.arange(1, len(combined_df) + 1)
    
    return combined_df

def iterative_clustering(df, overlap_threshold, max_iterations=10, n_jobs=-1):
    """
    Iteratively clusters and combines transitions until no elements in the overlap matrix exceed the threshold.
    
    Args:
        df (pd.DataFrame): DataFrame containing the transition data.
        overlap_threshold (float): Threshold for the maximum allowable overlap between clusters.
        max_iterations (int): Maximum number of iterations for the clustering process.
        n_jobs (int): Number of jobs to run in parallel for calculating the overlap matrix.
    
    Returns:
        combined_df (pd.DataFrame): DataFrame with the final clustered transitions.
        percent_overlap_matrix (pd.DataFrame): DataFrame with the final percent overlap matrix.
        iteration (int): Number of iterations performed.
        final_Z (ndarray): Linkage matrix from the final hierarchical clustering.
    """
    iteration = 1
    percent_overlap_matrix = calculate_percent_overlap(df, n_jobs=n_jobs)
    clusters, Z = hierarchical_clustering(percent_overlap_matrix, overlap_threshold)
    df['cluster'] = clusters
    combined_df = combine_transitions(df)
    final_Z = Z

    while iteration < max_iterations:
        percent_overlap_matrix = calculate_percent_overlap(combined_df, n_jobs=n_jobs)
        clusters, Z = hierarchical_clustering(percent_overlap_matrix, overlap_threshold)
        
        # Enforce the desired number of clusters
        final_clusters = fcluster(Z, t=overlap_threshold, criterion='distance')
        combined_df['cluster'] = final_clusters
        combined_df = combine_transitions(combined_df)
        
        max_overlap = percent_overlap_matrix.values[np.triu_indices(len(combined_df), k=1)].max()
        if max_overlap <= overlap_threshold:
            final_Z = Z
            break
        
        iteration += 1
        final_Z = Z

    # Final clusters to enforce consistency
    final_clusters = fcluster(final_Z, t=overlap_threshold, criterion='distance')
    combined_df['final_cluster'] = final_clusters

    return combined_df, percent_overlap_matrix, iteration, final_Z

def visualize_overlap_matrix(overlap_matrix, title='Percent Overlap Matrix'):
    """
    Visualizes the overlap matrix using a heatmap with a different color for values below a specified threshold.

    Parameters:
    overlap_matrix (pd.DataFrame): The percent overlap matrix to visualize.
    title (str): Title of the heatmap.
    """
    fig = go.Figure(data=go.Heatmap(
        z=overlap_matrix.values,
        x=overlap_matrix.columns,
        y=overlap_matrix.index,
        colorscale='Viridis',
        colorbar=dict(title='Percent Overlap')
    ))

    fig.update_layout(
        #title=title,
        xaxis_title='Transition Index',
        yaxis_title='Transition Index',
        width=500,
        height=500
    )

    st.plotly_chart(fig)

def generate_clusters(overlap_matrix, threshold):
    """
    Generates clusters from the overlap matrix using hierarchical clustering.

    Parameters:
    overlap_matrix (pd.DataFrame): The percent overlap matrix.
    threshold (float): The threshold for clustering.

    Returns:
    np.ndarray: An array of cluster labels.
    """
    # Convert the overlap matrix to a condensed distance matrix
    condensed_distance_matrix = squareform(1 - overlap_matrix.values / 100)

    # Perform hierarchical/agglomerative clustering
    linkage_matrix = sch.linkage(condensed_distance_matrix, method='complete')

    # Generate cluster labels using the specified threshold
    cluster_labels = sch.fcluster(linkage_matrix, t=threshold, criterion='distance')

    return cluster_labels
 
def visualize_clusters(overlap_matrix, cluster_labels, title='Clustered Overlap Matrix'):
    """
    Visualizes the clustered overlap matrix using a heatmap with clusters indicated.

    Parameters:
    overlap_matrix (pd.DataFrame): The percent overlap matrix to visualize.
    cluster_labels (np.ndarray): Cluster labels for each transition.
    title (str): Title of the heatmap.
    """
    plt.figure(figsize=(10, 4))
    sns.heatmap(overlap_matrix, annot=False, cmap='viridis', cbar_kws={'label': 'Percent Overlap'})

    # Add cluster borders
    unique_clusters = np.unique(cluster_labels)
    for cluster in unique_clusters:
        indices = np.where(cluster_labels == cluster)[0]
        for i in indices:
            for j in indices:
                plt.gca().add_patch(plt.Rectangle((j, i), 1, 1, fill=True, edgecolor='white', lw=2))

    plt.title(title)
    plt.xlabel('Transition Index')
    plt.ylabel('Transition Index')
    st.pyplot(plt.gcf())

def plot_clustered_spectrum(df, original_df, Emax):
    """
    Plots the clustered spectrum and individual cluster peaks, along with the original transitions spectrum.

    Args:
        df (pd.DataFrame): DataFrame containing the clustered peak parameters.
                           Columns should include 'E', 'width', 'OS', and 'cluster'.
        original_df (pd.DataFrame): DataFrame containing the original transitions.
                           Columns should include 'E', 'width', and 'OS'.
    """
    
    x = np.linspace(original_df['E'].min() - 2, Emax, 2000)

    # Function to calculate the Gaussian for multiple rows at once
    def calculate_gaussian_spectrum(E_range, E_values, width_values, amplitude_values):
        E_range_2D = E_range[:, np.newaxis]
        gaussians = (amplitude_values / (width_values * np.sqrt(2 * np.pi))) * \
                    np.exp(-((E_range_2D - E_values) ** 2) / (2 * width_values ** 2))
        return np.sum(gaussians, axis=1)

    # Plot each cluster and add to the total spectrum
    clusters = df['cluster'].unique()
    total_spectrum = np.zeros_like(x)

    # Generate colors from a built-in Plotly color scale
    color_scale = plotly.colors.qualitative.Prism

    fig = go.Figure()

    for i, cluster in enumerate(clusters):
        cluster_df = df[df['cluster'] == cluster]
        cluster_spectrum = calculate_gaussian_spectrum(x, cluster_df['E'].values, cluster_df['width'].values, cluster_df['OS'].values)
        total_spectrum += cluster_spectrum
        fig.add_trace(go.Scatter(
            x=x, 
            y=cluster_spectrum, 
            mode='lines', 
            name=f'Cluster {cluster}',
            line=dict(color=color_scale[i % len(color_scale)], width=2)
        ))

    # Plot the total clustered spectrum
    fig.add_trace(go.Scatter(
        x=x, 
        y=total_spectrum, 
        mode='lines', 
        name='Clustered NEXAFS', 
        line=dict(color='white', width=2)
    ))

    # Plot the original transitions
    original_spectrum = calculate_gaussian_spectrum(x, original_df['E'].values, original_df['width'].values, original_df['OS'].values)
    fig.add_trace(go.Scatter(
        x=x, 
        y=original_spectrum, 
        mode='lines', 
        name='Original NEXAFS', 
        line=dict(color='blue', width=2, dash='dash')
    ))

    fig.update_layout(
        xaxis_title='Energy [eV]',
        yaxis_title='Intensity',
        legend_title='Spectra',
        template='plotly_white',
        width=900,
        height=600,
    )

    st.plotly_chart(fig)

def plot_dendrogram(final_overlap_matrix, Z):
    """
    Plots a dendrogram and heatmap of the overlap matrix.

    Args:
        final_overlap_matrix (pd.DataFrame): The final overlap matrix to plot.
        Z (np.ndarray): The linkage matrix resulting from hierarchical clustering.
    
    Returns:
        None
    """
    sns.set(context="notebook", font_scale=1.2)
    
    # Create a dendrogram to get the leaf order
    dendrogram = sns.clustermap(final_overlap_matrix, row_linkage=Z, col_linkage=Z, cmap="viridis", figsize=(10, 8))
    
    # Extract the order of the leaves
    row_order = dendrogram.dendrogram_row.reordered_ind
    col_order = dendrogram.dendrogram_col.reordered_ind
    
    # Reorder the dataframe
    final_overlap_matrix_ordered = final_overlap_matrix.iloc[row_order, col_order]
    
    # Plot the heatmap with reordered matrix
    g = sns.clustermap(final_overlap_matrix_ordered, row_linkage=Z, col_linkage=Z, cmap="viridis", figsize=(10, 8))
    
    # Add title and labels
    g.ax_heatmap.set_xlabel("Clusters")
    g.ax_heatmap.set_ylabel("Clusters")
    
    # Label colorbar
    colorbar = g.cax
    colorbar.set_title('% Overlap')
    
    # Show plot in Streamlit
    st.pyplot(g.fig)

def plot_heatmap(final_overlap_matrix):
    """
    Plots a heatmap of the overlap matrix using Plotly.

    Args:
        final_overlap_matrix (pd.DataFrame): The final overlap matrix to plot.

    Returns:
        None
    """
    fig = px.imshow(final_overlap_matrix, 
                    color_continuous_scale='Viridis',
                    labels=dict(x="Clusters", y="Clusters", color="% Overlap"))
    
    fig.update_layout(
                      xaxis_title="Clusters",
                      yaxis_title="Clusters",
                      height = 600,
                      width = 500)
    
    st.plotly_chart(fig)
    
def save_dataframe_to_csv_in_memory(dataframe, filename):
    """
    Saves a DataFrame to a CSV file in memory.

    Args:
        dataframe (pd.DataFrame): The DataFrame to save.
        filename (str): The name of the CSV file.

    Returns:
        tuple: A tuple containing the in-memory CSV buffer and the filename.
    """
    buffer = BytesIO()
    if dataframe is not None:
        dataframe.to_csv(buffer, index=False)
    buffer.seek(0)
    return buffer, filename

def zip_dataframes(dataframes_dict, zip_filename):
    """
    Zips multiple DataFrames into a single zip file in memory.

    Args:
        dataframes_dict (dict): A dictionary where keys are filenames and values are DataFrames.
        zip_filename (str): The name of the resulting zip file.

    Returns:
        BytesIO: The in-memory zip file buffer.
    """
    zip_buffer = BytesIO()
    with ZipFile(zip_buffer, "a") as zf:
        for name, df in dataframes_dict.items():
            if df is not None:  # Check if the dataframe is not None
                csv_buffer, filename = save_dataframe_to_csv_in_memory(df, f"{name}.csv")
                zf.writestr(filename, csv_buffer.read())
    zip_buffer.seek(0)
    return zip_buffer

def load_data(directory, width1, width2, maxEnergy):
    """
    Loads data from a directory, processes it, and returns the results as a dictionary.

    Args:
        directory (str): The path to the directory containing data files.
        width1 (float): The width parameter for processing.
        width2 (float): The second width parameter for processing.
        maxEnergy (float): The maximum energy value for processing.

    Returns:
        dict: A dictionary containing the processed data as DataFrames, or None if the directory is invalid.
    """
    if os.path.isdir(directory):
        st.write('Processing directory:', directory)
        progress_bar = st.progress(0)
        progress_text = st.empty()
        start_time = time.time()
        # Process the directory
        basis_sets, energy_results, orbital_alpha, orbital_beta, xray_transitions, atomic_coordinates = process_directory(
            directory, progress_bar, progress_text, width1, width2, 290, maxEnergy)
        end_time = time.time()
        st.write(f'Processing completed in {end_time - start_time:.2f} seconds.')
        return {
            'basis_sets': basis_sets,
            'energy_results': energy_results,
            'orbital_alpha': orbital_alpha,
            'orbital_beta': orbital_beta,
            'xray_transitions': xray_transitions,
            'atomic_coordinates': atomic_coordinates
        }
    else:
        st.write('Invalid directory. Please enter a valid directory path.')
        return None

def display_initial_data(data):
    df = data['atomic_coordinates']
    unique_file = df['Originating File'].unique()[0]
    filtered_atomic_coordinates = df[df['Originating File'] == unique_file]

    col1, col2, col3 = st.columns([2, 3, 3])
    with col1:
        st.write('### Atomic Coordinates')
        st.dataframe(filtered_atomic_coordinates)
    with col2:
        st.write('### Basis Sets')
        st.dataframe(data['basis_sets'])
    with col3:
        st.write('### Molecule Visualization')
        view = visualize_xyz_with_stmol(filtered_atomic_coordinates, "molecule.xyz")
        showmol(view, height=400, width=400)
        
    col1, col2, col3, col4 = st.columns([4, 2, 2, 2])    
    with col1:
        st.write('### Molecule Energies')
        st.dataframe(data['energy_results'])
    with col2:
        st.write('### Orbital Alpha')
        st.dataframe(data['orbital_alpha'])
    with col3:
        st.write('### Orbital Beta')
        st.dataframe(data['orbital_beta'])
    with col4:
        st.write('### Core Hole, HOMO, and LUMO')
        core_hole_homo_lumo_df = find_core_hole_homo_lumo(data['orbital_alpha'])
        st.dataframe(core_hole_homo_lumo_df)

def display_filtered_data(data, maxEnergy, OST, molName):
    filtered_xray_transitions = filter_and_normalize_xray(data['xray_transitions'], maxEnergy, OST)

    col1, col2, col3 = st.columns([3, 2, 2])
    with col1:
        st.write('### Energy and OS Filtered X-ray Transitions')
        st.dataframe(filtered_xray_transitions)
    with col2:
        st.write('### Energy and OS Filtered DFT NEXAFS')
        plot_total_spectra(filtered_xray_transitions, maxEnergy, molName, 'normalized_os', (OST / 100) * 2)
    with col3:
        st.write('### Energy and OS Filtered KDE')
        plot_density_spectra(filtered_xray_transitions, maxEnergy, 'normalized_os')
    st.write('### Excitation Centers DFT NEXAFS')
    plot_individual_spectra(data['xray_transitions'], maxEnergy)
    
    return filtered_xray_transitions

def display_clustering(data, OVPT, maxEnergy, molName):
    final_df, final_overlap_matrix, iterations, Z = iterative_clustering(data, OVPT, n_jobs=8)
    print(Z)

    col1, col2, col3 = st.columns([1,2,3])

    with col1:
        st.write('### Final Clusters')
        st.write(f'Final Clusters after {iterations} iterations:')
        st.dataframe(final_df)

    with col2:
        st.write('### Dendrogram for Clusters')
        plot_heatmap(final_overlap_matrix)
        
    with col3:
        st.write('### Original vs Clustered DFT NEXAFS')
        plot_clustered_spectrum(final_df, data, maxEnergy)

    return final_df

def main():
    """
    Main function to run the Streamlit application for StoBe Loader and Clustering Algorithm.

    This application allows the user to load data from a specified directory, display initial and filtered data,
    and perform clustering on the processed data. The results can be downloaded as a ZIP file.
    """
    st.set_page_config(layout="wide")
    st.title('StoBe Loader for Clustering Algorithm')

    col1, col2, col3, col4, col5, col6, col7 = st.columns([2, 1, 1, 1, 1, 1, 1])
    with col1:
        directory = st.text_input('Enter the directory containing the output folders:')
    with col2:
        width1 = st.number_input('Width 1 (eV)', min_value=0.01, max_value=20.0, value=0.5) / 2.355
    with col3:
        width2 = st.number_input('Width 2 (eV)', min_value=0.01, max_value=20.0, value=12.0) / 2.355
    with col4:
        maxEnergy = st.number_input('Maximum DFT Energy (eV)', min_value=0.0, max_value=1000.0, value=320.0)
    with col5:
        molName = st.text_input('Enter the name of your molecule:')
    with col6:
        OST = st.number_input('OS Threshold (%)', min_value=0.0, max_value=100.0, value=10.0)
    with col7:
        OVPT = st.number_input('OVP Threshold (%)', min_value=0.0, max_value=100.0, value=50.0)

    routine = st.radio("Select Routine", ("Initial Data Loading", "Initial Data Displays", "Filtered Data Displays", "Clustering"))

    if routine == "Initial Data Loading":
        if directory and st.button('Process Directory'):
            data = load_data(directory, width1, width2, maxEnergy)
            if data:
                st.session_state['processed_data'] = data
    elif routine == "Initial Data Displays":
        if 'processed_data' in st.session_state:
            display_initial_data(st.session_state['processed_data'])
    elif routine == "Filtered Data Displays":
        if 'processed_data' in st.session_state:
            filtered_data = display_filtered_data(st.session_state['processed_data'], maxEnergy, OST, molName)
            st.session_state['filtered_data'] = filtered_data
    elif routine == "Clustering":
        if 'filtered_data' in st.session_state:
            final_df = display_clustering(st.session_state['filtered_data'], OVPT, maxEnergy, molName)
            st.session_state['final_df'] = final_df

    if 'processed_data' in st.session_state:
        data_to_save = {
            f"{molName}_basis_sets": st.session_state['processed_data'].get('basis_sets'),
            f"{molName}_energy_results": st.session_state['processed_data'].get('energy_results'),
            f"{molName}_orbital_alpha": st.session_state['processed_data'].get('orbital_alpha'),
            f"{molName}_orbital_beta": st.session_state['processed_data'].get('orbital_beta'),
            f"{molName}_xray_transitions": st.session_state['processed_data'].get('xray_transitions'),
            f"{molName}_atomic_coordinates": st.session_state['processed_data'].get('atomic_coordinates'),
            f"{molName}_final_df": st.session_state.get('final_df'),
            f"{molName}_core_hole_homo_lumo": find_core_hole_homo_lumo(st.session_state['processed_data'].get('orbital_alpha'))
        }
        zip_filename = f"{molName}_dataframes.zip"
        zip_buffer = zip_dataframes(data_to_save, zip_filename)
        st.download_button(
            label="Download ZIP",
            data=zip_buffer,
            file_name=zip_filename,
            mime='application/zip'
        )

if __name__ == "__main__":
    main()
