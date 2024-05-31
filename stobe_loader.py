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

def parse_basis_line(line):
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
                    "E (eV)": e_ev,
                    "OSCL": oscl,
                    "oslx": oslx,
                    "osly": osly,
                    "oslz": oslz,
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
    df[column] = df[column].astype(str)
    sorted_index = natsorted(df[column].tolist())
    df = df.set_index(column).loc[sorted_index].reset_index()
    return df

def process_file(file_info):
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
        if E < ewid1:
            return width1
        elif E > ewid2:
            return width2
        else:
            return width1 + (width2 - width1) * (E - ewid1) / (ewid2 - ewid1)
    
    xray_transitions['width'] = xray_transitions['E (eV)'].apply(broad)
    
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

def visualize_xyz_with_stmol(df, file_name):
    """
    Visualize an XYZ file using Stmol and py3Dmol.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing atomic coordinates and labels.
    file_name (str): The name of the XYZ file to read.
    """
    # Read the XYZ file
    with open(file_name, 'r') as f:
        xyz_data = f.read()
    
    # Create a 3Dmol.js viewer
    view = py3Dmol.view(width=800, height=600)
    view.addModel(xyz_data, 'xyz')
    view.setStyle({'stick': {}})
    
    # Add labels based on the Atom column with atom numbers
    atom_counters = {}
    for i, row in df.iterrows():
        atom_symbol = ''.join([i for i in row['Atom'] if not i.isdigit()])  # Strip numbers from atom label
        if atom_symbol not in atom_counters:
            atom_counters[atom_symbol] = 0
        atom_counters[atom_symbol] += 1
        label = f"{atom_symbol}{atom_counters[atom_symbol]}"
        x, y, z = row['x'], row['y'], row['z']
        view.addLabel(label, {'position': {'x': x, 'y': y, 'z': z}, 'backgroundColor': 'white', 'fontColor': 'black'})
    
    view.zoomTo()
    return view

def plot_spectra(xray_transitions, E_max):
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
        E_min = filtered_df['E (eV)'].min() - 10
        E_range = np.linspace(E_min, E_max, 2000)

        # Initialize the spectrum
        spectrum = np.zeros_like(E_range)

        E_values = filtered_df['E (eV)'].values
        width_values = filtered_df['width'].values
        amplitude_values = filtered_df['OSCL'].values
        
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

def find_core_hole_homo_lumo(orbital_alpha):
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
        
def main():
    st.set_page_config(layout="wide")
    st.title('X-ray Absorption Spectrum Analysis')

    col1, col2, col3, col4, col5 = st.columns([2, 1, 1,1,1])

    with col1:
        directory = st.text_input('Enter the directory containing the output folders:')
    
    with col2:
        width1 = st.number_input('Width 1 (eV)', min_value=0.01, max_value=20.0, value=1.0)/2.355
    
    with col3:
        width2 = st.number_input('Width 2 (eV)', min_value=0.01, max_value=20.0, value=1.0)/2.355
        
    with col4:
        maxEnergy = st.number_input('Maximum DFT Energy (eV)', min_value=0.0, max_value=1000.0, value=1.0)



    if directory and st.button('Process Directory'):
        if os.path.isdir(directory):
            st.write('Processing directory:', directory)
            
            progress_bar = st.progress(0)
            progress_text = st.empty()
            
            start_time = time.time()
            # Process the directory
            basis_sets, energy_results, orbital_alpha, orbital_beta, xray_transitions, atomic_coordinates = process_directory(directory, progress_bar, progress_text, width1, width2, 290, maxEnergy)
            end_time = time.time()
            
            st.write(f'Processing completed in {end_time - start_time:.2f} seconds.')
            
            # Store the results in session state
            st.session_state['processed_data'] = {
                'basis_sets': basis_sets,
                'energy_results': energy_results,
                'orbital_alpha': orbital_alpha,
                'orbital_beta': orbital_beta,
                'xray_transitions': xray_transitions,
                'atomic_coordinates': atomic_coordinates
            }
        else:
            st.write('Invalid directory. Please enter a valid directory path.')

    if 'processed_data' in st.session_state:
        if st.button('Display Results'):
            data = st.session_state['processed_data']
            atomic_coordinates = data['atomic_coordinates']
            
            
            # Display Atomic Coordinates and Molecule Visualization side by side
            st.write('### Atomic Coordinates and Molecule Visualization')
            col1,col2  = st.columns([2,2])
            with col1:
                st.dataframe(atomic_coordinates)
            
            with col2:
                # Dropdown menu to filter based on Originating File
                unique_file = atomic_coordinates['Originating File'].unique()[0]
                #selected_file = st.selectbox('Select Originating File:', unique_files)
                
                # Filter the dataframe based on the selected file
                filtered_atomic_coordinates = atomic_coordinates[atomic_coordinates['Originating File'] == unique_file]
                
                if not filtered_atomic_coordinates.empty:
                    # Create XYZ file from filtered DataFrame
                    dataframe_to_xyz(filtered_atomic_coordinates, "molecule.xyz")
                    
                    # Visualize the molecule from the XYZ file using Stmol and py3Dmol
                    view = visualize_xyz_with_stmol(filtered_atomic_coordinates, "molecule.xyz")
                    showmol(view, height=400, width=400)
                else:
                    st.write("No data available for the selected file.")
                
            st.write('### Combined Basis Sets Results')
            st.dataframe(data['basis_sets'])
            st.write('### Combined Energy Results')
            st.dataframe(data['energy_results'])
            
            # Display alfa and beta orbital dataframes next to one another
            col1, col2,col3 = st.columns([2, 2, 2])
            with col1:
                st.write('### Orbital Alpha Data')
                st.dataframe(data['orbital_alpha'])
            with col2:
                st.write('### Orbital Beta Data')
                st.dataframe(data['orbital_beta'])
                
            # Finding Core Hole, HOMO and LUMO
            core_hole_homo_lumo_df = find_core_hole_homo_lumo(data['orbital_alpha'])
            with col3:
                st.write('### Core Hole, HOMO, and LUMO Results')
                st.dataframe(core_hole_homo_lumo_df)
            
            # Display X-ray Transitions Data and plot next to it
            col1, col2 = st.columns([2, 3])
            
            with col1:
                st.write('### X-ray Transitions Data')
                st.dataframe(data['xray_transitions'])
                
            with col2:
                plot_spectra(data['xray_transitions'], maxEnergy)

if __name__ == '__main__':
    main()
