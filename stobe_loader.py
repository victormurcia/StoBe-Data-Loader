# -*- coding: utf-8 -*-
"""
Created on Wed May 29 17:35:57 2024

@author: vmurc
"""

import os
import re
import streamlit as st
import pandas as pd
import time
from natsort import natsorted

def parse_basis_line(line):
    """
    Parses a single line of basis set information.
    
    Parameters:
    line (str): A line containing basis set information.
    
    Returns:
    tuple: A tuple containing the atom and the basis set information.
    """
    parts = line.split(':', 1)  # Split only on the first colon
    if len(parts) == 2:
        atom = parts[0].split()[-1]  # Get the last element after splitting by space
        basis = parts[1].strip()
        return atom, basis
    return None, None

def extract_orbital_data(file_path):
    """
    Extracts spin alpha and beta orbital data from a formatted text file.

    Args:
        file_path: Path to the text file containing the orbital data.

    Returns:
        A tuple of two Pandas DataFrames: (df_alpha, df_beta)

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file format is incorrect or the data cannot be parsed.
    """

    data_alpha = []
    data_beta = []

    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()

            try:  
                start_index = lines.index("         Spin alpha                              Spin beta\n") + 2
            except ValueError:
                raise ValueError("Invalid file format: Could not find the data start marker.")

            for line in lines[start_index:]:
                if line.strip() == "" or line.startswith(" IV)"):
                    break

                components = [x for x in line.split() if x.strip()]
                if len(components) < 9:  # Ensure we have enough elements
                    raise ValueError("Invalid line format: Expected at least 9 elements.")
                
                mo_index_alpha, occup_alpha, energy_alpha, sym_alpha = components[:4]
                mo_index_beta, occup_beta, energy_beta, sym_beta = components[5:9]

                mo_index_alpha = mo_index_alpha.strip(')')
                mo_index_beta = mo_index_beta.strip(')')

                data_alpha.append({
                    "MO_Index": mo_index_alpha, 
                    "Occup.": occup_alpha, 
                    "Energy(eV)": energy_alpha, 
                    "Sym.": sym_alpha
                })
                data_beta.append({
                    "MO_Index": mo_index_beta, 
                    "Occup.": occup_beta, 
                    "Energy(eV)": energy_beta, 
                    "Sym.": sym_beta
                })

    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")

def extract_basis_sets(file_path, orbital_type, verbose=False):
    """
    Extracts the basis sets information from a file containing results of transition potential density functional theory calculations.
    
    The function reads the file, identifies the section containing the specified basis sets, and extracts the relevant data into a pandas DataFrame.
    
    Parameters:
    file_path (str): The path to the file containing the calculation results.
    orbital_type (str): The type of orbital to extract ('auxiliary', 'orbital', or 'model').
    verbose (bool): If True, prints debug information. Default is False.
    
    Returns:
    pd.DataFrame: A DataFrame containing the basis sets information with columns ['Atom', column_name].
    """
    # Determine start_marker, end_markers, and column_name based on orbital_type
    if orbital_type == 'auxiliary':
        start_marker = "I)  AUXILIARY BASIS SETS"
        end_markers = ["II)  ORBITAL BASIS SETS"]
        column_name = "Auxiliary Basis"
    elif orbital_type == 'orbital':
        start_marker = "II)  ORBITAL BASIS SETS"
        end_markers = ["BASIS DIMENSIONS"]
        column_name = "Orbital Basis"
    elif orbital_type == 'model':
        start_marker = "III)  MODEL POTENTIALS"
        end_markers = ["WARNING! Electron count may be inconsistent:", "(NEW) SYMMETRIZATION INFORMATION"]
        column_name = "Model Potential"
    else:
        raise ValueError("Invalid orbital_type. Expected 'auxiliary', 'orbital', or 'model'.")

    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Find the start and end indices for the basis sets section
    start_index = None
    end_index = None
    for i, line in enumerate(lines):
        if start_marker in line:
            start_index = i + 1
        elif any(end_marker in line for end_marker in end_markers):
            end_index = i
            break

    if start_index is None:
        return pd.DataFrame(columns=['Atom', column_name])

    if end_index is None:
        end_index = len(lines)

    basis_lines = lines[start_index:end_index]

    # Parse the lines
    data = []
    for line in basis_lines:
        if line.strip():
            atom, basis = parse_basis_line(line)
            if atom and basis:
                data.append([atom, basis])

    return pd.DataFrame(data, columns=['Atom', column_name])

def extract_energies(file_path, originating_atom):
    """
    Extracts the energy information from the given file.
    
    Parameters:
    file_path (str): The path to the file containing the calculation results.
    originating_atom (str): The atom from which the data is extracted.
    
    Returns:
    pd.DataFrame: A DataFrame containing the energy information.
    """
    energies = {
        "Total energy (H)": None,
        "Nuc-nuc energy (H)": None,
        "El-nuc energy (H)": None,
        "Kinetic energy (H)": None,
        "Coulomb energy (H)": None,
        "Ex-cor energy (H)": None,
        "Atom": originating_atom,
        "Calculation Type": "TP" if "_tp.out" in file_path else "GND" if "gnd.out" in file_path else "EXC"
    }
    
    with open(file_path, 'r') as file:
        for line in file:
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
    
    df = pd.DataFrame([energies])
    
    # Reorder columns to have 'Atom' first
    cols = ['Atom'] + [col for col in df.columns if col != 'Atom']
    df = df[cols]
    
    # Sort the DataFrame by 'Atom' using natural sorting
    df = df.sort_values(by='Atom', key=lambda x: natsorted(x))
    
    return df

def sort_dataframe_naturally(df, column):
    """
    Sorts a DataFrame naturally by a specified column.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to sort.
    column (str): The column name to sort by.
    
    Returns:
    pd.DataFrame: The naturally sorted DataFrame.
    """
    df[column] = df[column].astype(str)
    sorted_index = natsorted(df[column].tolist())
    df = df.set_index(column).loc[sorted_index].reset_index()
    return df

def process_directory(directory, progress_bar, progress_text):
    combined_results = pd.DataFrame()
    energy_results = pd.DataFrame()
    pattern = re.compile(r'^[A-Za-z]+\d+$')  # Pattern for folders with naming structure {chemical_symbol}{atom_number}
    entries = [entry for entry in os.listdir(directory) if os.path.isdir(os.path.join(directory, entry)) and pattern.match(entry)]
    
    # Calculate total number of files
    total_files = 0
    for entry in entries:
        entry_path = os.path.join(directory, entry)
        for file in os.listdir(entry_path):
            if file.endswith(('_tp.out', 'gnd.out', 'exc.out')):
                total_files += 1

    st.write(f'Total files to process: {total_files}')
    
    processed_files = 0
    for entry in entries:
        entry_path = os.path.join(directory, entry)
        originating_atom = entry  # The originating atom is the folder name
        for file in os.listdir(entry_path):
            if file.endswith(('_tp.out', 'gnd.out', 'exc.out')):
                file_path = os.path.join(entry_path, file)
                
                # Extract basis sets only for tp.out files
                if file.endswith('_tp.out'):
                    df_auxiliary = extract_basis_sets(file_path, orbital_type='auxiliary')
                    df_orbital = extract_basis_sets(file_path, orbital_type='orbital')
                    df_model = extract_basis_sets(file_path, orbital_type='model')
                    
                    # Add 'Originating File' column
                    df_auxiliary['Originating File'] = file
                    df_orbital['Originating File'] = file
                    df_model['Originating File'] = file
                    
                    # Combine dataframes using inner join on 'Atom' and 'Originating File'
                    df_combined = df_auxiliary.merge(df_orbital, on=['Atom', 'Originating File'], how='outer').merge(df_model, on=['Atom', 'Originating File'], how='outer')
                    
                    # Concatenate results
                    combined_results = pd.concat([combined_results, df_combined], ignore_index=True)
                
                # Extract energy information
                df_energies = extract_energies(file_path, originating_atom)
                energy_results = pd.concat([energy_results, df_energies], ignore_index=True)
                
                # Update progress bar with percentage
                processed_files += 1
                percentage_complete = min((processed_files / total_files), 1.0)
                progress_bar.progress(percentage_complete)
                progress_text.text(f'Processing: {percentage_complete*100:.2f}% completed.')
    
    energy_results = sort_dataframe_naturally(energy_results, 'Atom')
    return combined_results, energy_results


st.title('X-ray Absorption Spectrum Analysis')

directory = st.text_input('Enter the directory containing the output folders:')
if directory:
    if os.path.isdir(directory):
        st.write('Processing directory:', directory)
        
        # Initialize progress bar and text
        progress_bar = st.progress(0)
        progress_text = st.empty()
        
        # Process directory and update progress bar
        start_time = time.time()
        combined_results, energy_results = process_directory(directory, progress_bar, progress_text)
        end_time = time.time()
        
        st.write('### Combined Basis Sets Results')
        st.dataframe(combined_results)
        st.write('### Combined Energy Results')
        st.dataframe(energy_results)
        st.write(f'Processing completed in {end_time - start_time:.2f} seconds.')
    else:
        st.write('Invalid directory. Please enter a valid directory path.')
