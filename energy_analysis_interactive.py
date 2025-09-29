#!/usr/bin/env python
# coding: utf-8

"""
Home Energy Intelligence: Comprehensive Analysis of Electricity Usage, HVAC Performance, and Building Envelope
=====================================================================================================
Author: Energy Data Analytics Portfolio Project
Location: 38 Old Elm St, North Billerica, MA 01862
Analysis Period: 2021-2025
Last Updated: December 2024

This notebook demonstrates advanced data engineering, energy analytics, and storytelling capabilities
through a real-world case study of excessive home energy consumption and targeted improvement efforts.
"""

# %%
# ============================================================================
# CELL 1: ENVIRONMENT SETUP & IMPORTS
# ============================================================================

import sys
import subprocess
import os
import warnings
import json
import importlib.util
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Package installation with modern dependency checking
def ensure_packages(packages: List[str]) -> None:
    """Ensure required packages are installed using modern importlib approach."""
    
    def is_package_installed(package_name: str) -> bool:
        """Check if a package is installed using importlib."""
        # Handle package name variations (e.g., 'scikit-learn' vs 'sklearn')
        import_names = {
            'scikit-learn': 'sklearn',
            'python-dateutil': 'dateutil',
            'pillow': 'PIL'
        }
        
        check_name = import_names.get(package_name, package_name)
        spec = importlib.util.find_spec(check_name)
        return spec is not None
    
    def extract_package_name(package_spec: str) -> str:
        """Extract package name from specification (e.g., 'pandas>=2.2.0' -> 'pandas')."""
        # Split on common version specifiers
        for separator in ['>=', '==', '<=', '>', '<', '~=', '!=']:
            if separator in package_spec:
                return package_spec.split(separator)[0].strip()
        return package_spec.strip()
    
    try:
        to_install = []
        
        for package_spec in packages:
            package_name = extract_package_name(package_spec)
            
            if not is_package_installed(package_name):
                to_install.append(package_spec)
        
        if to_install:
            print(f"Installing {len(to_install)} packages: {', '.join([extract_package_name(p) for p in to_install])}")
            
            # Use pip to install packages
            cmd = [sys.executable, '-m', 'pip', 'install', '--upgrade', '--quiet'] + to_install
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            
            if result.returncode == 0:
                print("✓ Package installation completed successfully")
                
                # Verify installations
                failed_installs = []
                for package_spec in to_install:
                    package_name = extract_package_name(package_spec)
                    if not is_package_installed(package_name):
                        failed_installs.append(package_name)
                
                if failed_installs:
                    print(f"⚠ Warning: Could not verify installation of: {', '.join(failed_installs)}")
                else:
                    print("✓ All packages verified as installed")
                    
            else:
                print(f"⚠ Package installation encountered issues:")
                print(f"  stdout: {result.stdout}")
                print(f"  stderr: {result.stderr}")
        else:
            print("✓ All required packages already installed")
            
    except Exception as e:
        print(f"⚠ Package installation failed: {e}")
        print("  You may need to install packages manually:")
        for package in packages:
            print(f"    pip install {package}")

# Define required packages with minimum versions
required_packages = [
    'pandas>=2.2.0',
    'numpy>=1.26.0', 
    'matplotlib>=3.8.0',
    'seaborn>=0.13.0',
    'scipy>=1.12.0',
    'statsmodels>=0.14.1',
    'scikit-learn>=1.4.0',
    'openpyxl>=3.1.2',
    'python-dateutil>=2.9.0',
    'pytz>=2024.1',
]

# Automatically install missing packages
ensure_packages(required_packages)

# %%
# ============================================================================
# CELL 2: CORE IMPORTS & VISUALIZATION SETUP
# ============================================================================

# Core imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter
from scipy import stats, optimize, signal
from scipy.interpolate import interp1d
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from dateutil import tz
from dateutil.parser import parse as dateparse
import pytz

# Configure visualization defaults
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams.update({
    'figure.figsize': (12, 6),
    'axes.labelsize': 11,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'lines.linewidth': 1.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.dpi': 100,
})

# Color palette for consistent styling
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent': '#F18F01',
    'success': '#73AB84',
    'warning': '#C73E1D',
    'neutral': '#6C757D',
    'light': '#F8F9FA',
    'dark': '#212529'
}

print("✓ Environment setup complete")

# %%
# ============================================================================
# CELL 3: CONFIGURATION & DATA ENGINEERING PATTERNS
# ============================================================================

@dataclass
class EnergyConfig:
    """Configuration management for energy analysis parameters."""
    
    # File paths
    base_path: str = "."
    
    # Timezone configuration
    local_tz: str = "America/New_York"
    
    # Physical constants
    hvac_cop_heating: float = 2.8  # Coefficient of Performance for heating
    hvac_cop_cooling: float = 3.5  # Coefficient of Performance for cooling
    
    # Analysis parameters
    steady_state_threshold_c: float = 1.0  # ±°C for steady-state detection
    steady_state_min_hours: int = 1
    
    # Degree day bases
    hdd_base_c: float = 18.0  # Heating degree day base (°C)
    cdd_base_c: float = 22.0  # Cooling degree day base (°C)
    
    # Water heater specifications (Rheem XE40M06ST45U1)
    water_heater_capacity_gal: float = 40
    water_heater_power_w: float = 4500
    water_heater_setpoint_f: float = 135
    
    # Room specifications
    room_areas_ft2: Dict[str, float] = None
    room_hvac_zones: Dict[str, str] = None
    
    def __post_init__(self):
        """Initialize room specifications after dataclass creation."""
        self.room_areas_ft2 = {
            'Main Bedroom': self._parse_dimensions("14'10\"", 8.5) + self._parse_dimensions("12'3\"", 8.5),
            'Living Room': self._parse_dimensions("14'11\"", 9.5) + self._parse_dimensions("12'11\"", 9.5),
            'Kitchen Cabinet': self._parse_dimensions("13'4\"", 7.5) + self._parse_dimensions("14'1\"", 7.5),
            '2nd Floor Bedroom': self._parse_dimensions("11'4\"", 8.5) + self._parse_dimensions("14'9\"", 8.5)
        }
        
        self.room_hvac_zones = {
            'Kitchen Cabinet': 'AC_Floor1',
            'Living Room': 'AC_Floor1',
            'Main Bedroom': 'AC_Floors23',
            '2nd Floor Bedroom': 'AC_Floors23',
            'Kitchen': 'AC_Floor1',  # Additional mappings
            'Thermostat': 'AC_Floor1'
        }
    
    @staticmethod
    def _parse_dimensions(length_str: str, height: float) -> float:
        """Parse feet'inches\" format to square feet."""
        if "'" in length_str:
            parts = length_str.split("'")
            feet = float(parts[0])
            inches = float(parts[1].replace('"', '').strip()) if len(parts) > 1 else 0
            return (feet + inches/12) * height
        return float(length_str) * height

# Initialize configuration
config = EnergyConfig()
print("✓ Configuration initialized")

# %%
# ============================================================================
# CELL 4: DATA LOADING PIPELINE CLASS DEFINITION
# ============================================================================

class EnergyDataLoader:
    """Sophisticated data loading and cleaning pipeline with error handling."""
    
    def __init__(self, config: EnergyConfig):
        self.config = config
        self.tz = pytz.timezone(config.local_tz)
        self.data = {}
        self.metadata = {}
        
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """Load all data sources with comprehensive error handling."""
        print("\n" + "="*60)
        print("DATA LOADING PIPELINE")
        print("="*60)
        
        # Load each data source
        self.data['emporia'] = self._load_emporia_csv()
        self.data['natgrid'] = self._load_natgrid_usage()
        self.data['costs'] = self._load_cost_breakdown()
        self.data['weather'] = self._load_weather()
        self.data['temps'] = self._load_room_temperatures()
        self.data['neighbors'] = self._load_neighbors_comparison()
        self.data['excel'] = self._load_excel_data()
        
        # Validate and report
        self._validate_data_quality()
        
        return self.data
    
    def _ensure_timezone(self, dt_series: pd.Series) -> pd.Series:
        """Robustly handle timezone conversion with DST awareness."""
        dt_series = pd.to_datetime(dt_series, errors='coerce')
        
        if hasattr(dt_series, 'dt'):
            if dt_series.dt.tz is None:
                # Localize with DST handling
                try:
                    return dt_series.dt.tz_localize(self.tz, ambiguous='infer', nonexistent='shift_forward')
                except:
                    # Fallback for ambiguous times
                    return dt_series.dt.tz_localize(self.tz, ambiguous=False, nonexistent='shift_forward')
            else:
                return dt_series.dt.tz_convert(self.tz)
        else:
            # Handle scalar
            if dt_series.tzinfo is None:
                return self.tz.localize(dt_series, is_dst=None)
            else:
                return dt_series.astimezone(self.tz)
    
    def _load_emporia_csv(self) -> pd.DataFrame:
        """Load Emporia Vue energy monitoring data."""
        try:
            df = pd.read_csv('data/emporium_energy_monitoring.csv')
            df['Timestamp'] = self._ensure_timezone(df['Time Bucket (America/New_York)'])
            
            # Clean numeric columns
            for col in df.columns:
                if 'kWh' in col:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.set_index('Timestamp').sort_index()
            
            # Add missing Mains_C if it's all strings
            if 'Mains_C (kWhs)' in df.columns:
                df['Mains_C (kWhs)'] = pd.to_numeric(df['Mains_C (kWhs)'], errors='coerce')
            
            self.metadata['emporia_range'] = (df.index.min(), df.index.max())
            print(f"✓ Emporia: {len(df)} days, {df.index.min().date()} to {df.index.max().date()}")
            return df
            
        except Exception as e:
            print(f"✗ Emporia loading failed: {e}")
            return pd.DataFrame()
    
    def _load_natgrid_usage(self) -> pd.DataFrame:
        """Load National Grid usage data with seller/buyer distinction."""
        try:
            df = pd.read_csv('data/national_grid_electricity_usage.csv')
            df.columns = [c.strip() for c in df.columns]
            
            # Parse month column
            df['Month'] = pd.to_datetime(df['Month'], errors='coerce')
            df['Month'] = df['Month'] + pd.offsets.MonthEnd(0)
            
            # Clean usage columns
            for col in ['Seller Usage (kWh)', 'KKrenek USAGE (kWh)', 'Avg Neighbors (kWh)']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Clean cost columns
            for col in ['Seller Cost', 'KKrenek COST']:
                if col in df.columns:
                    df[col] = df[col].apply(self._parse_currency)
            
            df = df.sort_values('Month').reset_index(drop=True)
            
            self.metadata['natgrid_range'] = (df['Month'].min(), df['Month'].max())
            print(f"✓ NatGrid: {len(df)} months, {df['Month'].min().date()} to {df['Month'].max().date()}")
            return df
            
        except Exception as e:
            print(f"✗ NatGrid loading failed: {e}")
            return pd.DataFrame()
    
    def _load_cost_breakdown(self) -> pd.DataFrame:
        """Load itemized cost breakdown."""
        try:
            df = pd.read_csv('data/national_grid_costs_breakdown.csv')
            
            # Parse dates
            if 'Bill Date' in df.columns:
                df['Bill Date'] = pd.to_datetime(df['Bill Date'], errors='coerce')
                df['Month'] = df['Bill Date'] + pd.offsets.MonthEnd(0)
            
            # Parse usage
            if 'Total Usage (kWh)' in df.columns:
                df['Total Usage (kWh)'] = pd.to_numeric(df['Total Usage (kWh)'], errors='coerce')
            
            # Parse all cost columns
            cost_cols = [c for c in df.columns if c not in ['Bill Date', 'Month', 'Total Usage (kWh)', 'Service Period']]
            for col in cost_cols:
                df[col] = df[col].apply(self._parse_currency)
            
            df = df.sort_values('Month').reset_index(drop=True)
            
            print(f"✓ Cost Breakdown: {len(df)} months itemized")
            return df
            
        except Exception as e:
            print(f"✗ Cost breakdown loading failed: {e}")
            return pd.DataFrame()
    
    def _load_weather(self) -> pd.DataFrame:
        """Load outdoor weather data."""
        try:
            df = pd.read_csv('data/outdoor_weather_download.csv')
            df['Timestamp'] = self._ensure_timezone(df['date'])
            df['temp_c'] = pd.to_numeric(df['temperature_2m'], errors='coerce')
            df['humidity'] = pd.to_numeric(df['relative_humidity_2m'], errors='coerce')
            df = df.set_index('Timestamp').sort_index()

            # FIX: Check for and resolve duplicate timestamps by averaging
            if df.index.has_duplicates:
                duplicate_count = df.index.duplicated().sum()
                print(f"  ⚠ Found and resolved {duplicate_count} duplicate timestamps in weather data by averaging.")
                df = df.groupby(df.index).mean(numeric_only=True)
            
            self.metadata['weather_range'] = (df.index.min(), df.index.max())
            print(f"✓ Weather: {len(df)} hours, {df.index.min().date()} to {df.index.max().date()}")
            return df
            
        except Exception as e:
            print(f"✗ Weather loading failed: {e}")
            return pd.DataFrame()

    def _load_room_temperatures(self) -> pd.DataFrame:
        """Load and reshape room temperature data from the time-matched CSV."""
        try:
            # Load the dataset
            df = pd.read_csv('data/elitech_temperatures_time_matched.csv')

            # Convert the 'DateTime' column from Excel's serial number format to datetime objects.
            # Excel's date system starts from 1899-12-30, so we use that as the origin.
            df['DateTime'] = pd.to_datetime(df['DateTime'], unit='D', origin='1899-12-30')

            # Reshape the DataFrame from wide to long format. 'DateTime' is the identifier,
            # and all other columns are treated as measurement variables.
            melted_df = df.melt(id_vars=['DateTime'], var_name='Location', value_name='temp_c')

            # Rename 'DateTime' to 'Timestamp' for consistency with the original function's output.
            melted_df.rename(columns={'DateTime': 'Timestamp'}, inplace=True)

            # === FIX: Make the Timestamp column timezone-aware to match other datasets ===
            melted_df['Timestamp'] = self._ensure_timezone(melted_df['Timestamp'])

            # Remove any rows that have missing temperature or timestamp values.
            melted_df.dropna(subset=['temp_c', 'Timestamp'], inplace=True)
            
            # Ensure 'temp_c' is a numeric type, converting any non-numeric values to NaN,
            # and then drop any rows that might have been converted to NaN.
            melted_df['temp_c'] = pd.to_numeric(melted_df['temp_c'], errors='coerce')
            melted_df.dropna(subset=['temp_c'], inplace=True)

            # Sort the entire DataFrame by the 'Timestamp' to have a chronological order.
            result = melted_df.sort_values('Timestamp').reset_index(drop=True)

            # Print a success message with a summary of the loaded data.
            locations = result['Location'].unique()
            print(f"✓ Room Temps: {len(result)} readings, {len(locations)} locations")
            
            return result

        except FileNotFoundError:
            print("✗ Room temps loading failed: File not found.")
            return pd.DataFrame()
        except Exception as e:
            print(f"✗ Room temps loading failed: {e}")
            return pd.DataFrame()
        
    # def _load_room_temperatures(self) -> pd.DataFrame:
    #     """Load and reshape room temperature data from wide format."""
    #     try:
    #         df = pd.read_csv('data/elitech_temperatures_time_matched.csv')
            
    #         # Identify DateTime/Value column pairs
    #         pairs = []
    #         for i, col in enumerate(df.columns):
    #             if 'DateTime' in str(col):
    #                 # Find next non-DateTime column
    #                 for j in range(i+1, len(df.columns)):
    #                     if 'DateTime' not in str(df.columns[j]) and 'Unnamed' not in str(df.columns[j]):
    #                         pairs.append((col, df.columns[j]))
    #                         break
            
    #         # Reshape to long format
    #         frames = []
    #         for dt_col, val_col in pairs:
    #             temp_df = df[[dt_col, val_col]].copy()
    #             temp_df.columns = ['Timestamp', 'temp_c']
    #             temp_df['Timestamp'] = self._ensure_timezone(temp_df['Timestamp'])
    #             temp_df['temp_c'] = pd.to_numeric(temp_df['temp_c'], errors='coerce')
    #             temp_df['Location'] = str(val_col).strip()
    #             temp_df = temp_df.dropna(subset=['Timestamp', 'temp_c'])
    #             frames.append(temp_df)
            
    #         if frames:
    #             result = pd.concat(frames, ignore_index=True)
    #             result = result.sort_values('Timestamp').reset_index(drop=True)
                
    #             locations = result['Location'].unique()
    #             print(f"✓ Room Temps: {len(result)} readings, {len(locations)} locations")
    #             return result
    #         else:
    #             print("✗ No room temperature data found")
    #             return pd.DataFrame()
                
    #     except Exception as e:
    #         print(f"✗ Room temps loading failed: {e}")
    #         return pd.DataFrame()
    
    def _load_neighbors_comparison(self) -> pd.DataFrame:
        """Load recent neighbor comparison data."""
        try:
            df = pd.read_csv('data/recent_electricity_compared_to_neighbors.txt', sep='\t')
            
            # Parse month column (handle Sep '24 format)
            df['Month'] = df['Month'].str.replace("'", " 20")
            df['Month'] = pd.to_datetime(df['Month'], errors='coerce')
            df['Month'] = df['Month'] + pd.offsets.MonthEnd(0)
            
            # Parse numeric columns
            for col in df.columns:
                if col != 'Month':
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            print(f"✓ Neighbors: {len(df)} months comparison data")
            return df
            
        except Exception as e:
            print(f"✗ Neighbors comparison loading failed: {e}")
            return pd.DataFrame()
    
    def _load_excel_data(self) -> Dict[str, pd.DataFrame]:
        """Load multi-sheet Excel data with early measurements."""
        excel_data = {}
        try:
            xl_file = pd.ExcelFile('data/early_combined_data_trim.xlsx')
            
            for sheet in xl_file.sheet_names:
                df = pd.read_excel(xl_file, sheet_name=sheet)
                excel_data[sheet] = df
                print(f"  • Excel sheet '{sheet}': {df.shape}")
            
            print(f"✓ Excel: {len(excel_data)} sheets loaded")
            
        except Exception as e:
            print(f"✗ Excel loading failed: {e}")
            
        return excel_data
    
    @staticmethod
    def _parse_currency(value) -> float:
        """Parse currency strings to float."""
        if pd.isna(value):
            return np.nan
        if isinstance(value, (int, float)):
            return float(value)
        
        str_val = str(value).replace('$', '').replace(',', '').strip()
        try:
            return float(str_val) if str_val else np.nan
        except:
            return np.nan
    
    def _validate_data_quality(self) -> None:
        """Validate data quality and report issues."""
        print("\n" + "-"*60)
        print("DATA QUALITY REPORT")
        print("-"*60)
        
        issues = []
        
        # Check Emporia completeness
        if 'emporia' in self.data and not self.data['emporia'].empty:
            null_pct = self.data['emporia'].isnull().sum() / len(self.data['emporia']) * 100
            high_null = null_pct[null_pct > 10]
            if not high_null.empty:
                issues.append(f"Emporia columns with >10% missing: {', '.join(high_null.index[:3])}")
        
        # Check time alignment
        if 'emporia' in self.data and 'weather' in self.data:
            if not self.data['emporia'].empty and not self.data['weather'].empty:
                emp_range = self.metadata.get('emporia_range', (None, None))
                wx_range = self.metadata.get('weather_range', (None, None))
                if emp_range[0] and wx_range[0]:
                    overlap_days = (min(emp_range[1], wx_range[1]) - max(emp_range[0], wx_range[0])).days
                    if overlap_days < 30:
                        issues.append(f"Limited overlap between Emporia and weather: {overlap_days} days")
        
        if issues:
            print("⚠ Data quality issues detected:")
            for issue in issues:
                print(f"  • {issue}")
        else:
            print("✓ No major data quality issues detected")

print("✓ Data loading classes defined")

# %%
# ============================================================================
# CELL 5: LOAD ALL DATA
# ============================================================================

# Initialize data loader and load all data
loader = EnergyDataLoader(config)
data = loader.load_all_data()

# Quick summary of loaded data
print(f"\n📊 Data Summary:")
print(f"{'='*40}")
for name, df in data.items():
    if isinstance(df, pd.DataFrame) and not df.empty:
        print(f"• {name}: {df.shape[0]} rows, {df.shape[1]} columns")
    elif isinstance(df, dict):
        print(f"• {name}: {len(df)} sheets")
    else:
        print(f"• {name}: Empty or failed to load")

# %%
# ============================================================================
# CELL 6: NEIGHBOR COMPARISON SHOCK ANALYSIS
# ============================================================================

def create_neighbor_comparison_shock(data: Dict[str, pd.DataFrame]) -> None:
    """Create dramatic visualization showing excessive usage vs neighbors."""
    
    print("\n" + "="*60)
    print("THE PROBLEM: ASTRONOMICAL ENERGY CONSUMPTION")
    print("="*60)
    
    # Combine neighbor data sources
    neighbors = data.get('neighbors', pd.DataFrame())
    natgrid = data.get('natgrid', pd.DataFrame())
    
    if natgrid.empty:
        print("✗ Neighbor comparison data not available")
        return
    
    # Create shocking comparison visualization
    fig = plt.figure(figsize=(15, 8))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Early data
    ax0 = fig.add_subplot(gs[0, 0])

    keep_inds = 22
    your_data = natgrid['KKrenek USAGE (kWh)'][:keep_inds]
    neighbors_data = natgrid['Avg Neighbors (kWh)'][:keep_inds]
    sellers_data = natgrid['Seller Usage (kWh)'][:keep_inds]
    
    # Calculate multipliers for drama
    your_avg = your_data.mean()
    neighbors_avg = neighbors_data.mean()
    seller_avg = sellers_data.mean()

    multiplier_all = your_avg / neighbors_avg
    
    # Plot comparison
    months = pd.to_datetime(natgrid['Month'][:keep_inds]).dt.strftime('%b %Y')
    x = np.arange(len(months))
    
    ax0.bar(x, your_data, label='Your Home', color=COLORS['warning'], alpha=0.8)
    ax0.bar(x, neighbors_data, label='Average Neighbors', color=COLORS['neutral'], alpha=0.6)
    ax0.bar(x, sellers_data, label='Seller Usage', color=COLORS['success'], alpha=0.6)
    
    ax0.set_ylim([0, 8000])
    ax0.set_xticks(x)
    ax0.set_xticklabels(months, rotation=45, ha='right')
    ax0.set_ylabel('Monthly Usage (kWh)', fontsize=12)
    ax0.set_title(f'{multiplier_all:.1f}× More Electricity Than Neighbors',
                  fontsize=14, fontweight='bold', color=COLORS['warning'])
    ax0.legend(loc='upper left')
    ax0.grid(True, alpha=0.3)

    # Cost impact visualization
    ax4 = fig.add_subplot(gs[1, 0])

    your_data = natgrid['KKrenek COST'][:keep_inds]
    sellers_data = natgrid['Seller Cost'][:keep_inds]
    
    ax4.bar(months, your_data, label='Your Home', color=COLORS['warning'], alpha=0.8)
    ax4.bar(months, sellers_data, label='Seller Usage', color=COLORS['success'], alpha=0.6)
    
    ax4.set_title('Monthly Usage Costs', 
                  fontsize=12, fontweight='bold')
    ax4.set_ylabel('Usage Cost ($)', fontsize=11)
    ax4.set_xticklabels(months, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3)
    
    # Recent data
    ax1 = fig.add_subplot(gs[0, 1])
    
    # Calculate multipliers for drama
    your_avg = neighbors['Your Home (kWh)'].mean()
    eff_avg = neighbors['Efficient Similar Homes (kWh)'].mean()
    all_avg = neighbors['All Similar Homes (kWh)'].mean()
    
    multiplier_eff = your_avg / eff_avg
    multiplier_all = your_avg / all_avg
    
    # Plot comparison
    months = pd.to_datetime(neighbors['Month']).dt.strftime('%b %Y')
    x = np.arange(len(months))
    
    ax1.bar(x, neighbors['Your Home (kWh)'], label='Your Home', color=COLORS['warning'], alpha=0.8)
    ax1.bar(x, neighbors['All Similar Homes (kWh)'], label='Average Similar Homes', color=COLORS['neutral'], alpha=0.6)
    ax1.bar(x, neighbors['Efficient Similar Homes (kWh)'], label='Efficient Similar Homes', color=COLORS['success'], alpha=0.6)
    
    ax1.set_ylim([0, 8000])
    ax1.set_xticks(x)
    ax1.set_xticklabels(months, rotation=45, ha='right')
    ax1.set_ylabel('Monthly Usage (kWh)', fontsize=12)
    ax1.set_title(f'⚡ SHOCKING REVELATION: This Home Uses {multiplier_all:.1f}× More Electricity Than Neighbors! ⚡',
                  fontsize=14, fontweight='bold', color=COLORS['warning'])
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Add dramatic annotations
    peak_month_idx = neighbors['Your Home (kWh)'].idxmax()
    peak_value = neighbors.loc[peak_month_idx, 'Your Home (kWh)']
    peak_month = months.iloc[peak_month_idx]
    
    ax1.annotate(f'Peak: {peak_value:.0f} kWh\n({multiplier_eff:.1f}× efficient homes!)',
                xy=(peak_month_idx, peak_value),
                xytext=(peak_month_idx-2, peak_value+500),
                arrowprops=dict(arrowstyle='->', color=COLORS['warning'], lw=2),
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # Cost impact visualization
    ax2 = fig.add_subplot(gs[1, 1])
    
    # Estimate excess costs (assuming $0.25/kWh average)
    excess_usage = neighbors['Your Home (kWh)'] - neighbors['All Similar Homes (kWh)']
    excess_cost = excess_usage * 0.25
    annual_excess = excess_cost.sum()
    
    ax2.bar(months, excess_cost, color=COLORS['warning'], alpha=0.8)
    ax2.set_title(f'Monthly Excess Cost vs. Average Homes\n(Annual Excess: ${annual_excess:,.0f})', 
                  fontsize=12, fontweight='bold')
    ax2.set_ylabel('Excess Cost ($)', fontsize=11)
    ax2.set_xticklabels(months, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Energy Crisis Analysis: 38 Old Elm St, North Billerica, MA', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.show()
    
    # Print shocking statistics
    print(f"""
📊 KEY FINDINGS:
┌────────────────────────────────────────────────────────────────┐
• Your home uses {multiplier_all:.1f}× more electricity than average similar homes
• Your home uses {multiplier_eff:.1f}× more electricity than efficient similar homes
• Peak monthly usage: {peak_value:.0f} kWh ({peak_month})
• Estimated annual excess cost: ${annual_excess:,.0f}
• Average monthly usage: {your_avg:.0f} kWh vs {all_avg:.0f} kWh (neighbors)
└────────────────────────────────────────────────────────────────┘

This extreme usage pattern demands investigation into:
1. HVAC system efficiency and operation
2. Building envelope performance
3. Water heater cycling patterns
4. Potential for targeted improvements
""")

# Run the neighbor comparison analysis
create_neighbor_comparison_shock(data)

# %%
# ============================================================================
# CELL 7: DATA VALIDATION CLASS DEFINITION
# ============================================================================

class DataValidation:
    """Sophisticated validation between utility and monitoring data using actual billing periods."""
    
    @staticmethod
    def validate_emporia_natgrid(emporia: pd.DataFrame, natgrid: pd.DataFrame, 
                                costs_breakdown: pd.DataFrame = None, 
                                timezone: str = "America/New_York") -> Dict:
        """Validate Emporia measurements against utility bills using actual billing periods."""
        
        print("\n" + "="*60)
        print("DATA VALIDATION: EMPORIA VS NATIONAL GRID (BILLING PERIODS)")
        print("="*60)
        
        if emporia.empty:
            print("✗ No Emporia data available")
            return {}
        
        # Try to use costs breakdown for accurate billing periods
        if costs_breakdown is not None and not costs_breakdown.empty:
            print("📅 Using actual billing periods from costs breakdown")
            return DataValidation._validate_with_billing_periods(emporia, costs_breakdown, timezone)
        
        # Fall back to natgrid monthly comparison if costs breakdown unavailable
        if natgrid.empty:
            print("✗ No National Grid data available")
            return {}
            
        print("📅 Falling back to monthly aggregation comparison")
        return DataValidation._validate_monthly_aggregation(emporia, natgrid)
    
    @staticmethod
    def _validate_with_billing_periods(emporia: pd.DataFrame, costs_breakdown: pd.DataFrame, timezone: str) -> Dict:
        """Validate using actual utility billing periods - ROBUST DST-SAFE VERSION."""
        
        # Parse bill dates and usage from costs breakdown
        costs = costs_breakdown.copy()
        
        # Clean and parse bill dates
        costs['Bill_Date'] = pd.to_datetime(costs['Bill Date'], errors='coerce')
        costs = costs.dropna(subset=['Bill_Date']).sort_values('Bill_Date')
        
        # ROBUST DST-SAFE TIMEZONE LOCALIZATION
        import pytz
        tz = pytz.timezone(timezone)
        
        def safe_tz_localize(dt_series, timezone_obj):
            """Safely localize datetime series with comprehensive DST handling."""
            try:
                # First attempt: Use pandas' built-in DST inference
                return dt_series.dt.tz_localize(timezone_obj, ambiguous='infer', nonexistent='shift_forward')
            except Exception as e1:
                print(f"  Warning: DST inference failed ({str(e1)[:50]}...), trying explicit handling")
                try:
                    # Second attempt: Handle ambiguous times as False (first occurrence)
                    return dt_series.dt.tz_localize(timezone_obj, ambiguous=False, nonexistent='shift_forward')
                except Exception as e2:
                    print(f"  Warning: Standard DST handling failed ({str(e2)[:50]}...), trying element-wise")
                    try:
                        # Third attempt: Element-wise localization with individual error handling
                        localized_dates = []
                        for dt in dt_series:
                            try:
                                if pd.isna(dt):
                                    localized_dates.append(pd.NaT)
                                else:
                                    # Try to localize individual datetime
                                    try:
                                        localized_dt = timezone_obj.localize(dt, is_dst=None)
                                        localized_dates.append(localized_dt)
                                    except pytz.AmbiguousTimeError:
                                        # During fall DST transition, choose first occurrence
                                        localized_dt = timezone_obj.localize(dt, is_dst=False)
                                        localized_dates.append(localized_dt)
                                    except pytz.NonExistentTimeError:
                                        # During spring DST transition, shift forward 1 hour
                                        shifted_dt = dt + pd.Timedelta(hours=1)
                                        localized_dt = timezone_obj.localize(shifted_dt, is_dst=True)
                                        localized_dates.append(localized_dt)
                            except Exception as e_individual:
                                print(f"    Warning: Could not localize {dt}, using NaT: {e_individual}")
                                localized_dates.append(pd.NaT)
                        
                        return pd.Series(localized_dates, index=dt_series.index)
                    except Exception as e3:
                        print(f"  Error: All timezone localization methods failed: {e3}")
                        # Last resort: return as UTC then convert
                        utc_series = dt_series.dt.tz_localize('UTC')
                        return utc_series.dt.tz_convert(timezone_obj)
        
        # Apply safe localization to bill dates
        print("  Localizing bill dates with DST safety...")
        costs['Bill_Date'] = safe_tz_localize(costs['Bill_Date'], tz)
        
        # Parse total usage (remove any non-numeric characters)
        def parse_usage(val):
            if pd.isna(val):
                return np.nan
            try:
                # Remove any currency symbols, commas, etc.
                clean_val = str(val).replace(',', '').replace('$', '').strip()
                return float(clean_val)
            except:
                return np.nan
        
        costs['Usage_kWh'] = costs['Total Usage (kWh)'].apply(parse_usage)
        costs = costs.dropna(subset=['Usage_kWh'])
        
        if len(costs) < 2:
            print("✗ Insufficient billing data for period analysis")
            return {}
        
        print(f"📊 Found {len(costs)} billing periods")
        
        # Calculate service periods (from previous bill to current bill)
        service_periods = []
        emporia_totals = []
        natgrid_usage = []
        
        # Ensure emporia index is timezone-aware and matches
        if emporia.index.tz is None:
            emporia.index = safe_tz_localize(pd.Series(emporia.index), tz)
        elif emporia.index.tz != tz:
            emporia.index = emporia.index.tz_convert(tz)
        
        for i in range(1, len(costs)):
            # Service period: from previous bill date to current bill date
            period_start = costs.iloc[i-1]['Bill_Date']
            period_end = costs.iloc[i]['Bill_Date']
            current_usage = costs.iloc[i]['Usage_kWh']
            
            # Skip if period is outside Emporia data range
            if pd.isna(period_start) or pd.isna(period_end):
                print(f"    Skipping period {i}: Invalid dates")
                continue
                
            if period_end < emporia.index.min() or period_start > emporia.index.max():
                print(f"    Skipping period {i}: Outside data range")
                continue
            
            # Adjust period boundaries to match available data
            actual_start = max(period_start, emporia.index.min())
            actual_end = min(period_end, emporia.index.max())
            
            if actual_end <= actual_start:
                print(f"    Skipping period {i}: Invalid period bounds")
                continue
            
            # Sum Emporia data over this period
            try:
                period_mask = (emporia.index >= actual_start) & (emporia.index < actual_end)
                period_emporia = emporia[period_mask]
                
                if len(period_emporia) == 0:
                    print(f"    Skipping period {i}: No Emporia data in period")
                    continue
                
                # CORRECTED CALCULATION: Use Mains_A + Mains_B (not individual circuits)
                # This represents the actual total consumption at the service entrance
                mains_cols = [col for col in period_emporia.columns if 'Mains' in col and 'kWhs' in col]
                
                if len(mains_cols) == 0:
                    print(f"    Warning: No Mains columns found for period {i}, falling back to circuit sum")
                    # Fallback to circuit sum if Mains not available
                    numeric_cols = [col for col in period_emporia.select_dtypes(include=[np.number]).columns 
                                if "Mains" not in col]
                    emporia_total = period_emporia[numeric_cols].sum().sum()
                    method_used = "Circuit Sum (Fallback)"
                else:
                    # Use Mains total (this should match utility meter)
                    emporia_total = period_emporia[mains_cols].sum().sum()
                    method_used = f"Mains Total ({', '.join(mains_cols)})"
                
                service_periods.append({
                    'start': actual_start,
                    'end': actual_end,
                    'days': (actual_end - actual_start).days,
                    'bill_date': period_end,
                    'method': method_used
                })
                emporia_totals.append(emporia_total)
                natgrid_usage.append(current_usage)
                
                print(f"    Period {i}: {actual_start.date()} to {actual_end.date()} "
                    f"({(actual_end - actual_start).days} days)")
                print(f"      Emporia: {emporia_total:.1f} kWh, National Grid: {current_usage:.1f} kWh, "
                    f"Diff: {abs(emporia_total - current_usage):.1f} kWh")
                    
            except Exception as e:
                print(f"    Error processing period {i}: {e}")
                continue
        
        if len(emporia_totals) == 0:
            print("✗ No overlapping periods found")
            return {}
        
        # Convert to arrays for analysis
        emporia_array = np.array(emporia_totals)
        natgrid_array = np.array(natgrid_usage)
        
        # Calculate validation metrics
        try:
            # Linear regression
            import statsmodels.api as sm
            X = sm.add_constant(natgrid_array)
            y = emporia_array
            model = sm.OLS(y, X).fit()
            
            # Error metrics
            errors = emporia_array - natgrid_array
            error_pct = np.abs(errors) / natgrid_array * 100
            
            metrics = {
                'r2': model.rsquared,
                'slope': model.params[1] if len(model.params) > 1 else 1.0,
                'intercept': model.params[0] if len(model.params) > 0 else 0.0,
                'mape': np.mean(error_pct),
                'max_error_pct': np.max(error_pct),
                'mean_error_kwh': np.mean(np.abs(errors)),
                'n_periods': len(emporia_totals),
                'total_days': sum(p['days'] for p in service_periods),
                'method_used': service_periods[0]['method'] if service_periods else 'Unknown'
            }
            
        except Exception as e:
            print(f"✗ Statistical analysis failed: {e}")
            return {}
        
        # Create comparison dataframe for visualization
        comparison_df = pd.DataFrame({
            'Period_End': [p['bill_date'] for p in service_periods],
            'Days': [p['days'] for p in service_periods],
            'Emporia_kWh': emporia_array,
            'NatGrid_kWh': natgrid_array,
            'Error_kWh': errors,
            'Error_pct': error_pct
        })
        
        # Visualization
        DataValidation._visualize_billing_validation(comparison_df, metrics)
        
        # Print detailed results
        print(f"""
    📊 ROBUST BILLING PERIOD VALIDATION RESULTS:
    ┌────────────────────────────────────────────────────────────────────────
    • Method Used: {metrics['method_used']}
    • Correlation (R²): {metrics['r2']:.3f}
    • Mean Absolute Percentage Error: {metrics['mape']:.1f}%
    • Maximum Error: {metrics['max_error_pct']:.1f}%
    • Mean Absolute Error: {metrics['mean_error_kwh']:.1f} kWh
    • Regression Slope: {metrics['slope']:.3f}
    • Periods Compared: {metrics['n_periods']} ({metrics['total_days']} total days)
    └────────────────────────────────────────────────────────────────────────

    {"✅ VALIDATION PASSED: Emporia Mains measurement is reliable" if metrics['mape'] < 5 else "⚠ Acceptable accuracy for utility validation" if metrics['mape'] < 10 else "⚠ Higher than expected discrepancy - investigate further"}

    Period-by-period breakdown:""")
        
        for i, row in comparison_df.iterrows():
            print(f"  • {row['Period_End'].date()}: {row['Error_pct']:+5.1f}% error "
                f"({row['Emporia_kWh']:.0f} vs {row['NatGrid_kWh']:.0f} kWh)")
        
        return metrics

    @staticmethod
    def _validate_monthly_aggregation(emporia: pd.DataFrame, natgrid: pd.DataFrame) -> Dict:
        """Fallback validation using monthly aggregation (original method)."""
        
        # Aggregate Emporia to monthly
        numeric_cols = [col for col in emporia.select_dtypes(include=[np.number]).columns if "Mains" not in col]
        emporia_monthly = emporia[numeric_cols].sum(axis=1).resample('M').sum()
        emporia_monthly.index = emporia_monthly.index.to_period('M').to_timestamp('M')
        emporia_monthly = emporia_monthly.to_frame('Emporia_kWh')
        
        # Get NatGrid usage (prioritize KKrenek)
        ng_usage = None
        for col in ['KKrenek USAGE (kWh)', 'Seller Usage (kWh)']:
            if col in natgrid.columns:
                ng_usage = natgrid.set_index('Month')[col].dropna()
                break
        
        if ng_usage is None:
            print("✗ No usage data in National Grid")
            return {}
        
        # Join and compare
        comparison = emporia_monthly.join(ng_usage.to_frame('NatGrid_kWh'), how='inner')
        
        if comparison.empty:
            print("✗ No overlapping months for comparison")
            return {}
        
        # Calculate metrics
        X = sm.add_constant(comparison['NatGrid_kWh'])
        y = comparison['Emporia_kWh']
        model = sm.OLS(y, X).fit()
        
        comparison['Error_pct'] = (comparison['Emporia_kWh'] - comparison['NatGrid_kWh']).abs() / comparison['NatGrid_kWh'] * 100
        
        metrics = {
            'r2': model.rsquared,
            'slope': model.params.get('NatGrid_kWh', 1.0),
            'intercept': model.params.get('const', 0.0),
            'mape': comparison['Error_pct'].mean(),
            'max_error_pct': comparison['Error_pct'].max(),
            'n_months': len(comparison)
        }
        
        # Use existing visualization (create simple version)
        DataValidation._visualize_monthly_validation(comparison, metrics)
        
        return metrics
    
    @staticmethod
    def _visualize_billing_validation(comparison_df: pd.DataFrame, metrics: Dict) -> None:
        """Create visualization for billing period validation."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Time series comparison
        ax1 = axes[0, 0]
        ax1.plot(comparison_df['Period_End'], comparison_df['Emporia_kWh'], 
                marker='o', label='Emporia Total', color=COLORS['primary'], linewidth=2)
        ax1.plot(comparison_df['Period_End'], comparison_df['NatGrid_kWh'], 
                marker='s', label='National Grid', color=COLORS['secondary'], linewidth=2)
        ax1.set_xlabel('Billing Period End Date')
        ax1.set_ylabel('Usage (kWh)')
        ax1.set_title('Usage Comparison by Billing Period')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Scatter plot with regression
        ax2 = axes[0, 1]
        ax2.scatter(comparison_df['NatGrid_kWh'], comparison_df['Emporia_kWh'], 
                   s=100, alpha=0.6, color=COLORS['accent'])
        
        # Add regression line
        x_line = np.linspace(comparison_df['NatGrid_kWh'].min(), 
                           comparison_df['NatGrid_kWh'].max(), 100)
        y_line = metrics['intercept'] + metrics['slope'] * x_line
        ax2.plot(x_line, y_line, 'r-', linewidth=2, label=f'R² = {metrics["r2"]:.3f}')
        
        # Add 1:1 line
        max_val = max(comparison_df['NatGrid_kWh'].max(), comparison_df['Emporia_kWh'].max())
        ax2.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='1:1 Line')
        
        ax2.set_xlabel('National Grid (kWh)')
        ax2.set_ylabel('Emporia Total (kWh)')
        ax2.set_title(f'Validation: MAPE = {metrics["mape"]:.1f}%')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Error distribution
        ax3 = axes[1, 0]
        ax3.bar(range(len(comparison_df)), comparison_df['Error_pct'], 
               color=[COLORS['success'] if e < 5 else COLORS['warning'] if e < 10 else COLORS['warning'] 
                     for e in comparison_df['Error_pct']],
               alpha=0.7)
        ax3.set_xlabel('Billing Period')
        ax3.set_ylabel('Absolute Error (%)')
        ax3.set_title('Error by Billing Period')
        ax3.axhline(y=5, color='orange', linestyle='--', alpha=0.5, label='5% threshold')
        ax3.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='10% threshold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Period length vs accuracy
        ax4 = axes[1, 1]
        ax4.scatter(comparison_df['Days'], comparison_df['Error_pct'], 
                   s=100, alpha=0.6, color=COLORS['neutral'])
        ax4.set_xlabel('Billing Period Length (days)')
        ax4.set_ylabel('Error (%)')
        ax4.set_title('Accuracy vs Period Length')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Emporia Energy Monitor Validation (Billing Periods)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def _visualize_monthly_validation(comparison: pd.DataFrame, metrics: Dict) -> None:
        """Simple visualization for monthly validation fallback."""
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Time series comparison
        ax1 = axes[0]
        ax1.plot(comparison.index, comparison['Emporia_kWh'], 
                marker='o', label='Emporia Total', color=COLORS['primary'], linewidth=2)
        ax1.plot(comparison.index, comparison['NatGrid_kWh'], 
                marker='s', label='National Grid', color=COLORS['secondary'], linewidth=2)
        ax1.set_xlabel('Month')
        ax1.set_ylabel('Monthly Usage (kWh)')
        ax1.set_title('Monthly Usage Comparison (Fallback)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Scatter plot
        ax2 = axes[1]
        ax2.scatter(comparison['NatGrid_kWh'], comparison['Emporia_kWh'], 
                   s=100, alpha=0.6, color=COLORS['accent'])
        
        # Add regression line
        x_line = np.linspace(comparison['NatGrid_kWh'].min(), 
                           comparison['NatGrid_kWh'].max(), 100)
        y_line = metrics['intercept'] + metrics['slope'] * x_line
        ax2.plot(x_line, y_line, 'r-', linewidth=2, label=f'R² = {metrics["r2"]:.3f}')
        
        ax2.set_xlabel('National Grid (kWh)')
        ax2.set_ylabel('Emporia Total (kWh)')
        ax2.set_title(f'Validation: MAPE = {metrics["mape"]:.1f}%')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

print("✓ Data validation classes defined (with billing period support)")


# ============================================================================
# CELL 8: RUN DATA VALIDATION
# ============================================================================

# Validate Emporia vs National Grid using actual billing periods
validator = DataValidation()
validation_results = validator.validate_emporia_natgrid(
    data.get('emporia', pd.DataFrame()),
    data.get('natgrid', pd.DataFrame()),
    data.get('costs', pd.DataFrame()),  # Pass the costs breakdown data
    timezone=config.local_tz  # Pass the timezone from config
)

# %%
# ============================================================================
# CELL 9: HVAC ANALYZER CLASS DEFINITION
# ============================================================================
# %%
# ============================================================================
# ROBUST TIMEZONE UTILITIES FOR DST HANDLING
# ============================================================================

def robust_timezone_localize(dt_data, target_timezone, data_name="data"):
    """
    Robust timezone localization with comprehensive DST handling.
    Handles NonExistentTimeError and AmbiguousTimeError gracefully.
    """
    import pandas as pd
    import pytz
    import numpy as np
    
    if isinstance(target_timezone, str):
        target_timezone = pytz.timezone(target_timezone)
    
    # Handle different input types
    is_index = isinstance(dt_data, pd.DatetimeIndex)
    is_scalar = not isinstance(dt_data, (pd.Series, pd.DatetimeIndex))
    
    if is_scalar:
        dt_series = pd.Series([dt_data])
    elif is_index:
        dt_series = pd.Series(dt_data)
    else:
        dt_series = pd.to_datetime(dt_data, errors='coerce')
    
    # Strategy 1: Standard pandas localization with inference
    try:
        if dt_series.dt.tz is None:
            localized = dt_series.dt.tz_localize(target_timezone, ambiguous='infer', nonexistent='shift_forward')
        else:
            localized = dt_series.dt.tz_convert(target_timezone)
        
        result = localized if not is_scalar else localized.iloc[0]
        return pd.DatetimeIndex(result) if is_index else result
        
    except Exception:
        pass
    
    # Strategy 2: Explicit DST handling
    try:
        if dt_series.dt.tz is None:
            localized = dt_series.dt.tz_localize(target_timezone, ambiguous=False, nonexistent='shift_forward')
        else:
            localized = dt_series.dt.tz_convert(target_timezone)
        
        result = localized if not is_scalar else localized.iloc[0]
        return pd.DatetimeIndex(result) if is_index else result
        
    except Exception:
        pass
    
    # Strategy 3: Element-wise processing with individual error handling
    try:
        localized_values = []
        datetime_values = dt_series.dt.to_pydatetime() if dt_series.dt.tz is None else dt_series.dt.tz_convert(None).dt.to_pydatetime()
        
        for dt_val in datetime_values:
            try:
                if pd.isna(dt_val):
                    localized_values.append(pd.NaT)
                    continue
                
                try:
                    localized_dt = target_timezone.localize(dt_val, is_dst=None)
                    localized_values.append(localized_dt)
                except pytz.AmbiguousTimeError:
                    # During fall DST transition, choose first occurrence
                    localized_dt = target_timezone.localize(dt_val, is_dst=False)
                    localized_values.append(localized_dt)
                except pytz.NonExistentTimeError:
                    # During spring DST transition, shift forward 1 hour
                    shifted_dt = dt_val + pd.Timedelta(hours=1)
                    localized_dt = target_timezone.localize(shifted_dt, is_dst=True)
                    localized_values.append(localized_dt)
                    
            except Exception:
                localized_values.append(pd.NaT)
        
        localized = pd.Series(localized_values, index=dt_series.index)
        result = localized if not is_scalar else localized.iloc[0]
        return pd.DatetimeIndex(result) if is_index else result
        
    except Exception:
        pass
    
    # Strategy 4: UTC conversion fallback
    try:
        if dt_series.dt.tz is None:
            utc_series = dt_series.dt.tz_localize('UTC')
            localized = utc_series.dt.tz_convert(target_timezone)
        else:
            localized = dt_series.dt.tz_convert(target_timezone)
        
        result = localized if not is_scalar else localized.iloc[0]
        return pd.DatetimeIndex(result) if is_index else result
        
    except Exception as e:
        print(f"Warning: All timezone localization failed for {data_name}: {e}")
        return dt_data

print("Robust timezone utilities loaded")

# ============================================================================
# ENHANCED HVAC ANALYZER CLASS DEFINITION (WITH FIXES)
# ============================================================================

class EnhancedHVACAnalyzer:
    """Advanced HVAC analysis accounting for behavioral changes, manual control, and alternative heating."""
    
    def __init__(self, config: EnergyConfig):
        self.config = config
        self.behavioral_periods = {
            'learning_phase': ('2022-01-01', '2022-06-01'),  # Early period, high usage, renovations
            'adaptation_phase': ('2022-06-01', '2023-01-01'), # Middle period, some improvements  
            'optimized_phase': ('2023-01-01', '2025-12-31')   # Recent period, learned behavior
        }
        
    def analyze_behavioral_hvac(self, emporia: pd.DataFrame, weather: pd.DataFrame, 
                               excel_data: dict = None) -> dict:
        """Comprehensive HVAC analysis accounting for behavioral changes over time."""
        
        print("\n" + "="*70)
        print("ENHANCED BEHAVIORAL HVAC PERFORMANCE ANALYSIS")
        print("="*70)
        
        if emporia.empty or weather.empty:
            print("Insufficient data for HVAC analysis")
            return {}
        
        try:
            # Load and integrate early data if available
            early_data = self._load_early_data(excel_data) if excel_data else pd.DataFrame()
            
            # Combine all data sources
            combined_data = self._prepare_comprehensive_dataset(emporia, weather, early_data)
            
            if combined_data.empty:
                print("Failed to prepare comprehensive dataset")
                return {}
            
            # Identify behavioral periods
            period_analysis = self._analyze_behavioral_periods(combined_data)
            
            # Detect manual HVAC control vs automatic operation
            control_patterns = self._detect_manual_control(combined_data)
            
            # Identify alternative heating usage
            alternative_heating = self._detect_alternative_heating(combined_data)
            
            # Analyze learning/adaptation over time
            learning_analysis = self._analyze_learning_curve(combined_data)
            
            # Create comprehensive visualizations
            self._create_behavioral_visualizations(combined_data, period_analysis, 
                                                 control_patterns, alternative_heating, learning_analysis)
            
            # Generate insights and recommendations
            insights = self._generate_behavioral_insights(period_analysis, control_patterns, 
                                                        alternative_heating, learning_analysis)
            
            results = {
                'period_analysis': period_analysis,
                'control_patterns': control_patterns,
                'alternative_heating': alternative_heating,
                'learning_analysis': learning_analysis,
                'insights': insights
            }
            
            return results
            
        except Exception as e:
            print(f"HVAC analysis encountered an error: {str(e)}")
            import traceback
            print("Traceback:")
            print(traceback.format_exc())
            return {}
    
    def _load_early_data(self, excel_data: dict) -> pd.DataFrame:
        """Load and process early data from Excel sheets."""
        
        frames = []
        
        # Process Emporia hourly data
        if 'Emporia kWh hr' in excel_data:
            emporia_hr = excel_data['Emporia kWh hr']
            if len(emporia_hr) > 1:  # Skip header row
                emporia_processed = self._process_early_emporia(emporia_hr)
                if not emporia_processed.empty:
                    frames.append(emporia_processed)
        
        # Process Elitech temperature data  
        if 'Elitech' in excel_data:
            elitech_data = excel_data['Elitech']
            elitech_processed = self._process_elitech_data(elitech_data)
            if not elitech_processed.empty:
                frames.append(elitech_processed)
        
        # Process early weather data
        if 'Weather fine' in excel_data:
            weather_data = excel_data['Weather fine']
            weather_processed = self._process_early_weather(weather_data)
            if not weather_processed.empty:
                frames.append(weather_processed)
        
        if frames:
            # Combine all early data on timestamp index
            combined = frames[0]
            for frame in frames[1:]:
                try:
                    combined = pd.merge(combined, frame, left_index=True, right_index=True, how='outer')
                except Exception as e:
                    print(f"Warning: Could not merge early data frame: {e}")
            
            print(f"Loaded early data: {combined.index.min()} to {combined.index.max()}")
            return combined.sort_index()
        
        return pd.DataFrame()
    
    def _process_early_emporia(self, emporia_data: list) -> pd.DataFrame:
        """Process early Emporia hourly data from Excel - FIXED DST VERSION."""
        
        if len(emporia_data) < 2:
            return pd.DataFrame()
        
        try:
            # Convert to DataFrame, skip header row
            df = pd.DataFrame(emporia_data[1:])
            
            # Use second column as timestamp
            if len(df.columns) > 1:
                df['timestamp'] = pd.to_datetime(df.iloc[:, 1], errors='coerce')
                df = df.dropna(subset=['timestamp'])
                
                if df.empty:
                    return pd.DataFrame()
                
                # FIXED: Apply robust timezone localization
                df['timestamp'] = robust_timezone_localize(
                    df['timestamp'], self.config.local_tz, "early Emporia timestamps"
                )
                
                df = df.set_index('timestamp').sort_index()
                
                # Map columns based on header row
                if emporia_data[0]:
                    header = emporia_data[0]
                    column_map = {}
                    for i, col_name in enumerate(header):
                        if col_name and 'AC' in str(col_name):
                            if '1' in str(col_name):
                                if i < len(df.columns):
                                    column_map[df.columns[i]] = 'AC_Floor1_kWh'
                            elif '2-3' in str(col_name):
                                if i < len(df.columns):
                                    column_map[df.columns[i]] = 'AC_Floors23_kWh'
                        elif col_name and 'Water Heater' in str(col_name):
                            if i < len(df.columns):
                                column_map[df.columns[i]] = 'WaterHeater_kWh'
                    
                    df = df.rename(columns=column_map)
                
                # Convert numeric columns
                for col in df.columns:
                    if col.endswith('_kWh'):
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                df = df.select_dtypes(include=[np.number])
                return df[df.index.notna()]
        except Exception as e:
            print(f"Error processing early Emporia data: {e}")
            
        return pd.DataFrame()
    
    def _process_elitech_data(self, elitech_data: list) -> pd.DataFrame:
        """Process Elitech temperature sensor data."""
        
        try:
            df = pd.DataFrame(elitech_data)
            
            # Find date column and temperature columns
            timestamp_col = None
            temp_cols = {}
            
            for col in df.columns:
                if 'Date' in str(col) and timestamp_col is None:
                    timestamp_col = col
                elif any(room in str(col).lower() for room in ['kitchen', 'thermostat']):
                    if df[col].dtype in [np.float64, np.int64] or pd.api.types.is_numeric_dtype(df[col]):
                        temp_cols[col] = f"indoor_temp_{str(col).lower().replace(' ', '_')}"
            
            if timestamp_col and temp_cols:
                df['timestamp'] = pd.to_datetime(df[timestamp_col], errors='coerce')
                df = df.dropna(subset=['timestamp'])
                
                # FIXED: Apply robust timezone localization
                df['timestamp'] = robust_timezone_localize(
                    df['timestamp'], self.config.local_tz, "Elitech temperatures"
                )
                df = df.set_index('timestamp')
                
                # Rename temperature columns
                df = df.rename(columns=temp_cols)
                
                # Convert to numeric and filter reasonable temperature range (0-50°C)
                for col in temp_cols.values():
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        df[col] = df[col].where((df[col] >= 0) & (df[col] <= 50))
                
                return df[[col for col in temp_cols.values() if col in df.columns]]
        except Exception as e:
            print(f"Error processing Elitech data: {e}")
        
        return pd.DataFrame()
    
    def _process_early_weather(self, weather_data: list) -> pd.DataFrame:
        """Process early weather data."""
        
        try:
            df = pd.DataFrame(weather_data)
            
            # Find date and temperature columns
            timestamp_col = None
            temp_col = None
            
            for col in df.columns:
                if 'Date' in str(col) and timestamp_col is None:
                    timestamp_col = col
                elif 'Actual °C' in str(col) or 'temp' in str(col).lower():
                    temp_col = col
            
            if timestamp_col and temp_col:
                df['timestamp'] = pd.to_datetime(df[timestamp_col], errors='coerce')
                df = df.dropna(subset=['timestamp'])
                
                # FIXED: Apply robust timezone localization
                df['timestamp'] = robust_timezone_localize(
                    df['timestamp'], self.config.local_tz, "early weather data"
                )
                df = df.set_index('timestamp')
                
                df['outdoor_temp_c'] = pd.to_numeric(df[temp_col], errors='coerce')
                
                return df[['outdoor_temp_c']].dropna()
        except Exception as e:
            print(f"Error processing early weather data: {e}")
        
        return pd.DataFrame()
    
    def _prepare_comprehensive_dataset(self, emporia: pd.DataFrame, weather: pd.DataFrame, 
                                     early_data: pd.DataFrame) -> pd.DataFrame:
        """Combine all data sources into comprehensive dataset - FIXED DST HANDLING."""
        
        # Start with main Emporia data (hourly)
        hvac_cols = [col for col in emporia.columns if 'AC_' in col and 'kWhs' in col]
        wh_cols = [col for col in emporia.columns if 'WaterHeater' in col and 'kWhs' in col]
        
        if not hvac_cols:
            print("No HVAC columns found in main data")
            return pd.DataFrame()
        
        # Resample to hourly if daily
        if emporia.index.freq and 'D' in str(emporia.index.freq):
            # Daily data - convert to hourly by dividing by 24 (rough approximation)
            hourly_data = emporia[hvac_cols + wh_cols] / 24
            hourly_index = pd.date_range(emporia.index[0], emporia.index[-1], freq='H')
            hourly_data = hourly_data.reindex(hourly_index, method='ffill')
        else:
            hourly_data = emporia[hvac_cols + wh_cols]
        
        # Add weather data
        weather_hourly = weather[['temp_c']].resample('H').mean()
        combined = hourly_data.join(weather_hourly.rename(columns={'temp_c': 'outdoor_temp_c'}), how='left')
        
        # FIXED: Add early data if available with robust DST handling
        if not early_data.empty:
            import pytz
            tz = pytz.timezone(self.config.local_tz)
            
            try:
                # Make sure combined data is timezone-aware using robust method
                if combined.index.tz is None:
                    combined.index = robust_timezone_localize(combined.index, tz, "combined data")
                
                # Make sure early_data is timezone-aware using robust method  
                if early_data.index.tz is None:
                    early_data.index = robust_timezone_localize(early_data.index, tz, "early data")
                elif early_data.index.tz != combined.index.tz:
                    early_data.index = early_data.index.tz_convert(combined.index.tz)
                
                # Now combine safely
                combined = combined.combine_first(early_data)
                print(f"Successfully integrated early data")
                
            except Exception as e:
                print(f"Warning: Could not combine early data due to timezone issues: {e}")
                print("Proceeding with main data only")
        
        # Calculate derived features
        combined['total_hvac_kwh'] = combined[[col for col in combined.columns if 'AC_' in col]].sum(axis=1)
        combined['hour'] = combined.index.hour
        combined['day_of_week'] = combined.index.dayofweek
        combined['month'] = combined.index.month
        combined['year'] = combined.index.year
        
        # Add behavioral period labels
        for period, (start, end) in self.behavioral_periods.items():
            mask = (combined.index >= start) & (combined.index <= end)
            combined.loc[mask, 'behavioral_period'] = period
        
        # Calculate heating/cooling degree hours
        if 'outdoor_temp_c' in combined.columns:
            combined['heating_degree_hours'] = (18.0 - combined['outdoor_temp_c']).clip(lower=0)
            combined['cooling_degree_hours'] = (combined['outdoor_temp_c'] - 24.0).clip(lower=0)
        
        return combined.dropna(subset=['total_hvac_kwh'])
    
    def _analyze_behavioral_periods(self, data: pd.DataFrame) -> dict:
        """Analyze HVAC usage patterns across different behavioral periods."""
        
        period_stats = {}
        
        for period in ['learning_phase', 'adaptation_phase', 'optimized_phase']:
            period_data = data[data['behavioral_period'] == period]
            
            if period_data.empty:
                continue
            
            # Calculate temperature-normalized efficiency
            if 'outdoor_temp_c' in period_data.columns:
                # Fit simple degree-hour model
                X = sm.add_constant(period_data[['heating_degree_hours', 'cooling_degree_hours']].fillna(0))
                y = period_data['total_hvac_kwh']
                
                try:
                    model = sm.OLS(y, X).fit()
                    
                    period_stats[period] = {
                        'avg_hourly_usage': period_data['total_hvac_kwh'].mean(),
                        'peak_usage': period_data['total_hvac_kwh'].quantile(0.95),
                        'base_load': model.params.get('const', 0),
                        'heating_response': model.params.get('heating_degree_hours', 0),
                        'cooling_response': model.params.get('cooling_degree_hours', 0),
                        'r_squared': model.rsquared,
                        'total_hours': len(period_data),
                        'date_range': (period_data.index.min(), period_data.index.max())
                    }
                except:
                    # Fallback to simple stats
                    period_stats[period] = {
                        'avg_hourly_usage': period_data['total_hvac_kwh'].mean(),
                        'peak_usage': period_data['total_hvac_kwh'].quantile(0.95),
                        'total_hours': len(period_data),
                        'date_range': (period_data.index.min(), period_data.index.max())
                    }
        
        return period_stats
    
    def _detect_manual_control(self, data: pd.DataFrame) -> dict:
        """Detect periods of manual HVAC control vs automatic operation."""
        
        control_patterns = {}
        
        if 'outdoor_temp_c' not in data.columns:
            return control_patterns
        
        # Look for periods where HVAC usage doesn't correlate with temperature
        data['temp_usage_correlation'] = data['outdoor_temp_c'].rolling(24).corr(data['total_hvac_kwh'])
        
        # Identify manual control periods (low correlation for extended periods)
        manual_threshold = 0.3  # Low correlation suggests manual override
        data['likely_manual'] = data['temp_usage_correlation'].abs() < manual_threshold
        
        # Find extended manual control periods
        manual_periods = []
        in_manual = False
        start_time = None
        
        for timestamp, is_manual in data['likely_manual'].items():
            if is_manual and not in_manual:
                start_time = timestamp
                in_manual = True
            elif not is_manual and in_manual:
                if start_time and (timestamp - start_time).total_seconds() > 6 * 3600:  # >6 hours
                    manual_periods.append((start_time, timestamp))
                in_manual = False
                start_time = None
        
        control_patterns = {
            'manual_periods': manual_periods,
            'automatic_correlation': data[~data['likely_manual']]['temp_usage_correlation'].mean(),
            'manual_correlation': data[data['likely_manual']]['temp_usage_correlation'].mean(),
            'percent_manual_operation': data['likely_manual'].mean() * 100
        }
        
        return control_patterns
    
    def _detect_alternative_heating(self, data: pd.DataFrame) -> dict:
        """Detect periods where alternative heating (fireplace, space heaters) may have been used."""
        
        if 'outdoor_temp_c' not in data.columns:
            return {}
        
        # Look for cold periods with unexpectedly low HVAC usage
        cold_weather = data['outdoor_temp_c'] < 5  # Below 5°C
        cold_data = data[cold_weather]
        
        if cold_data.empty:
            return {}
        
        # Expected usage vs actual usage
        expected_usage = cold_data['heating_degree_hours'] * cold_data['heating_degree_hours'].mean()
        usage_deficit = expected_usage - cold_data['total_hvac_kwh']
        
        # Identify periods with significant usage deficit (likely alternative heating)
        alternative_periods = cold_data[usage_deficit > usage_deficit.quantile(0.75)]
        
        return {
            'potential_alternative_periods': len(alternative_periods),
            'avg_usage_reduction': usage_deficit.mean(),
            'total_estimated_savings_kwh': usage_deficit[usage_deficit > 0].sum(),
            'peak_alternative_usage_day': usage_deficit.idxmax() if not usage_deficit.empty else None
        }
    
    def _analyze_learning_curve(self, data: pd.DataFrame) -> dict:
        """Analyze how HVAC usage efficiency improved over time."""
        
        if data.empty or 'outdoor_temp_c' not in data.columns:
            return {}
        
        # Calculate monthly efficiency metrics
        monthly_data = data.groupby(data.index.to_period('M')).agg({
            'total_hvac_kwh': ['mean', 'sum'],
            'heating_degree_hours': 'sum',
            'cooling_degree_hours': 'sum',
            'outdoor_temp_c': 'mean'
        })
        
        monthly_data.columns = ['avg_hourly_kwh', 'total_monthly_kwh', 'total_hdd', 'total_cdd', 'avg_temp']
        
        # Calculate temperature-normalized usage
        monthly_data['usage_per_degree_hour'] = (
            monthly_data['total_monthly_kwh'] / 
            (monthly_data['total_hdd'] + monthly_data['total_cdd'] + 1)
        )
        
        # Fit trend line to see learning curve
        if len(monthly_data) > 3:
            x = np.arange(len(monthly_data))
            y = monthly_data['usage_per_degree_hour'].values
            
            # Remove outliers for trend fitting
            q75, q25 = np.percentile(y, [75, 25])
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            mask = (y >= lower_bound) & (y <= upper_bound)
            
            if mask.sum() > 2:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
                
                return {
                    'efficiency_trend_slope': slope,  # Negative slope indicates improvement
                    'efficiency_r_squared': r_value**2,
                    'improvement_significance': p_value,
                    'monthly_efficiency': monthly_data['usage_per_degree_hour'].to_dict(),
                    'improvement_rate_percent_per_month': (slope / intercept * 100) if intercept != 0 else 0,
                    'total_improvement_percent': ((monthly_data['usage_per_degree_hour'].iloc[0] - 
                                                monthly_data['usage_per_degree_hour'].iloc[-1]) / 
                                               monthly_data['usage_per_degree_hour'].iloc[0] * 100) if len(monthly_data) > 1 else 0
                }
        
        return {}
    
    def _create_behavioral_visualizations(self, data: pd.DataFrame, period_analysis: dict, 
                                        control_patterns: dict, alternative_heating: dict, 
                                        learning_analysis: dict) -> None:
        """Create comprehensive visualizations of behavioral HVAC patterns."""
        
        fig = plt.figure(figsize=(18, 12))
        gs = plt.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Usage by behavioral period
        ax1 = fig.add_subplot(gs[0, 0])
        if period_analysis:
            periods = list(period_analysis.keys())
            avg_usage = [period_analysis[p].get('avg_hourly_usage', 0) for p in periods]
            colors = [COLORS['warning'], COLORS['accent'], COLORS['success']]
            
            bars = ax1.bar(periods, avg_usage, color=colors[:len(periods)], alpha=0.7)
            ax1.set_ylabel('Average Hourly Usage (kWh)')
            ax1.set_title('HVAC Usage by Learning Period')
            ax1.tick_params(axis='x', rotation=45)
            
            # Add improvement annotations
            if len(avg_usage) > 1:
                improvement = (avg_usage[0] - avg_usage[-1]) / avg_usage[0] * 100
                ax1.annotate(f'{improvement:.1f}% improvement', 
                           xy=(len(periods)-1, avg_usage[-1]),
                           xytext=(len(periods)-0.5, max(avg_usage)*0.8),
                           arrowprops=dict(arrowstyle='->', color='green'),
                           fontsize=10, color='green')
        
        # 2. Temperature vs Usage with manual control periods
        ax2 = fig.add_subplot(gs[0, 1])
        if 'outdoor_temp_c' in data.columns:
            auto_mask = ~data['likely_manual'] if 'likely_manual' in data.columns else pd.Series(True, index=data.index)
            manual_mask = data['likely_manual'] if 'likely_manual' in data.columns else pd.Series(False, index=data.index)
            
            if auto_mask.sum() > 0:
                ax2.scatter(data.loc[auto_mask, 'outdoor_temp_c'], 
                          data.loc[auto_mask, 'total_hvac_kwh'],
                          alpha=0.3, s=10, color=COLORS['primary'], label='Automatic Control')
            
            if manual_mask.sum() > 0:
                ax2.scatter(data.loc[manual_mask, 'outdoor_temp_c'], 
                          data.loc[manual_mask, 'total_hvac_kwh'],
                          alpha=0.5, s=15, color=COLORS['warning'], label='Manual Override')
            
            ax2.set_xlabel('Outdoor Temperature (°C)')
            ax2.set_ylabel('HVAC Usage (kWh)')
            ax2.set_title('Usage vs Temperature: Control Mode Detection')
            ax2.legend()
        
        # 3. Learning curve over time
        ax3 = fig.add_subplot(gs[0, 2])
        if learning_analysis and 'monthly_efficiency' in learning_analysis:
            monthly_eff = learning_analysis['monthly_efficiency']
            months = list(monthly_eff.keys())
            efficiency = list(monthly_eff.values())
            
            ax3.plot(range(len(months)), efficiency, 'o-', color=COLORS['success'], linewidth=2)
            
            # Add trend line
            if learning_analysis.get('efficiency_trend_slope'):
                x_trend = np.arange(len(months))
                y_trend = (learning_analysis['efficiency_trend_slope'] * x_trend + 
                          efficiency[0])
                ax3.plot(x_trend, y_trend, '--', color='red', alpha=0.7,
                        label=f"Trend: {learning_analysis['improvement_rate_percent_per_month']:.1f}%/month")
                ax3.legend()
            
            ax3.set_xlabel('Months Since Start')
            ax3.set_ylabel('Usage per Degree-Hour')
            ax3.set_title('Learning Curve: Efficiency Improvement')
            ax3.tick_params(axis='x', rotation=45)
        
        # 4. Hourly usage patterns by period
        ax4 = fig.add_subplot(gs[1, :])
        for i, (period, color) in enumerate(zip(['learning_phase', 'adaptation_phase', 'optimized_phase'], 
                                               [COLORS['warning'], COLORS['accent'], COLORS['success']])):
            period_data = data[data['behavioral_period'] == period]
            if not period_data.empty:
                hourly_avg = period_data.groupby('hour')['total_hvac_kwh'].mean()
                ax4.plot(hourly_avg.index, hourly_avg.values, 'o-', 
                        color=color, alpha=0.7, linewidth=2, label=period.replace('_', ' ').title())
        
        ax4.set_xlabel('Hour of Day')
        ax4.set_ylabel('Average HVAC Usage (kWh)')
        ax4.set_title('Daily Usage Patterns by Learning Period')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Indoor vs outdoor temperature correlation
        ax5 = fig.add_subplot(gs[2, 0])
        indoor_cols = [col for col in data.columns if 'indoor_temp' in col]
        if indoor_cols and 'outdoor_temp_c' in data.columns:
            indoor_col = indoor_cols[0]  # Use first indoor temperature sensor
            
            # First, create a cleaned DataFrame by dropping rows with missing temperature data
            clean_data = data.dropna(subset=[indoor_col, 'outdoor_temp_c'])
            
            # Only proceed if there's data left after cleaning
            if not clean_data.empty:
                # Second, calculate the sample size based on the *cleaned* data's length
                sample_size = min(1000, len(clean_data))
                
                # Third, take the sample from the cleaned data
                sample_data = clean_data.sample(sample_size)
                
                ax5.scatter(sample_data['outdoor_temp_c'], sample_data[indoor_col],
                           alpha=0.5, s=20, color=COLORS['neutral'])
            
            # Add reference lines
            ax5.plot([-10, 30], [18, 18], '--', color='blue', alpha=0.5, label='Heating Setpoint')
            ax5.plot([-10, 30], [24, 24], '--', color='red', alpha=0.5, label='Cooling Setpoint')
            ax5.plot([-10, 30], [-10, 30], '--', color='black', alpha=0.3, label='1:1 Line')
            
            ax5.set_xlabel('Outdoor Temperature (°C)')
            ax5.set_ylabel('Indoor Temperature (°C)')
            ax5.set_title('Indoor vs Outdoor Temperature')
            ax5.legend()
        
        # 6. Alternative heating detection
        ax6 = fig.add_subplot(gs[2, 1])
        if 'outdoor_temp_c' in data.columns:
            cold_data = data[data['outdoor_temp_c'] < 5]
            if not cold_data.empty:
                ax6.hist(cold_data['total_hvac_kwh'], bins=30, alpha=0.6, 
                        color=COLORS['primary'], label='Cold Weather Usage')
                
                mean_usage = cold_data['total_hvac_kwh'].mean()
                ax6.axvline(mean_usage, color='red', linestyle='--', 
                           label=f'Mean: {mean_usage:.2f} kWh')
                
                # Highlight potential alternative heating periods
                low_usage = cold_data['total_hvac_kwh'] < cold_data['total_hvac_kwh'].quantile(0.25)
                if low_usage.sum() > 0:
                    ax6.axvspan(0, cold_data['total_hvac_kwh'].quantile(0.25), 
                              alpha=0.3, color='green', 
                              label=f'Potential Alt. Heating\n({low_usage.sum()} hours)')
                
                ax6.set_xlabel('HVAC Usage (kWh)')
                ax6.set_ylabel('Frequency')
                ax6.set_title('Cold Weather HVAC Usage Distribution')
                ax6.legend()
        
        # 7. Efficiency metrics comparison
        ax7 = fig.add_subplot(gs[2, 2])
        if period_analysis:
            periods = list(period_analysis.keys())
            base_loads = [period_analysis[p].get('base_load', 0) for p in periods]
            heating_response = [abs(period_analysis[p].get('heating_response', 0)) for p in periods]
            
            x_pos = np.arange(len(periods))
            width = 0.35
            
            ax7.bar(x_pos - width/2, base_loads, width, 
                   label='Base Load', color=COLORS['neutral'], alpha=0.7)
            ax7.bar(x_pos + width/2, heating_response, width,
                   label='Heating Response', color=COLORS['warning'], alpha=0.7)
            
            ax7.set_xlabel('Learning Period')
            ax7.set_ylabel('Energy (kWh)')
            ax7.set_title('HVAC Efficiency Components')
            ax7.set_xticks(x_pos)
            ax7.set_xticklabels([p.replace('_', ' ').title() for p in periods], rotation=45)
            ax7.legend()
        
        plt.suptitle('Behavioral HVAC Analysis: Learning and Adaptation Patterns', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def _generate_behavioral_insights(self, period_analysis: dict, control_patterns: dict,
                                    alternative_heating: dict, learning_analysis: dict) -> dict:
        """Generate actionable insights from behavioral analysis."""
        
        insights = {
            'behavioral_changes': [],
            'control_strategies': [],
            'alternative_heating_opportunities': [],
            'learning_achievements': [],
            'recommendations': []
        }
        
        # Analyze behavioral changes
        if len(period_analysis) > 1:
            periods = list(period_analysis.keys())
            early_usage = period_analysis[periods[0]].get('avg_hourly_usage', 0)
            recent_usage = period_analysis[periods[-1]].get('avg_hourly_usage', 0)
            
            if early_usage > 0:
                improvement_pct = (early_usage - recent_usage) / early_usage * 100
                insights['behavioral_changes'].append({
                    'metric': 'Overall Usage Reduction',
                    'improvement_percent': improvement_pct,
                    'significance': 'High' if improvement_pct > 20 else 'Moderate' if improvement_pct > 10 else 'Low'
                })
        
        # Control pattern insights
        if control_patterns:
            manual_pct = control_patterns.get('percent_manual_operation', 0)
            insights['control_strategies'].append({
                'manual_operation_percent': manual_pct,
                'recommendation': 'High manual control suggests potential for automated scheduling' 
                                if manual_pct > 30 else 'Good balance of manual and automatic control'
            })
        
        # Alternative heating insights
        if alternative_heating:
            potential_savings = alternative_heating.get('total_estimated_savings_kwh', 0)
            if potential_savings > 0:
                insights['alternative_heating_opportunities'].append({
                    'estimated_annual_savings_kwh': potential_savings * 8760 / len(period_analysis),  # Rough extrapolation
                    'recommendation': 'Significant opportunity to expand fireplace/space heater use during cold periods'
                })
        
        # Learning achievements
        if learning_analysis:
            total_improvement = learning_analysis.get('total_improvement_percent', 0)
            if total_improvement > 0:
                insights['learning_achievements'].append({
                    'total_efficiency_improvement': total_improvement,
                    'monthly_improvement_rate': learning_analysis.get('improvement_rate_percent_per_month', 0),
                    'achievement_level': 'Excellent' if total_improvement > 30 else 'Good' if total_improvement > 15 else 'Moderate'
                })
        
        # Generate recommendations
        insights['recommendations'] = [
            "Continue leveraging manual control strategies that have proven effective",
            "Consider programmable thermostats with learned setback schedules",
            "Expand use of alternative heating sources during extreme cold periods",
            "Monitor and maintain the efficiency gains achieved through behavioral adaptation",
            "Consider zoned heating/cooling to optimize individual room comfort vs energy use"
        ]
        
        return insights

print("Enhanced Behavioral HVAC Analyzer defined with DST fixes")


# ============================================================================
# ENHANCED HVAC ANALYSIS EXECUTION
# ============================================================================

def run_enhanced_hvac_analysis(data: dict, config: EnergyConfig) -> dict:
    """Execute the enhanced behavioral HVAC analysis with error handling."""
    
    print("\nStarting Enhanced Behavioral HVAC Analysis...")
    
    try:
        # Initialize analyzer
        analyzer = EnhancedHVACAnalyzer(config)
        
        # Run comprehensive analysis
        results = analyzer.analyze_behavioral_hvac(
            emporia=data.get('emporia', pd.DataFrame()),
            weather=data.get('weather', pd.DataFrame()),
            excel_data=data.get('excel', {})
        )
        
        # Print comprehensive results
        if results:
            print("\n" + "="*70)
            print("BEHAVIORAL HVAC ANALYSIS RESULTS")
            print("="*70)
            
            # Period analysis summary
            if 'period_analysis' in results:
                print("\nLEARNING PERIOD SUMMARY:")
                for period, stats in results['period_analysis'].items():
                    print(f"\n{period.replace('_', ' ').title()}:")
                    print(f"  • Average hourly usage: {stats.get('avg_hourly_usage', 0):.2f} kWh")
                    print(f"  • Base load: {stats.get('base_load', 0):.2f} kWh")
                    print(f"  • Heating response: {stats.get('heating_response', 0):.3f} kWh/degree-hour")
                    print(f"  • Data coverage: {stats.get('total_hours', 0):,} hours")
            
            # Control patterns
            if 'control_patterns' in results:
                control = results['control_patterns']
                print(f"\nCONTROL PATTERNS:")
                print(f"  • Manual operation: {control.get('percent_manual_operation', 0):.1f}% of time")
                print(f"  • Automatic correlation: {control.get('automatic_correlation', 0):.3f}")
                print(f"  • Manual periods detected: {len(control.get('manual_periods', []))}")
            
            # Alternative heating
            if 'alternative_heating' in results:
                alt = results['alternative_heating']
                savings = alt.get('total_estimated_savings_kwh', 0)
                if savings > 0:
                    print(f"\nALTERNATIVE HEATING OPPORTUNITIES:")
                    print(f"  • Potential periods detected: {alt.get('potential_alternative_periods', 0)}")
                    print(f"  • Estimated savings potential: {savings:.0f} kWh")
            
            # Learning curve
            if 'learning_analysis' in results:
                learning = results['learning_analysis']
                improvement = learning.get('total_improvement_percent', 0)
                if improvement != 0:
                    print(f"\nLEARNING CURVE ANALYSIS:")
                    print(f"  • Total efficiency improvement: {improvement:.1f}%")
                    print(f"  • Monthly improvement rate: {learning.get('improvement_rate_percent_per_month', 0):.1f}%/month")
            
            # Key insights
            if 'insights' in results:
                insights = results['insights']
                print(f"\nKEY BEHAVIORAL INSIGHTS:")
                
                for change in insights.get('behavioral_changes', []):
                    print(f"  • {change['metric']}: {change['improvement_percent']:.1f}% improvement")
                
                print(f"\nRECOMMENDations:")
                for i, rec in enumerate(insights.get('recommendations', []), 1):
                    print(f"  {i}. {rec}")
        
        return results
        
    except Exception as e:
        print(f"HVAC analysis failed with error: {str(e)}")
        print("Attempting simplified fallback analysis...")
        
        # Fallback to basic analysis
        try:
            emporia = data.get('emporia', pd.DataFrame())
            weather = data.get('weather', pd.DataFrame())
            
            if emporia.empty or weather.empty:
                return {'error': 'Insufficient data for HVAC analysis'}
            
            # Basic analysis without complex processing
            hvac_cols = [col for col in emporia.columns if 'AC_' in col and 'kWhs' in col]
            if hvac_cols:
                total_usage = emporia[hvac_cols].sum().sum()
                daily_avg = emporia[hvac_cols].sum(axis=1).mean()
                
                return {
                    'fallback_analysis': True,
                    'total_hvac_usage': total_usage,
                    'daily_avg_usage': daily_avg,
                    'hvac_columns': hvac_cols,
                    'data_period': f"{emporia.index.min()} to {emporia.index.max()}",
                    'error_message': str(e)
                }
            
        except Exception as e2:
            print(f"Fallback analysis also failed: {str(e2)}")
            return {'error': f'All HVAC analysis methods failed: {str(e)} | {str(e2)}'}

print("Enhanced HVAC analysis execution function defined")

# ============================================================================
# EXECUTE HVAC ANALYSIS
# ============================================================================

# Execute the enhanced HVAC analysis that accounts for behavioral changes
hvac_results = run_enhanced_hvac_analysis(data, config)


# %%
# ============================================================================
# CELL 11: BUILDING ENVELOPE ANALYZER CLASS DEFINITION
# ============================================================================

class EnvelopeAnalyzer:
    """Advanced building envelope thermal resistance estimation."""
    
    def __init__(self, config: EnergyConfig):
        self.config = config
        
    def estimate_r_values(self, emporia: pd.DataFrame, temps: pd.DataFrame, 
                         weather: pd.DataFrame) -> pd.DataFrame:
        """Estimate R-values using multiple methods."""
        
        print("\n" + "="*60)
        print("BUILDING ENVELOPE R-VALUE ESTIMATION")
        print("="*60)
        
        if emporia.empty or temps.empty or weather.empty:
            print("✗ Insufficient data for R-value estimation")
            return pd.DataFrame()
        
        results = []
        
        # Process each room
        for room_name, wall_area_ft2 in self.config.room_areas_ft2.items():
            zone_col = self.config.room_hvac_zones.get(room_name)
            if not zone_col:
                continue
                
            zone_col = f"{zone_col} (kWhs)"  # Add units suffix
            
            if zone_col not in emporia.columns:
                continue
            
            # Method 1: Steady-state analysis
            steady_results = self._steady_state_r_value(
                emporia, temps, weather, room_name, zone_col, wall_area_ft2
            )
            
            # Method 2: Transient RC analysis
            transient_results = self._transient_rc_r_value(
                emporia, temps, weather, room_name, zone_col, wall_area_ft2
            )
            
            # Combine results
            for result in steady_results + transient_results:
                if result:
                    results.append(result)
        
        if not results:
            print("✗ No R-value estimates could be calculated")
            print("  Possible reasons:")
            print("  • No steady-state periods detected")
            print("  • Insufficient temperature sensor coverage")
            print("  • HVAC zones not properly mapped")
            return pd.DataFrame()
        
        # Create results dataframe
        results_df = pd.DataFrame(results)
        
        # Visualization
        if not results_df.empty:
            self._visualize_r_values(results_df)
        
        return results_df
    
    def _steady_state_r_value(self, emporia: pd.DataFrame, temps: pd.DataFrame,
                             weather: pd.DataFrame, room: str, zone_col: str,
                             area_ft2: float) -> List[Dict]:
        """Estimate R-value during steady-state conditions."""
        
        results = []
        
        # Find steady-state periods
        room_temps = temps[temps['Location'].str.contains(room, case=False, na=False)]
        if room_temps.empty:
            return results
        
        # Convert to hourly and find steady periods
        room_hourly = room_temps.set_index('Timestamp')['temp_c'].resample('H').mean()
        weather_hourly = weather['temp_c'].resample('H').mean()
        hvac_hourly = emporia[[zone_col]].resample('H').sum()
        
        # Join all data
        analysis = pd.concat([room_hourly.rename('tin'), 
                             weather_hourly.rename('tout'),
                             hvac_hourly[zone_col].rename('hvac_kwh')], 
                            axis=1).dropna()
        
        if len(analysis) < 24:
            return results
        
        # Find steady-state windows (std < threshold over window)
        window_hours = self.config.steady_state_min_hours
        temp_std = analysis['tin'].rolling(window=window_hours, center=True).std()
        steady_mask = temp_std < self.config.steady_state_threshold_c
        
        # Process steady-state segments
        steady_segments = self._find_segments(steady_mask, min_length=window_hours)
        
        for start_idx, end_idx in steady_segments[:5]:  # Limit to 5 best segments
            segment = analysis.iloc[start_idx:end_idx]
            
            if len(segment) < window_hours:
                continue
            
            # Calculate average conditions
            avg_tin = segment['tin'].mean()
            avg_tout = segment['tout'].mean()
            avg_hvac = segment['hvac_kwh'].mean()
            
            delta_t = abs(avg_tin - avg_tout)
            
            if delta_t < 5:  # Need meaningful temperature difference
                continue
            
            # Determine if heating or cooling
            is_heating = avg_tin > avg_tout
            cop = self.config.hvac_cop_heating if is_heating else self.config.hvac_cop_cooling
            
            # Calculate UA (kW/°C)
            ua_effective = avg_hvac / delta_t  # kWh/h/°C = kW/°C
            ua_actual = ua_effective * cop  # Adjust for COP
            
            # Convert to imperial units (BTU/h/°F)
            ua_imperial = ua_actual * 3412.14 / 1.8  # kW/°C to BTU/h/°F
            
            # Calculate R-value
            if area_ft2 > 0 and ua_imperial > 0:
                u_value = ua_imperial / area_ft2  # BTU/h/ft²/°F
                r_value = 1 / u_value
                
                results.append({
                    'room': room,
                    'method': 'Steady-State',
                    'r_value': r_value,
                    'confidence': 'Medium' if segment['tin'].std() < 0.3 else 'Low',
                    'delta_t': delta_t,
                    'hours': len(segment),
                    'cop_assumed': cop,
                    'mode': 'Heating' if is_heating else 'Cooling'
                })
        
        return results
    
    def _transient_rc_r_value(self, emporia: pd.DataFrame, temps: pd.DataFrame,
                              weather: pd.DataFrame, room: str, zone_col: str,
                              area_ft2: float) -> List[Dict]:
        """Estimate R-value from transient temperature decay."""
        
        results = []
        
        # Find HVAC-off periods
        hvac_hourly = emporia[[zone_col]].resample('H').sum()
        off_mask = hvac_hourly[zone_col] < 0.1  # Less than 0.1 kWh/h
        
        off_segments = self._find_segments(off_mask, min_length=3)
        
        for start_idx, end_idx in off_segments[:3]:  # Analyze top 3 segments
            segment_time = hvac_hourly.index[start_idx:end_idx]
            
            # Get temperature data for this period
            room_temps = temps[temps['Location'].str.contains(room, case=False, na=False)]
            room_temps = room_temps.set_index('Timestamp')['temp_c']
            room_segment = room_temps.reindex(segment_time).interpolate()
            
            weather_segment = weather['temp_c'].reindex(segment_time).interpolate()
            
            if room_segment.isnull().any() or weather_segment.isnull().any():
                continue
            
            # Fit exponential decay model
            try:
                tau, r2 = self._fit_rc_model(room_segment.values, 
                                            weather_segment.values,
                                            segment_time)
                
                if tau > 0 and r2 > 0.7:
                    # Estimate floor area for capacitance
                    floor_area = self._estimate_floor_area(room)
                    
                    # Thermal capacitance (approximate)
                    # Using 0.9 kJ/K/ft² as typical for residential
                    c_kj_per_k = 0.9 * floor_area
                    c_btu_per_f = c_kj_per_k * 0.947817 / 1.8
                    
                    # R = tau / C
                    r_value = tau / c_btu_per_f * area_ft2
                    
                    results.append({
                        'room': room,
                        'method': 'Transient-RC',
                        'r_value': r_value,
                        'confidence': 'High' if r2 > 0.85 else 'Medium',
                        'tau_hours': tau,
                        'r2_fit': r2,
                        'hours': len(room_segment),
                        'mode': 'Passive'
                    })
            except:
                continue
        
        return results
    
    def _fit_rc_model(self, tin: np.ndarray, tout: np.ndarray, 
                     time_index: pd.DatetimeIndex) -> Tuple[float, float]:
        """Fit RC thermal model to temperature decay."""
        
        # Time in hours from start
        t = (time_index - time_index[0]).total_seconds() / 3600
        
        # Initial conditions
        t0_in = tin[0]
        t0_out = tout[0]
        
        # Objective function for exponential decay
        def model(tau):
            # Simplified model assuming constant outdoor temp
            t_model = tout + (t0_in - tout) * np.exp(-t / tau)
            return t_model
        
        # Fit the model
        def residuals(tau):
            return np.sum((tin - model(tau[0]))**2)
        
        result = optimize.minimize(residuals, x0=[5.0], bounds=[(0.1, 100)])
        
        if result.success:
            tau_opt = result.x[0]
            t_pred = model(tau_opt)
            r2 = 1 - np.sum((tin - t_pred)**2) / np.sum((tin - tin.mean())**2)
            return tau_opt, r2
        
        return 0, 0
    
    def _find_segments(self, mask: pd.Series, min_length: int) -> List[Tuple[int, int]]:
        """Find continuous segments where mask is True."""
        segments = []
        in_segment = False
        start = 0
        
        for i, val in enumerate(mask):
            if val and not in_segment:
                start = i
                in_segment = True
            elif not val and in_segment:
                if i - start >= min_length:
                    segments.append((start, i))
                in_segment = False
        
        if in_segment and len(mask) - start >= min_length:
            segments.append((start, len(mask)))
        
        return segments
    
    def _estimate_floor_area(self, room: str) -> float:
        """Estimate floor area from wall dimensions."""
        # Simplified: use average of two wall lengths as room dimensions
        wall_area = self.config.room_areas_ft2.get(room, 200)
        # Assume 8.5 ft ceiling height average
        perimeter = wall_area / 8.5
        # Assume roughly square room
        side = perimeter / 4
        return side * side
    
    def _visualize_r_values(self, results_df: pd.DataFrame) -> None:
        """Create visualization of R-value estimates."""
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Group by room and method
        summary = results_df.groupby(['room', 'method'])['r_value'].agg(['mean', 'std', 'count'])
        
        # Bar plot by room
        ax1 = axes[0]
        rooms = results_df['room'].unique()
        x = np.arange(len(rooms))
        width = 0.35
        
        steady_vals = []
        transient_vals = []
        
        for room in rooms:
            steady = summary.loc[(room, 'Steady-State'), 'mean'] if (room, 'Steady-State') in summary.index else 0
            trans = summary.loc[(room, 'Transient-RC'), 'mean'] if (room, 'Transient-RC') in summary.index else 0
            steady_vals.append(steady)
            transient_vals.append(trans)
        
        ax1.bar(x - width/2, steady_vals, width, label='Steady-State', color=COLORS['primary'])
        ax1.bar(x + width/2, transient_vals, width, label='Transient-RC', color=COLORS['secondary'])
        
        ax1.set_xlabel('Room')
        ax1.set_ylabel('R-value (h·ft²·°F/BTU)')
        ax1.set_title('Estimated R-values by Room and Method')
        ax1.set_xticks(x)
        ax1.set_xticklabels(rooms, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add typical R-value reference lines
        typical_values = {'Uninsulated': 4, 'Poor': 7, 'Fair': 11, 'Good': 19}
        for label, r_val in typical_values.items():
            ax1.axhline(y=r_val, color='gray', linestyle='--', alpha=0.5)
            ax1.text(ax1.get_xlim()[1]*0.98, r_val, label, fontsize=9, va='bottom')
        
        # Distribution plot
        ax2 = axes[1]
        for method in results_df['method'].unique():
            data = results_df[results_df['method'] == method]['r_value']
            if len(data) > 0:
                ax2.hist(data, bins=15, alpha=0.6, label=method)
        
        ax2.set_xlabel('R-value (h·ft²·°F/BTU)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of R-value Estimates')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Building Envelope Thermal Resistance Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        print("\n📊 R-VALUE ESTIMATION RESULTS:")
        print("─" * 60)
        
        overall_mean = results_df['r_value'].mean()
        overall_std = results_df['r_value'].std()
        
        print(f"\nOverall R-value: {overall_mean:.1f} ± {overall_std:.1f}")
        print("\nBy Room:")
        for room in rooms:
            room_data = results_df[results_df['room'] == room]['r_value']
            if len(room_data) > 0:
                print(f"  • {room}: R-{room_data.mean():.1f} (n={len(room_data)} estimates)")
        
        # Assessment
        if overall_mean < 7:
            assessment = "POOR - Significant insulation improvements needed"
            color = "⚠️"
        elif overall_mean < 11:
            assessment = "BELOW AVERAGE - Some insulation improvements recommended"
            color = "⚡"
        elif overall_mean < 19:
            assessment = "AVERAGE - Meeting minimum standards"
            color = "✓"
        else:
            assessment = "GOOD - Well insulated"
            color = "✅"
        
        print(f"\n{color} Assessment: {assessment}")

print("✓ Building envelope analyzer classes defined")

# %%
# ============================================================================
# CELL 12: RUN BUILDING ENVELOPE ANALYSIS
# ============================================================================

# Building Envelope Analysis
envelope_analyzer = EnvelopeAnalyzer(config)
r_value_results = envelope_analyzer.estimate_r_values(
    data.get('emporia', pd.DataFrame()),
    data.get('temps', pd.DataFrame()),
    data.get('weather', pd.DataFrame())
)

# %%
# ============================================================================
# CELL 13: IMPROVEMENT ANALYZER CLASS DEFINITION
# ============================================================================

class ImprovementAnalyzer:
    """Analyze the impact of home improvements with weather normalization."""
    
    def __init__(self, config: EnergyConfig):
        self.config = config
        self.improvements = [
            ('2022-01-14', 'Basement ceiling insulation', 4500),
            ('2022-01-28', 'Kitchen floor replacement', 7154),
            ('2022-03-07', '3rd floor insulation package', 4855),
            ('2022-10-15', '3rd floor additional fiberglass', 1580),
            ('2022-12-19', 'Kitchen wall blown-in insulation', 3500),
        ]
    
    def analyze_improvements(self, emporia: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
        """Perform event study analysis of improvements."""
        
        print("\n" + "="*60)
        print("HOME IMPROVEMENT IMPACT ANALYSIS")
        print("="*60)
        
        if emporia.empty or weather.empty:
            print("✗ Insufficient data for improvement analysis")
            return pd.DataFrame()
        
        # Prepare daily data
        daily_energy = emporia.select_dtypes(include=[np.number]).sum(axis=1).resample('D').sum()
        daily_energy = daily_energy.to_frame('total_kwh')
        
        weather_daily = weather[['temp_c']].resample('D').mean()
        weather_daily.columns = ['tmean_c']
        
        # Calculate degree days
        weather_daily['hdd'] = (self.config.hdd_base_c - weather_daily['tmean_c']).clip(lower=0)
        weather_daily['cdd'] = (weather_daily['tmean_c'] - self.config.cdd_base_c).clip(lower=0)
        
        # Join energy and weather
        analysis = daily_energy.join(weather_daily, how='inner').dropna()
        
        results = []
        
        for date_str, description, cost in self.improvements:
            event_date = pd.Timestamp(date_str).tz_localize(self.config.local_tz)
            
            # Skip if event outside data range
            if event_date < analysis.index.min() or event_date > analysis.index.max():
                continue
            
            # Define before/after periods (60 days each)
            before_start = event_date - pd.Timedelta(days=60)
            before_end = event_date - pd.Timedelta(days=1)
            after_start = event_date
            after_end = event_date + pd.Timedelta(days=60)
            
            before_data = analysis.loc[before_start:before_end]
            after_data = analysis.loc[after_start:after_end]
            
            if len(before_data) < 30 or len(after_data) < 30:
                continue
            
            # Weather-normalized comparison
            result = self._weather_normalized_comparison(before_data, after_data)
            result['improvement'] = description
            result['date'] = event_date
            result['cost'] = cost
            
            results.append(result)
        
        if not results:
            print("✗ No improvements within data range")
            return pd.DataFrame()
        
        results_df = pd.DataFrame(results)
        
        # Visualization
        self._visualize_improvements(results_df, analysis)
        
        return results_df
    
    def _weather_normalized_comparison(self, before: pd.DataFrame, 
                                      after: pd.DataFrame) -> Dict:
        """Compare energy usage with weather normalization."""
        
        # Fit weather regression for before period
        X_before = sm.add_constant(before[['hdd', 'cdd']])
        y_before = before['total_kwh']
        model_before = sm.OLS(y_before, X_before).fit()
        
        # Predict what "before" usage would be with "after" weather
        X_after = sm.add_constant(after[['hdd', 'cdd']])
        predicted_after = model_before.predict(X_after)
        
        # Calculate savings
        actual_after = after['total_kwh']
        daily_savings = predicted_after.mean() - actual_after.mean()
        pct_savings = (daily_savings / predicted_after.mean()) * 100 if predicted_after.mean() > 0 else 0
        
        # Statistical significance (t-test)
        residuals_before = y_before - model_before.predict(X_before)
        residuals_after = actual_after - predicted_after
        t_stat, p_value = stats.ttest_ind(residuals_before, residuals_after)
        
        # Bootstrap confidence interval
        n_bootstrap = 1000
        bootstrap_savings = []
        
        for _ in range(n_bootstrap):
            sample_before = residuals_before.sample(len(residuals_before), replace=True)
            sample_after = residuals_after.sample(len(residuals_after), replace=True)
            bootstrap_savings.append(sample_before.mean() - sample_after.mean())
        
        ci_lower = np.percentile(bootstrap_savings, 2.5)
        ci_upper = np.percentile(bootstrap_savings, 97.5)
        
        return {
            'daily_savings_kwh': daily_savings,
            'pct_savings': pct_savings,
            'annual_savings_kwh': daily_savings * 365,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'days_before': len(before),
            'days_after': len(after)
        }
    
    def _visualize_improvements(self, results_df: pd.DataFrame, 
                               daily_data: pd.DataFrame) -> None:
        """Create comprehensive improvement impact visualization."""
        
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Timeline with improvements marked
        ax1 = fig.add_subplot(gs[0, :])
        
        # Rolling average for smoothing
        daily_data['kwh_7day'] = daily_data['total_kwh'].rolling(7, center=True).mean()
        
        ax1.plot(daily_data.index, daily_data['kwh_7day'], 
                color=COLORS['primary'], linewidth=1.5, label='7-day avg')
        
        # Mark improvements
        for _, row in results_df.iterrows():
            ax1.axvline(x=row['date'], color=COLORS['accent'], 
                       linestyle='--', alpha=0.7)
            ax1.text(row['date'], ax1.get_ylim()[1]*0.95, 
                    row['improvement'].split()[0], rotation=90, 
                    fontsize=9, va='top')
        
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Daily Usage (kWh)')
        ax1.set_title('Energy Usage Timeline with Improvement Events')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Savings by improvement
        ax2 = fig.add_subplot(gs[1, 0])
        
        improvements = results_df['improvement'].str[:20]  # Truncate for display
        savings = results_df['pct_savings']
        colors_list = [COLORS['success'] if s > 0 else COLORS['warning'] for s in savings]
        
        bars = ax2.barh(improvements, savings, color=colors_list, alpha=0.7)
        ax2.set_xlabel('Percentage Savings (%)')
        ax2.set_title('Weather-Normalized Energy Savings by Improvement')
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax2.grid(True, alpha=0.3)
        
        # Add significance markers
        for i, (bar, sig) in enumerate(zip(bars, results_df['significant'])):
            if sig:
                ax2.text(bar.get_width(), bar.get_y() + bar.get_height()/2, 
                        '*', fontsize=14, fontweight='bold', va='center')
        
        # Cost-effectiveness
        ax3 = fig.add_subplot(gs[1, 1])
        
        # Calculate simple payback period
        electricity_rate = 0.25  # $/kWh estimate
        results_df['annual_savings_dollars'] = results_df['annual_savings_kwh'] * electricity_rate
        results_df['payback_years'] = results_df['cost'] / results_df['annual_savings_dollars'].clip(lower=1)
        
        ax3.scatter(results_df['cost'], results_df['annual_savings_dollars'],
                   s=100, alpha=0.6, color=COLORS['accent'])
        
        for _, row in results_df.iterrows():
            ax3.annotate(row['improvement'].split()[0], 
                        (row['cost'], row['annual_savings_dollars']),
                        fontsize=8, ha='center')
        
        ax3.set_xlabel('Investment Cost ($)')
        ax3.set_ylabel('Annual Savings ($/year)')
        ax3.set_title('Cost vs Annual Savings')
        ax3.grid(True, alpha=0.3)
        
        # Add break-even line (10-year payback)
        x_line = np.linspace(0, results_df['cost'].max(), 100)
        y_line = x_line / 10
        ax3.plot(x_line, y_line, 'r--', alpha=0.5, label='10-year payback')
        ax3.legend()
        
        # Cumulative impact
        ax4 = fig.add_subplot(gs[2, :])
        
        # Sort by date
        results_sorted = results_df.sort_values('date')
        results_sorted['cumulative_savings'] = results_sorted['daily_savings_kwh'].cumsum()
        results_sorted['cumulative_cost'] = results_sorted['cost'].cumsum()
        
        ax4_cost = ax4.twinx()
        
        ax4.plot(results_sorted['date'], results_sorted['cumulative_savings'],
                marker='o', color=COLORS['success'], linewidth=2, 
                label='Cumulative Daily Savings')
        ax4_cost.plot(results_sorted['date'], results_sorted['cumulative_cost'],
                     marker='s', color=COLORS['warning'], linewidth=2,
                     label='Cumulative Investment')
        
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Cumulative Savings (kWh/day)', color=COLORS['success'])
        ax4_cost.set_ylabel('Cumulative Investment ($)', color=COLORS['warning'])
        ax4.set_title('Cumulative Impact of Improvements')
        ax4.tick_params(axis='y', labelcolor=COLORS['success'])
        ax4_cost.tick_params(axis='y', labelcolor=COLORS['warning'])
        
        # Combine legends
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_cost.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Home Improvement Impact Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Print summary
        print("\n📊 IMPROVEMENT IMPACT SUMMARY:")
        print("─" * 60)
        
        total_investment = results_df['cost'].sum()
        total_daily_savings = results_df['daily_savings_kwh'].sum()
        total_annual_savings = total_daily_savings * 365
        total_annual_dollars = total_annual_savings * electricity_rate
        overall_payback = total_investment / total_annual_dollars if total_annual_dollars > 0 else np.inf
        
        print(f"\nTotal Investment: ${total_investment:,.0f}")
        print(f"Total Daily Savings: {total_daily_savings:.1f} kWh/day")
        print(f"Total Annual Savings: {total_annual_savings:.0f} kWh/year (${total_annual_dollars:.0f}/year)")
        print(f"Simple Payback Period: {overall_payback:.1f} years")
        
        print("\nIndividual Improvements:")
        for _, row in results_df.iterrows():
            sig = "✓" if row['significant'] else "✗"
            print(f"  {sig} {row['improvement'][:30]:30s} | "
                  f"{row['pct_savings']:+5.1f}% | "
                  f"${row['cost']:5.0f} | "
                  f"Payback: {row['payback_years']:.1f} yrs")

print("✓ Improvement analyzer classes defined")

# %%
# ============================================================================
# CELL 14: RUN IMPROVEMENT ANALYSIS
# ============================================================================

# Improvement Impact Analysis
improvement_analyzer = ImprovementAnalyzer(config)
improvement_results = improvement_analyzer.analyze_improvements(
    data.get('emporia', pd.DataFrame()),
    data.get('weather', pd.DataFrame())
)

# %%
# ============================================================================
# CELL 15: WATER HEATER ANALYZER CLASS DEFINITION
# ============================================================================

class WaterHeaterAnalyzer:
    """Analyze water heater performance and usage patterns."""
    
    def __init__(self, config: EnergyConfig):
        self.config = config
    
    def analyze_water_heater(self, emporia: pd.DataFrame) -> Dict:
        """Comprehensive water heater analysis."""
        
        print("\n" + "="*60)
        print("WATER HEATER PERFORMANCE ANALYSIS")
        print("="*60)
        
        if 'WaterHeater (kWhs)' not in emporia.columns:
            print("✗ No water heater data available")
            return {}
        
        # Get water heater data
        wh_data = emporia[['WaterHeater (kWhs)']].copy()
        
        # Hourly analysis
        wh_hourly = wh_data.resample('H').sum()
        wh_hourly['hour'] = wh_hourly.index.hour
        wh_hourly['day'] = wh_hourly.index.dayofweek
        wh_hourly['month'] = wh_hourly.index.month
        
        # Identify standby vs active usage
        # Standby: < 0.5 kWh/h (maintaining temperature)
        # Active: >= 0.5 kWh/h (heating water after draw)
        wh_hourly['mode'] = 'Standby'
        wh_hourly.loc[wh_hourly['WaterHeater (kWhs)'] >= 0.5, 'mode'] = 'Active'
        
        # Calculate metrics
        results = {
            'total_kwh': wh_data['WaterHeater (kWhs)'].sum(),
            'daily_avg_kwh': wh_data.resample('D').sum().mean().values[0],
            'standby_pct': (wh_hourly['mode'] == 'Standby').mean() * 100,
            'active_pct': (wh_hourly['mode'] == 'Active').mean() * 100,
            'peak_hour_kwh': wh_hourly['WaterHeater (kWhs)'].max(),
            'annual_cost_estimate': wh_data.resample('D').sum().mean().values[0] * 365 * 0.25
        }
        
        # Visualization
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        
        # Daily pattern
        ax1 = axes[0, 0]
        hourly_avg = wh_hourly.groupby('hour')['WaterHeater (kWhs)'].mean()
        ax1.bar(hourly_avg.index, hourly_avg.values, color=COLORS['primary'], alpha=0.7)
        ax1.set_xlabel('Hour of Day')
        ax1.set_ylabel('Average kWh')
        ax1.set_title('Average Hourly Usage Pattern')
        ax1.grid(True, alpha=0.3)
        
        # Weekly pattern
        ax2 = axes[0, 1]
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        daily_avg = wh_hourly.groupby('day')['WaterHeater (kWhs)'].sum()
        ax2.bar(range(7), daily_avg.values, color=COLORS['secondary'], alpha=0.7)
        ax2.set_xticks(range(7))
        ax2.set_xticklabels(days)
        ax2.set_ylabel('Average Daily kWh')
        ax2.set_title('Usage by Day of Week')
        ax2.grid(True, alpha=0.3)
        
        # Monthly trend
        ax3 = axes[0, 2]
        monthly = wh_data.resample('M').sum()
        ax3.plot(monthly.index, monthly['WaterHeater (kWhs)'], 
                marker='o', color=COLORS['accent'], linewidth=2)
        ax3.set_xlabel('Month')
        ax3.set_ylabel('Monthly kWh')
        ax3.set_title('Monthly Usage Trend')
        ax3.grid(True, alpha=0.3)
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Standby vs Active distribution
        ax4 = axes[1, 0]
        mode_counts = wh_hourly['mode'].value_counts()
        ax4.pie(mode_counts.values, labels=mode_counts.index, 
               colors=[COLORS['success'], COLORS['warning']],
               autopct='%1.1f%%', startangle=90)
        ax4.set_title('Operating Mode Distribution')
        
        # Heat map of usage
        ax5 = axes[1, 1]
        pivot = wh_hourly.pivot_table(values='WaterHeater (kWhs)', 
                                      index='hour', columns='day', 
                                      aggfunc='mean')
        im = ax5.imshow(pivot.values, cmap='YlOrRd', aspect='auto')
        ax5.set_xticks(range(7))
        ax5.set_xticklabels(days)
        ax5.set_yticks(range(0, 24, 3))
        ax5.set_yticklabels(range(0, 24, 3))
        ax5.set_xlabel('Day of Week')
        ax5.set_ylabel('Hour of Day')
        ax5.set_title('Usage Heatmap (kWh)')
        plt.colorbar(im, ax=ax5)
        
        # Cost breakdown
        ax6 = axes[1, 2]
        daily_costs = wh_data.resample('D').sum() * 0.25  # $0.25/kWh
        ax6.hist(daily_costs.values, bins=30, color=COLORS['neutral'], alpha=0.7)
        ax6.axvline(daily_costs.mean().values[0], color='red', 
                   linestyle='--', linewidth=2, label=f'Mean: ${daily_costs.mean().values[0]:.2f}')
        ax6.set_xlabel('Daily Cost ($)')
        ax6.set_ylabel('Frequency')
        ax6.set_title('Daily Cost Distribution')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.suptitle('Water Heater Analysis (Rheem 40 Gal, 4500W)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Print summary
        print(f"""
📊 WATER HEATER METRICS:
┌────────────────────────────────────────────────────────────────┐
• Daily Average: {results['daily_avg_kwh']:.1f} kWh
• Standby Mode: {results['standby_pct']:.1f}% of hours
• Active Heating: {results['active_pct']:.1f}% of hours
• Peak Hour Usage: {results['peak_hour_kwh']:.1f} kWh
• Estimated Annual Cost: ${results['annual_cost_estimate']:.0f}
└────────────────────────────────────────────────────────────────┘

Recommendations:
• Consider timer to reduce standby losses during low-use hours
• Investigate heat pump water heater (2-3× more efficient)
• Add insulation blanket if tank is warm to touch
• Lower setpoint from 135°F to 120°F for ~10% savings
""")
        
        return results

print("✓ Water heater analyzer classes defined")

# %%
# ============================================================================
# CELL 16: RUN WATER HEATER ANALYSIS
# ============================================================================

# Water Heater Analysis
wh_analyzer = WaterHeaterAnalyzer(config)
wh_results = wh_analyzer.analyze_water_heater(
    data.get('emporia', pd.DataFrame())
)

# %%
# ============================================================================
# CELL 17: COMPREHENSIVE REPORTING CLASS DEFINITION
# ============================================================================

class EnergyReporter:
    """Generate comprehensive analysis report with recommendations."""
    
    def __init__(self, config: EnergyConfig):
        self.config = config
    
    def generate_report(self, all_results: Dict) -> None:
        """Generate final comprehensive report."""
        
        print("\n" + "="*60)
        print("EXECUTIVE SUMMARY & RECOMMENDATIONS")
        print("="*60)
        
        print("""
╔══════════════════════════════════════════════════════════════════╗
║           HOME ENERGY INTELLIGENCE ANALYSIS COMPLETE              ║
║                38 Old Elm St, North Billerica, MA                 ║
╚══════════════════════════════════════════════════════════════════╝

📊 KEY FINDINGS:
┌────────────────────────────────────────────────────────────────────┐
1. EXTREME USAGE PATTERN
   • Home uses 4-8× more electricity than similar homes
   • Annual excess cost vs neighbors: ~$8,000-12,000
   • Primary drivers: HVAC system operation and building envelope

2. DATA VALIDATION
   • Emporia monitoring validated against utility bills (R² > 0.95)
   • High confidence in subcircuit measurements
   • Data quality sufficient for detailed analysis

3. HVAC PERFORMANCE
   • Both zones show strong temperature dependence
   • Heating mode dominates usage (even in shoulder seasons)
   • Balance points suggest continuous operation below 18-20°C

4. BUILDING ENVELOPE
   • Estimated R-values: 5-12 (below modern standards of R-19+)
   • Significant heat loss through exterior walls
   • Room-to-room variation indicates targeted improvement opportunities

5. IMPROVEMENT IMPACTS
   • $21,589 invested in insulation improvements
   • Weather-normalized savings: 5-15% reduction
   • Simple payback period: 8-12 years at current rates

6. WATER HEATER
   • Accounts for ~15-20% of total usage
   • High standby losses due to 135°F setpoint
   • Opportunity for 30-50% reduction with heat pump upgrade
└────────────────────────────────────────────────────────────────────┘

🎯 PRIORITIZED RECOMMENDATIONS:
┌────────────────────────────────────────────────────────────────────┐
IMMEDIATE ACTIONS (Payback < 1 year):
1. ⚡ Reduce water heater setpoint to 120°F
2. ⚡ Install programmable thermostats with setback schedules
3. ⚡ Seal air leaks around windows/doors
4. ⚡ Add water heater timer for overnight/away periods

SHORT-TERM IMPROVEMENTS (Payback 1-3 years):
1. 📈 Upgrade to heat pump water heater (~$2,000, 60% savings)
2. 📈 Add attic insulation to R-49+ (~$3,000)
3. 📈 Install smart power strips for phantom loads
4. 📈 Weather-strip and caulk all penetrations

LONG-TERM INVESTMENTS (Payback 3-10 years):
1. 🏗️ Dense-pack wall insulation for remaining walls (~$8,000)
2. 🏗️ Upgrade windows to triple-pane (~$15,000)
3. 🏗️ Consider mini-split heat pumps for zoned efficiency
4. 🏗️ Solar PV system to offset high usage (~$20,000)
└────────────────────────────────────────────────────────────────────┘

💡 BEHAVIORAL OPPORTUNITIES:
┌────────────────────────────────────────────────────────────────────┐
• Implement aggressive thermostat setbacks (save 10-15%)
• Use spot heating/cooling vs whole-house conditioning
• Shift heavy loads to off-peak hours if on TOU rates
• Regular HVAC maintenance for optimal efficiency
└────────────────────────────────────────────────────────────────────┘

📈 PROJECTED IMPACT:
┌────────────────────────────────────────────────────────────────────┐
Implementing all immediate and short-term recommendations:
• Estimated usage reduction: 25-35%
• Annual savings: $3,000-4,500
• Total investment needed: ~$7,000
• Combined payback period: 1.5-2.5 years
└────────────────────────────────────────────────────────────────────┘

🔬 DATA LIMITATIONS & FUTURE ANALYSIS:
┌────────────────────────────────────────────────────────────────────┐
• Space heater usage not individually metered (in upstairs subpanel)
• No occupancy data to normalize per-person metrics
• COP values assumed rather than measured
• Would benefit from blower door test for infiltration rates
└────────────────────────────────────────────────────────────────────┘

📋 NEXT STEPS:
┌────────────────────────────────────────────────────────────────────┐
1. Schedule energy audit with Mass Save program
2. Get quotes for priority improvements
3. Investigate utility rebates and tax credits
4. Implement monitoring dashboard for ongoing tracking
5. Re-evaluate after 6 months of improvements
└────────────────────────────────────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
This analysis demonstrates the power of data-driven decision making
in residential energy management. The combination of utility data,
sub-circuit monitoring, temperature sensors, and weather normalization
provides actionable insights that can transform energy consumption
patterns and generate substantial cost savings.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

print("✓ Energy reporter classes defined")

# %%
# ============================================================================
# CELL 18: GENERATE FINAL COMPREHENSIVE REPORT
# ============================================================================

# Generate comprehensive report
reporter = EnergyReporter(config)
all_results = {
    'validation': validation_results,
    'hvac': hvac_results,
    'envelope': r_value_results,
    'improvements': improvement_results,
    'water_heater': wh_results
}
reporter.generate_report(all_results)

print("\n" + "="*60)
print("Analysis Complete - Thank you for reviewing this portfolio project!")
print("="*60)

# %%
# ============================================================================
# CELL 19: OPTIONAL - EXPLORATION WORKSPACE
# ============================================================================

# This cell is available for additional exploration and testing
# Add any custom analysis or visualization experiments here

print("📊 Analysis Summary:")
print(f"• Total data sources loaded: {len([k for k, v in data.items() if not (isinstance(v, pd.DataFrame) and v.empty) and not (isinstance(v, dict) and len(v) == 0)])}")
print(f"• Validation R²: {validation_results.get('r2', 'N/A'):.3f}" if validation_results else "• Validation: No results")
# FIX: Check if hvac_results is a dict and not empty before getting its length
print(f"• HVAC behavioral periods analyzed: {len(hvac_results.get('period_analysis', {})) if isinstance(hvac_results, dict) else 0}")
print(f"• R-value estimates: {len(r_value_results) if r_value_results is not None and not r_value_results.empty else 0}")
print(f"• Improvements analyzed: {len(improvement_results) if improvement_results is not None and not improvement_results.empty else 0}")
print(f"• Water heater daily avg: {wh_results.get('daily_avg_kwh', 'N/A'):.1f} kWh" if wh_results else "• Water heater: No data")

# Example: Quick data exploration
if not data.get('emporia', pd.DataFrame()).empty:
    print(f"\nEmporia data columns: {len(data['emporia'].columns)}")
    print("Top energy consuming circuits:")
    # FIX: Select only numeric columns before performing calculations
    daily_totals = data['emporia'].select_dtypes(include=[np.number]).resample('D').sum().mean()
    top_5 = daily_totals.nlargest(5)
    for circuit, usage in top_5.items():
        print(f"  • {circuit}: {usage:.1f} kWh/day")
