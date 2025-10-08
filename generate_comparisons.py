#!/usr/bin/env python3
"""
generate_comparisons.py - Generate line graphs from Huffman gridsearch results

This script analyzes gridsearch output directories and generates line graphs depicting 
the relationship between each variable and final reconstruction error. All other 
variables are held constant for each graph.

Usage:
    python generate_comparisons.py --input huffman_outputs --output comparison_plots
    python generate_comparisons.py --status-file grid_search_status_20250930_220809.json
    python generate_comparisons.py --help

The script can work with either:
1. A gridsearch status JSON file (recommended)
2. A directory containing gridsearch results

For each combination of constant variables, it generates line graphs showing how 
changing each varying parameter affects reconstruction error.
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
from datetime import datetime
import warnings

# Set style for better-looking plots
try:
    plt.style.use('seaborn-v0_8')
except:
    try:
        plt.style.use('seaborn')
    except:
        pass  # Use default style if seaborn not available

warnings.filterwarnings('ignore')


class GridSearchAnalyzer:
    """Analyzes gridsearch results and generates comparison plots."""
    
    def __init__(self, output_dir: str = "comparison_plots"):
        """
        Initialize the analyzer.
        
        Args:
            output_dir: Directory to save generated plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_data = []
        
        # Define the variables we can analyze
        self.numeric_variables = [
            'bin_size',
            'target_bits',
            'original_bits'
        ]
        
        self.categorical_variables = [
            'heuristic_metric',
            'heuristic_algo', 
            'lsa_metric',
            'lsa_prep',
            'heuristic',
            'lsa',
            'strategy'
        ]
        
        self.grouping_variables = [
            'model_name',
            'layer_name'
        ]
        
        # Metrics to analyze
        self.metrics = [
            'final_reconstruction_error',
            'percent_reconstruction_error_increase',
            'entropy',
            'average_huffman_length',
            'bin_violations_percentage'
        ]
    
    def load_from_status_file(self, status_file: str) -> None:
        """
        Load results from a gridsearch status JSON file.
        
        Args:
            status_file: Path to the gridsearch status JSON file
        """
        print(f"Loading results from status file: {status_file}")
        
        with open(status_file, 'r') as f:
            status_data = json.load(f)
        
        total_combinations = len(status_data.get('combinations', []))
        completed_count = 0
        failed_count = 0
        
        for combo in status_data.get('combinations', []):
            if combo.get('status') == 'completed' and 'result' in combo:
            # if 'result' in combo:
                try:
                    result_data = self._extract_result_data(combo['config'], combo['result'])
                    if result_data:
                        self.results_data.append(result_data)
                        completed_count += 1
                except Exception as e:
                    print(f"Warning: Failed to process combination {combo.get('experiment_id', 'unknown')}: {e}")
                    failed_count += 1
            elif combo.get('status') == 'failed':
                failed_count += 1
        
        print(f"Loaded {completed_count} completed results out of {total_combinations} total combinations")
        if failed_count > 0:
            print(f"Found {failed_count} failed experiments")
    
    def load_from_directory(self, results_dir: str) -> None:
        """
        Load results by scanning a directory structure.
        
        Args:
            results_dir: Base directory containing gridsearch results
        """
        print(f"Scanning directory for results: {results_dir}")
        
        results_path = Path(results_dir)
        if not results_path.exists():
            raise ValueError(f"Results directory does not exist: {results_dir}")
        
        # Find all metrics.json files
        metrics_files = list(results_path.rglob("metrics.json"))
        print(f"Found {len(metrics_files)} metrics files")
        
        loaded_count = 0
        for metrics_file in metrics_files:
            try:
                # Load metrics
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                
                # Load config (should be in same directory)
                config_file = metrics_file.parent / "config.json"
                if config_file.exists():
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                    
                    result_data = self._extract_result_data(config, metrics)
                    if result_data:
                        self.results_data.append(result_data)
                        loaded_count += 1
                else:
                    print(f"Warning: No config.json found for {metrics_file}")
                    
            except Exception as e:
                print(f"Warning: Failed to process {metrics_file}: {e}")
        
        print(f"Successfully loaded {loaded_count} results")
    
    def _extract_result_data(self, config: Dict, result: Dict) -> Optional[Dict]:
        """
        Extract relevant data from config and result dictionaries.
        
        Args:
            config: Configuration dictionary
            result: Result dictionary (either from status file or metrics.json)
            
        Returns:
            Dictionary with extracted data or None if extraction fails
        """
        try:
            # Handle different result formats
            if 'quantization_metrics' in result:
                # Direct metrics format
                metrics = result
            elif 'result' in result:
                # Nested result format
                metrics = result['result']
            else:
                # Assume result is already the metrics
                metrics = result
            
            # Extract basic info
            data = {
                'model_name': config['model']['name'],
                'layer_name': config['model']['layer_name'],
                'bin_size': config['huffman']['bin_size'],
                'compression_severity': config['compression']['severity'],
                'strategy': config['huffman']['strategy'],
                'heuristic_metric': config['huffman']['heuristic_metric'],
                'heuristic_algo': config['huffman']['heuristic_algo'],
                'lsa_metric': config['huffman']['lsa_metric'],
                'lsa_prep': config['huffman']['lsa_prep'],
                'heuristic': config['huffman']['heuristic'],
                'lsa': config['huffman']['lsa']
            }
            
            # Calculate derived values
            if 'quantization_metrics' in metrics and 'vq_config' in metrics['quantization_metrics']:
                original_bits = metrics['quantization_metrics']['vq_config']['kwargs']['n_bits']
                data['original_bits'] = original_bits
                data['target_bits'] = original_bits - data['compression_severity']
            else:
                # Default values if not available
                data['original_bits'] = 4
                data['target_bits'] = 4 - data['compression_severity']
            
            # Extract initial reconstruction error
            if 'quantization_metrics' in metrics and 'reconstruction_error' in metrics['quantization_metrics']:
                data['initial_reconstruction_error'] = metrics['quantization_metrics']['reconstruction_error']
            
            # Extract metrics
            if 'final_reconstruction_error' in metrics:
                data['final_reconstruction_error'] = metrics['final_reconstruction_error']
            
            if 'reconstruction_error_increase' in metrics:
                data['reconstruction_error_increase'] = metrics['reconstruction_error_increase']
                # Calculate percent reconstruction error increase
                if 'initial_reconstruction_error' in data and data['initial_reconstruction_error'] > 0:
                    data['percent_reconstruction_error_increase'] = (data['reconstruction_error_increase'] / data['initial_reconstruction_error']) * 100
            
            if 'entropy' in metrics:
                data['entropy'] = metrics['entropy']
            
            # Extract violation stats
            if 'violation_stats' in metrics:
                vs = metrics['violation_stats']
                data['average_huffman_length'] = vs.get('average_huffman_length', 0)
                data['bin_violations_percentage'] = vs.get('bin_violations_percentage', 0)
            
            return data
            
        except Exception as e:
            print(f"Error extracting result data: {e}")
            return None
    
    def generate_all_comparisons(self, min_points: int = 3) -> None:
        """
        Generate all possible comparison plots.
        
        Args:
            min_points: Minimum number of data points required to generate a plot
        """
        if not self.results_data:
            print("No results data loaded. Please load data first.")
            return
        
        print(f"Generating comparison plots from {len(self.results_data)} results...")
        
        # Check which metrics are available
        available_metrics = []
        for metric in self.metrics:
            if any(metric in result for result in self.results_data):
                available_metrics.append(metric)
            else:
                print(f"Skipping metric {metric} - not found in data")
        
        # Generate plots for each metric
        for metric in available_metrics:
            print(f"\nGenerating plots for metric: {metric}")
            
            # Generate plots for each numeric variable
            for variable in self.numeric_variables:
                if any(variable in result for result in self.results_data):
                    self._generate_variable_plots(variable, metric, min_points)
            
            # Generate plots for categorical variables
            for variable in self.categorical_variables:
                if any(variable in result for result in self.results_data):
                    self._generate_categorical_plots(variable, metric, min_points)
    
    def _generate_variable_plots(self, variable: str, metric: str, min_points: int) -> None:
        """
        Generate line plots for a numeric variable vs metric.
        
        Args:
            variable: Variable to plot on x-axis
            metric: Metric to plot on y-axis
            min_points: Minimum points required for a plot
        """
        # Group data by all other variables except the one we're varying
        groups = self._group_data_by_constants(variable, metric)
        
        plot_count = 0
        for group_key, group_data in groups.items():
            if len(group_data) < min_points:
                continue
            
            # Check if we have variation in the variable
            variable_values = [item[variable] for item in group_data if variable in item]
            if len(set(variable_values)) < 2:
                continue
            
            # Determine if we need log scale plots
            needs_log_scale = (metric in ['final_reconstruction_error', 'percent_reconstruction_error_increase'])
            needs_log_x = (variable == 'bin_size')
            
            # Create linear scale plot (unless x-axis is bin_size, then use log x)
            if needs_log_x:
                # bin_size should always be log on x-axis
                self._create_single_line_plot(group_data, variable, metric, group_key, min_points, 
                                            log_x=True, log_y=False)
                plot_count += 1
                
                # If metric also needs log scale, create log-log plot
                if needs_log_scale:
                    self._create_single_line_plot(group_data, variable, metric, group_key, min_points, 
                                                log_x=True, log_y=True)
                    plot_count += 1
            else:
                # Create linear scale plot
                self._create_single_line_plot(group_data, variable, metric, group_key, min_points, 
                                            log_x=False, log_y=False)
                plot_count += 1
                
                # Create log scale plots if needed
                if needs_log_scale:
                    # Only y-axis log scale
                    self._create_single_line_plot(group_data, variable, metric, group_key, min_points, 
                                                log_x=False, log_y=True)
                    plot_count += 1
        
        print(f"  Generated {plot_count} plots for {variable} vs {metric}")
    
    def _group_data_by_constants(self, varying_variable: str, metric: str) -> Dict[str, List[Dict]]:
        """
        Group data by constant variables, varying only the specified variable.
        
        Args:
            varying_variable: The variable that should vary within each group
            metric: The metric being analyzed
            
        Returns:
            Dictionary mapping group keys to lists of data points
        """
        groups = defaultdict(list)
        
        for result in self.results_data:
            if varying_variable not in result or metric not in result:
                continue
            
            # Create group key from all variables except the varying one and metrics
            group_parts = []
            for key, value in result.items():
                if (key != varying_variable and 
                    key not in self.metrics and 
                    key in self.grouping_variables + self.categorical_variables + self.numeric_variables):
                    group_parts.append(f"{key}={value}")
            
            group_key = "|".join(sorted(group_parts))
            groups[group_key].append(result)
        
        return dict(groups)
    
    def _generate_categorical_plots(self, variable: str, metric: str, min_points: int) -> None:
        """
        Generate bar plots for categorical variables vs metric.
        
        Args:
            variable: Categorical variable
            metric: Metric to plot
            min_points: Minimum points required for a plot
        """
        # Group data by all other variables except the one we're varying
        groups = self._group_data_by_constants(variable, metric)
        
        plot_count = 0
        for group_key, group_data in groups.items():
            if len(group_data) < min_points:
                continue
            
            # Check if we have variation in the variable
            variable_values = [item[variable] for item in group_data if variable in item]
            if len(set(variable_values)) < 2:
                continue
            
            self._create_single_bar_plot(group_data, variable, metric, group_key, min_points)
            plot_count += 1
        
        print(f"  Generated {plot_count} plots for {variable} vs {metric}")
    
    def _create_single_line_plot(self, data: List[Dict], variable: str, metric: str, 
                                group_key: str, min_points: int, log_x: bool = False, log_y: bool = False) -> None:
        """Create a single line plot."""
        if len(data) < min_points:
            return
        
        # Extract x and y values
        points = [(item[variable], item[metric]) for item in data 
                 if variable in item and metric in item]
        
        if len(points) < min_points:
            return
        
        # Sort by variable for proper line plotting
        points.sort(key=lambda x: x[0])
        
        x_values = [p[0] for p in points]
        y_values = [p[1] for p in points]
        
        # Create plot
        plt.figure(figsize=(10, 6))
        
        # Check if we have multiple y-values for the same x-value
        x_unique = list(set(x_values))
        if len(points) > len(x_unique):
            # Multiple points per x-value, aggregate by taking mean and std
            x_means = []
            y_means = []
            y_stds = []
            
            for x_val in sorted(x_unique):
                y_vals_for_x = [p[1] for p in points if p[0] == x_val]
                x_means.append(x_val)
                y_means.append(np.mean(y_vals_for_x))
                y_stds.append(np.std(y_vals_for_x) if len(y_vals_for_x) > 1 else 0)
            
            plt.errorbar(x_means, y_means, yerr=y_stds, marker='o', capsize=5, linewidth=2, markersize=6)
        else:
            # Simple line plot
            plt.plot(x_values, y_values, marker='o', linewidth=2, markersize=6)
        
        # Add horizontal line for initial reconstruction error if plotting final reconstruction error
        if metric == 'final_reconstruction_error':
            initial_errors = [item['initial_reconstruction_error'] for item in data 
                            if 'initial_reconstruction_error' in item]
            if initial_errors:
                avg_initial_error = np.mean(initial_errors)
                plt.axhline(y=avg_initial_error, color='red', linestyle='--', alpha=0.7, 
                           label=f'Initial Error: {avg_initial_error:.2e}')
                plt.legend()
        
        # Add vertical line for entropy if plotting against target_bits
        if variable == 'target_bits':
            entropy_values = [item['entropy'] for item in data if 'entropy' in item]
            if entropy_values:
                avg_entropy = np.mean(entropy_values)
                plt.axvline(x=avg_entropy, color='green', linestyle='--', alpha=0.7, 
                           label=f'Entropy: {avg_entropy:.2f}')
                plt.legend()
        
        # Set log scales
        if log_x:
            plt.xscale('log')
        if log_y:
            plt.yscale('log')
        
        # Formatting
        plt.xlabel(self._format_variable_name(variable))
        plt.ylabel(self._format_metric_name(metric))
        
        # Create title from group key
        title_parts = self._parse_group_key(group_key)
        scale_suffix = ""
        if log_x and log_y:
            scale_suffix = " (Log-Log Scale)"
        elif log_x:
            scale_suffix = " (Log X Scale)"
        elif log_y:
            scale_suffix = " (Log Y Scale)"
        
        if title_parts:
            plt.title(f"{self._format_metric_name(metric)} vs {self._format_variable_name(variable)}{scale_suffix}\n" + 
                     ", ".join(title_parts))
        else:
            plt.title(f"{self._format_metric_name(metric)} vs {self._format_variable_name(variable)}{scale_suffix}")
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Create subdirectory structure: plots/y_axis_metric/x_axis_variable/
        metric_dir = self.output_dir / metric
        variable_dir = metric_dir / variable
        variable_dir.mkdir(parents=True, exist_ok=True)
        
        # Save plot
        filename = self._generate_filename(variable, metric, group_key, "line", log_x, log_y)
        filepath = variable_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_single_bar_plot(self, data: List[Dict], variable: str, metric: str, 
                               group_key: str, min_points: int) -> None:
        """Create a single bar plot for categorical variables."""
        if len(data) < min_points:
            return
        
        # Aggregate data by categorical variable
        category_data = defaultdict(list)
        for item in data:
            if variable in item and metric in item:
                category_data[item[variable]].append(item[metric])
        
        if not category_data:
            return
        
        # Calculate statistics for each category
        categories = sorted(category_data.keys())
        means = []
        stds = []
        counts = []
        
        for cat in categories:
            values = category_data[cat]
            means.append(np.mean(values))
            stds.append(np.std(values) if len(values) > 1 else 0)
            counts.append(len(values))
        
        # Create plot
        plt.figure(figsize=(10, 6))
        
        bars = plt.bar(range(len(categories)), means, yerr=stds, capsize=5, alpha=0.7)
        
        # Add value labels on bars
        for i, (bar, count) in enumerate(zip(bars, counts)):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'n={count}', ha='center', va='bottom', fontsize=8)
        
        plt.xlabel(self._format_variable_name(variable))
        plt.ylabel(self._format_metric_name(metric))
        plt.xticks(range(len(categories)), categories, rotation=45)
        
        # Create title from group key
        title_parts = self._parse_group_key(group_key)
        if title_parts:
            plt.title(f"{self._format_metric_name(metric)} vs {self._format_variable_name(variable)}\n" + 
                     ", ".join(title_parts))
        else:
            plt.title(f"{self._format_metric_name(metric)} vs {self._format_variable_name(variable)}")
        
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        # Create subdirectory structure: plots/y_axis_metric/x_axis_variable/
        metric_dir = self.output_dir / metric
        variable_dir = metric_dir / variable
        variable_dir.mkdir(parents=True, exist_ok=True)
        
        # Save plot
        filename = self._generate_filename(variable, metric, group_key, "bar")
        filepath = variable_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _parse_group_key(self, group_key: str) -> List[str]:
        """Parse group key into readable title parts."""
        if not group_key:
            return []
        
        parts = group_key.split("|")
        title_parts = []
        
        for part in parts:
            if "=" in part:
                key, value = part.split("=", 1)
                # Include all non-metric variables in the title for clarity
                if key in self.grouping_variables + self.categorical_variables + self.numeric_variables:
                    title_parts.append(f"{self._format_variable_name(key)}: {value}")
        
        return title_parts
    
    def _format_variable_name(self, variable: str) -> str:
        """Format variable names for display."""
        formatting = {
            'bin_size': 'Bin Size',
            'compression_severity': 'Compression Severity',
            'target_bits': 'Target Bits',
            'original_bits': 'Original Bits',
            'heuristic_metric': 'Heuristic Metric',
            'heuristic_algo': 'Heuristic Algorithm',
            'lsa_metric': 'LSA Metric',
            'lsa_prep': 'LSA Preparation',
            'heuristic': 'Use Heuristic',
            'lsa': 'Use LSA',
            'strategy': 'Huffman Strategy',
            'model_name': 'Model',
            'layer_name': 'Layer'
        }
        return formatting.get(variable, variable.replace('_', ' ').title())
    
    def _format_metric_name(self, metric: str) -> str:
        """Format metric names for display."""
        formatting = {
            'final_reconstruction_error': 'Final Reconstruction Error',
            'percent_reconstruction_error_increase': 'Percent Reconstruction Error Increase',
            'entropy': 'Entropy',
            'average_huffman_length': 'Average Huffman Length',
            'bin_violations_percentage': 'Bin Violations (%)'
        }
        return formatting.get(metric, metric.replace('_', ' ').title())
    
    def _generate_filename(self, variable: str, metric: str, group_key: str, plot_type: str, 
                           log_x: bool = False, log_y: bool = False) -> str:
        """Generate a filename for the plot."""
        # Start with metric and variable
        filename_parts = [metric, "vs", variable]
        
        # Add scale suffix
        if log_x and log_y:
            filename_parts.append("log_log")
        elif log_x:
            filename_parts.append("log_x")
        elif log_y:
            filename_parts.append("log_y")
        
        # Add only essential grouping information to keep filename reasonable
        if group_key:
            parts = group_key.split("|")
            essential_vars = ['model_name', 'layer_name', 'target_bits', 'bin_size']  # Only include key variables
            
            for part in parts:
                if "=" in part:
                    key, value = part.split("=", 1)
                    # Only include essential variables and only if they're not the varying variable
                    if key in essential_vars and key != variable:
                        # Clean up the value for filename
                        clean_value = str(value).replace('/', '_').replace(' ', '_').replace('.', '_')
                        filename_parts.append(f"{key}_{clean_value}")
        
        filename = "_".join(filename_parts) + f"_{plot_type}.png"
        
        # Clean up filename
        filename = filename.replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'pct')
        
        # Truncate if still too long
        if len(filename) > 200:
            filename = filename[:200] + f"_{plot_type}.png"
        
        return filename
    
    def generate_summary_report(self) -> None:
        """Generate a summary report of the analysis."""
        if not self.results_data:
            print("No results data loaded.")
            return
        
        # Create summary report
        report_path = self.output_dir / "analysis_summary.txt"
        
        with open(report_path, 'w') as f:
            f.write("Huffman Gridsearch Analysis Summary\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total experiments analyzed: {len(self.results_data)}\n\n")
            
            # Data overview
            f.write("Data Overview:\n")
            f.write("-" * 15 + "\n")
            
            all_variables = self.numeric_variables + self.categorical_variables + self.grouping_variables
            for var in all_variables:
                values = [result[var] for result in self.results_data if var in result]
                if values:
                    unique_vals = len(set(values))
                    f.write(f"{self._format_variable_name(var)}: {unique_vals} unique values\n")
            
            f.write("\nMetrics Available:\n")
            f.write("-" * 18 + "\n")
            for metric in self.metrics:
                values = [result[metric] for result in self.results_data if metric in result]
                if values:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    f.write(f"{self._format_metric_name(metric)}: {mean_val:.6f} ± {std_val:.6f}\n")
            
            # Variable ranges
            f.write("\nVariable Ranges:\n")
            f.write("-" * 16 + "\n")
            for var in self.numeric_variables:
                values = [result[var] for result in self.results_data if var in result]
                if values:
                    min_val = min(values)
                    max_val = max(values)
                    f.write(f"{self._format_variable_name(var)}: {min_val} to {max_val}\n")
        
        print(f"Summary report saved to: {report_path}")


def main():
    """Main function to run the comparison generator."""
    parser = argparse.ArgumentParser(
        description="Generate comparison plots from Huffman gridsearch results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use gridsearch status file (recommended)
  python generate_comparisons.py --status-file grid_search_status_20250930_220809.json
  
  # Scan directory for results
  python generate_comparisons.py --input huffman_outputs --output comparison_plots
  
  # Customize minimum points and output directory
  python generate_comparisons.py --status-file status.json --output my_plots --min-points 5
        """
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--status-file", 
        type=str,
        help="Path to gridsearch status JSON file"
    )
    input_group.add_argument(
        "--input", 
        type=str,
        help="Directory containing gridsearch results"
    )
    
    # Output options
    parser.add_argument(
        "--output", 
        type=str, 
        default="comparison_plots",
        help="Output directory for generated plots (default: comparison_plots)"
    )
    
    parser.add_argument(
        "--min-points", 
        type=int, 
        default=3,
        help="Minimum number of data points required to generate a plot (default: 3)"
    )
    
    parser.add_argument(
        "--metrics",
        nargs='+',
        default=None,
        help="Specific metrics to analyze (default: all available)"
    )
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = GridSearchAnalyzer(args.output)
    
    # Load data
    try:
        if args.status_file:
            analyzer.load_from_status_file(args.status_file)
        else:
            analyzer.load_from_directory(args.input)
        
        if not analyzer.results_data:
            print("No valid results found. Please check your input.")
            return
        
        # Filter metrics if specified
        if args.metrics:
            analyzer.metrics = [m for m in args.metrics if m in analyzer.metrics]
            if not analyzer.metrics:
                print("No valid metrics specified. Available metrics:")
                print("  " + "\n  ".join(GridSearchAnalyzer().metrics))
                return
        
        # Generate plots
        analyzer.generate_all_comparisons(args.min_points)
        
        # Generate summary report
        analyzer.generate_summary_report()
        
        print(f"\nAnalysis complete! Plots saved to: {analyzer.output_dir}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
