import sys
import os
import re
import numpy as np
import pandas as pd
from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                               QWidget, QPushButton, QLabel, QLineEdit, QComboBox,
                               QTextEdit, QSplitter, QGroupBox, QCheckBox, QSlider,
                               QFileDialog, QMessageBox, QTabWidget, QSpinBox,
                               QDoubleSpinBox, QColorDialog, QToolBar, QGridLayout,
                               QScrollArea, QFrame, QButtonGroup, QRadioButton)
from PySide6.QtCore import Qt, QMimeData, Signal, QThread, QTimer
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QPixmap, QColor, QAction, QFont
from PySide6.QtWebEngineWidgets import QWebEngineView

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.figure_factory as ff
import plotly.io as pio


class DataParser:
    """Advanced parser for various scientific data formats"""
    
    @staticmethod
    def detect_delimiter(file_path, sample_lines=10):
        """Auto-detect delimiter in data file using statistical analysis"""
        delimiters = ['\t', ',', ';', ' ', '|']
        delimiter_scores = {}
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = [f.readline().strip() for _ in range(sample_lines)]
        
        for delimiter in delimiters:
            scores = []
            for line in lines:
                if line and not line.startswith('#'):
                    parts = [p.strip() for p in line.split(delimiter) if p.strip()]
                    # Score based on number of numeric values
                    numeric_count = sum(1 for p in parts if DataParser._is_numeric(p))
                    if numeric_count >= 2:
                        scores.append(numeric_count)
            
            if scores:
                delimiter_scores[delimiter] = (np.mean(scores), np.std(scores), len(scores))
        
        if not delimiter_scores:
            return '\t'  # Default fallback
        
        # Choose delimiter with highest mean numeric columns and consistency
        best_delimiter = max(delimiter_scores.keys(), 
                           key=lambda x: delimiter_scores[x][0] * delimiter_scores[x][2] / (delimiter_scores[x][1] + 1))
        return best_delimiter
    
    @staticmethod
    def _is_numeric(value):
        """Check if a string represents a numeric value"""
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def parse_metadata(file_path):
        """Extract metadata from file headers with improved parsing"""
        metadata = {}
        data_start_line = 0
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Check if line contains numeric data
            if DataParser._line_is_data(line):
                data_start_line = i
                break
            else:
                # Parse metadata
                if ':' in line:
                    key, value = line.split(':', 1)
                    metadata[key.strip()] = value.strip()
                elif '=' in line:
                    key, value = line.split('=', 1)
                    metadata[key.strip()] = value.strip()
                else:
                    metadata[f'header_{i}'] = line
        
        return metadata, data_start_line
    
    @staticmethod
    def _line_is_data(line):
        """Determine if a line contains numeric data"""
        parts = re.split(r'[\s,;\t|]+', line)
        numeric_parts = [p for p in parts if p and DataParser._is_numeric(p)]
        return len(numeric_parts) >= 2
    
    @staticmethod
    def load_data(file_path):
        """Load data with automatic format detection and error handling"""
        try:
            # Parse metadata
            metadata, data_start = DataParser.parse_metadata(file_path)
            
            # Detect delimiter
            delimiter = DataParser.detect_delimiter(file_path)
            
            # Load data with robust error handling
            data = pd.read_csv(file_path, delimiter=delimiter, skiprows=data_start, 
                             header=None, comment='#', skip_blank_lines=True,
                             on_bad_lines='skip', engine='python')
            
            # Clean data - remove non-numeric columns
            numeric_columns = []
            for col in data.columns:
                if pd.api.types.is_numeric_dtype(data[col]) or data[col].apply(DataParser._is_numeric).all():
                    numeric_columns.append(col)
            
            data = data[numeric_columns]
            
            # Convert to numeric, handling errors
            for col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            
            # Remove rows with all NaN values
            data = data.dropna(how='all')
            
            return data, metadata
            
        except Exception as e:
            raise Exception(f"Failed to load data: {str(e)}")


class DataNormalizer:
    """Data normalization utilities for spectroscopic data"""
    
    @staticmethod
    def min_max_normalize(data, feature_range=(0, 1)):
        """Min-max normalization"""
        min_val, max_val = feature_range
        data_min = data.min()
        data_max = data.max()
        return (data - data_min) / (data_max - data_min) * (max_val - min_val) + min_val
    
    @staticmethod
    def z_score_normalize(data):
        """Z-score standardization"""
        return (data - data.mean()) / data.std()
    
    @staticmethod
    def area_normalize(x, y):
        """Area normalization using trapezoidal integration"""
        area = np.trapz(y, x)
        return y / area if area != 0 else y
    
    @staticmethod
    def baseline_correct(x, y, degree=1):
        """Simple polynomial baseline correction"""
        # Fit polynomial to data
        coeffs = np.polyfit(x, y, degree)
        baseline = np.polyval(coeffs, x)
        return y - baseline
    
    @staticmethod
    def smooth_data(data, window_size=5):
        """Simple moving average smoothing"""
        return pd.Series(data).rolling(window=window_size, center=True).mean().fillna(data)
    
    @staticmethod
    def peak_normalize(y):
        """Normalize to peak intensity"""
        return y / y.max() if y.max() != 0 else y
    
    @staticmethod
    def vector_normalize(y):
        """Vector (L2) normalization"""
        norm = np.linalg.norm(y)
        return y / norm if norm != 0 else y


class DatasetItem:
    """Class to manage individual dataset properties"""
    def __init__(self, name, data, metadata, file_path):
        self.name = name
        self.data = data
        self.metadata = metadata
        self.file_path = file_path
        self.visible = True
        self.x_column = 0
        self.y_column = 1
        self.y_axis = 'left'  # 'left' or 'right'
        self.color = 'blue'
        self.line_style = 'solid'
        self.line_width = 2
        self.marker_style = 'none'
        self.marker_size = 6
        self.normalization = 'None'
        self.opacity = 0.8
        self.fill_area = False
        self.error_bars = False
        self.error_column = None


class UnifiedPlotlyCanvas(QWebEngineView):
    """Unified Plotly canvas for all visualization types"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.datasets = {}
        self.current_plot_type = "line_plot"
        self.setMinimumSize(800, 600)
        
        self.plot_config = {
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToAdd': [
                'drawline', 'drawopenpath', 'drawclosedpath',
                'drawcircle', 'drawrect', 'eraseshape'
            ],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'scientific_plot',
                'height': 800,
                'width': 1200,
                'scale': 2
            },
            'responsive': True,
            'doubleClick': 'reset+autosize'
        }
        
        # Initialize empty plot
        self.create_empty_plot()
    
    def create_empty_plot(self):
        """Create an empty plot with proper layout"""
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.update_layout(
            template='plotly_white',
            title=dict(text="Scientific Data Visualizer", x=0.5, font_size=16),
            xaxis=dict(
                title="X-axis",
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                zeroline=True
            ),
            yaxis=dict(
                title="Y-axis",
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                zeroline=True
            ),
            showlegend=True,
            legend=dict(x=0.02, y=0.98),
            hovermode='x unified',
            font=dict(family="Arial", size=12),
            autosize=True,
            margin=dict(l=60, r=60, t=80, b=60)
        )
        
        # Configure secondary y-axis
        fig.update_yaxes(title_text="Right Y-axis", secondary_y=True)
        
        html = fig.to_html(include_plotlyjs='cdn', config=self.plot_config)
        self.setHtml(html)
    
    def update_plot(self, datasets, plot_settings, plot_type="line_plot"):
        """Update plot based on type and settings"""
        self.datasets = datasets
        self.current_plot_type = plot_type
        
        if not datasets:
            self.create_empty_plot()
            return
        
        if plot_type == "line_plot":
            fig = self.create_line_plot(datasets, plot_settings)
        elif plot_type == "correlation_matrix":
            fig = self.create_correlation_matrix(datasets, plot_settings)
        elif plot_type == "distributions":
            fig = self.create_distribution_plots(datasets, plot_settings)
        elif plot_type == "scatter_matrix":
            fig = self.create_scatter_matrix(datasets, plot_settings)
        elif plot_type == "box_plots":
            fig = self.create_box_plots(datasets, plot_settings)
        elif plot_type == "violin_plots":
            fig = self.create_violin_plots(datasets, plot_settings)
        else:
            fig = self.create_line_plot(datasets, plot_settings)
        
        if fig:
            html = fig.to_html(include_plotlyjs='cdn', config=self.plot_config)
            self.setHtml(html)
    
    def create_line_plot(self, datasets, plot_settings):
        """Create interactive line plots with dual y-axes"""
        has_secondary = any(ds.y_axis == 'right' and ds.visible for ds in datasets.values())
        
        fig = make_subplots(specs=[[{"secondary_y": has_secondary}]])
        
        for dataset_item in datasets.values():
            if not dataset_item.visible:
                continue
            
            if (dataset_item.x_column >= len(dataset_item.data.columns) or 
                dataset_item.y_column >= len(dataset_item.data.columns)):
                continue
            
            x_data = dataset_item.data.iloc[:, dataset_item.x_column]
            y_data = dataset_item.data.iloc[:, dataset_item.y_column]
            
            # Apply normalization
            y_data = self.apply_normalization(x_data, y_data, dataset_item.normalization)
            
            trace = go.Scatter(
                x=x_data,
                y=y_data,
                mode=self.get_plotly_mode(dataset_item),
                name=dataset_item.name,
                line=dict(
                    color=dataset_item.color,
                    width=dataset_item.line_width,
                    dash=self.get_plotly_line_style(dataset_item.line_style)
                ),
                marker=dict(
                    symbol=self.get_plotly_marker(dataset_item.marker_style),
                    size=dataset_item.marker_size,
                    color=dataset_item.color
                ),
                opacity=dataset_item.opacity,
                fill='tonexty' if dataset_item.fill_area else None,
                hovertemplate='<b>%{fullData.name}</b><br>' +
                             'X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>'
            )
            
            # Add error bars if enabled
            if dataset_item.error_bars and dataset_item.error_column is not None:
                if dataset_item.error_column < len(dataset_item.data.columns):
                    error_data = dataset_item.data.iloc[:, dataset_item.error_column]
                    trace.error_y = dict(type='data', array=error_data, visible=True)
            
            # Add to appropriate y-axis
            secondary_y = dataset_item.y_axis == 'right'
            fig.add_trace(trace, secondary_y=secondary_y)
        
        self.update_layout(fig, plot_settings, has_secondary)
        return fig
    
    def create_correlation_matrix(self, datasets, plot_settings):
        """Create interactive correlation heatmap"""
        combined_data = pd.DataFrame()
        
        for name, dataset_item in datasets.items():
            if dataset_item.visible and len(dataset_item.data.columns) >= 2:
                x_data = dataset_item.data.iloc[:, dataset_item.x_column]
                y_data = dataset_item.data.iloc[:, dataset_item.y_column]
                
                if dataset_item.normalization != "None":
                    y_data = self.apply_normalization(x_data, y_data, dataset_item.normalization)
                
                combined_data[f"{name}_X"] = x_data
                combined_data[f"{name}_Y"] = y_data
        
        if len(combined_data.columns) > 1:
            corr_matrix = combined_data.corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=np.round(corr_matrix.values, 3),
                texttemplate="%{text}",
                textfont={"size": 10},
                hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>'
            ))
            
            fig.update_layout(
                title=dict(text=plot_settings.get('title', 'Dataset Correlation Matrix'), x=0.5),
                xaxis_title="Variables",
                yaxis_title="Variables",
                font=dict(family="Arial", size=12),
                template=plot_settings.get('theme', 'plotly_white'),
                autosize=True
            )
            
            return fig
        
        return None
    
    def create_distribution_plots(self, datasets, plot_settings):
        """Create interactive distribution plots with subplots"""
        visible_datasets = [ds for ds in datasets.values() if ds.visible]
        if not visible_datasets:
            return None
        
        n_datasets = len(visible_datasets)
        n_cols = min(3, n_datasets)
        n_rows = (n_datasets + n_cols - 1) // n_cols
        
        subplot_titles = [ds.name for ds in visible_datasets]
        
        fig = make_subplots(
            rows=n_rows, 
            cols=n_cols,
            subplot_titles=subplot_titles,
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        for i, dataset_item in enumerate(visible_datasets):
            row = i // n_cols + 1
            col = i % n_cols + 1
            
            y_data = dataset_item.data.iloc[:, dataset_item.y_column]
            
            if dataset_item.normalization != "None":
                x_data = dataset_item.data.iloc[:, dataset_item.x_column]
                y_data = self.apply_normalization(x_data, y_data, dataset_item.normalization)
            
            # Add histogram
            fig.add_trace(
                go.Histogram(
                    x=y_data,
                    name=dataset_item.name,
                    marker_color=dataset_item.color,
                    opacity=0.7,
                    showlegend=False,
                    hovertemplate='Value: %{x:.3f}<br>Count: %{y}<extra></extra>'
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title=dict(text=plot_settings.get('title', 'Distribution Analysis'), x=0.5),
            template=plot_settings.get('theme', 'plotly_white'),
            showlegend=False,
            font=dict(family="Arial", size=10),
            autosize=True
        )
        
        return fig
    
    def create_scatter_matrix(self, datasets, plot_settings):
        """Create interactive scatter matrix (pairplot equivalent)"""
        combined_data = pd.DataFrame()
        
        for name, dataset_item in datasets.items():
            if dataset_item.visible and len(dataset_item.data.columns) >= 3:
                for i in range(min(4, len(dataset_item.data.columns))):
                    col_data = dataset_item.data.iloc[:, i]
                    combined_data[f"{name}_Col{i}"] = col_data
        
        if len(combined_data.columns) >= 2:
            # Create scatter matrix using plotly express
            fig = px.scatter_matrix(
                combined_data,
                title=plot_settings.get('title', 'Multivariate Analysis (Pairplot)'),
                template=plot_settings.get('theme', 'plotly_white'),
                height=600
            )
            
            fig.update_traces(diagonal_visible=False)
            fig.update_layout(
                font=dict(family="Arial", size=10),
                autosize=True
            )
            
            return fig
        
        return None
    
    def create_box_plots(self, datasets, plot_settings):
        """Create comparative box plots"""
        fig = go.Figure()
        
        for dataset_item in datasets.values():
            if not dataset_item.visible:
                continue
            
            y_data = dataset_item.data.iloc[:, dataset_item.y_column]
            
            if dataset_item.normalization != "None":
                x_data = dataset_item.data.iloc[:, dataset_item.x_column]
                y_data = self.apply_normalization(x_data, y_data, dataset_item.normalization)
            
            fig.add_trace(go.Box(
                y=y_data,
                name=dataset_item.name,
                marker_color=dataset_item.color,
                boxpoints='outliers',
                hovertemplate='<b>%{fullData.name}</b><br>' +
                             'Q1: %{q1:.3f}<br>' +
                             'Median: %{median:.3f}<br>' +
                             'Q3: %{q3:.3f}<br>' +
                             'Value: %{y:.3f}<extra></extra>'
            ))
        
        fig.update_layout(
            title=dict(text=plot_settings.get('title', 'Distribution Comparison (Box Plots)'), x=0.5),
            yaxis_title=plot_settings.get('ylabel_left', 'Values'),
            template=plot_settings.get('theme', 'plotly_white'),
            showlegend=True,
            font=dict(family="Arial", size=12),
            autosize=True
        )
        
        return fig
    
    def create_violin_plots(self, datasets, plot_settings):
        """Create violin plots for distribution shape analysis"""
        fig = go.Figure()
        
        for dataset_item in datasets.values():
            if not dataset_item.visible:
                continue
            
            y_data = dataset_item.data.iloc[:, dataset_item.y_column]
            
            if dataset_item.normalization != "None":
                x_data = dataset_item.data.iloc[:, dataset_item.x_column]
                y_data = self.apply_normalization(x_data, y_data, dataset_item.normalization)
            
            fig.add_trace(go.Violin(
                y=y_data,
                name=dataset_item.name,
                fillcolor=dataset_item.color,
                opacity=0.7,
                box_visible=True,
                meanline_visible=True,
                hovertemplate='<b>%{fullData.name}</b><br>' +
                             'Value: %{y:.3f}<extra></extra>'
            ))
        
        fig.update_layout(
            title=dict(text=plot_settings.get('title', 'Distribution Shape Analysis (Violin Plots)'), x=0.5),
            yaxis_title=plot_settings.get('ylabel_left', 'Values'),
            template=plot_settings.get('theme', 'plotly_white'),
            showlegend=True,
            font=dict(family="Arial", size=12),
            autosize=True
        )
        
        return fig
    
    def update_layout(self, fig, plot_settings, has_secondary=False):
        """Apply consistent layout styling"""
        fig.update_layout(
            template=plot_settings.get('theme', 'plotly_white'),
            title=dict(text=plot_settings.get('title', ''), x=0.5, font_size=16),
            xaxis=dict(
                title=plot_settings.get('xlabel', 'X-axis'),
                showgrid=plot_settings.get('show_grid', True),
                gridwidth=1,
                gridcolor='lightgray',
                zeroline=True,
                range=plot_settings.get('x_range')
            ),
            yaxis=dict(
                title=plot_settings.get('ylabel_left', 'Y-axis'),
                showgrid=plot_settings.get('show_grid', True),
                gridwidth=1,
                gridcolor='lightgray',
                zeroline=True,
                range=plot_settings.get('y_left_range')
            ),
            showlegend=True,
            legend=dict(x=0.02, y=0.98),
            hovermode='closest',
            font=dict(family="Arial", size=12),
            autosize=True,
            margin=dict(l=60, r=60, t=80, b=60)
        )
        
        if has_secondary:
            fig.update_yaxes(
                title_text=plot_settings.get('ylabel_right', 'Right Y-axis'),
                range=plot_settings.get('y_right_range'),
                secondary_y=True
            )
    
    def apply_normalization(self, x_data, y_data, norm_type):
        """Apply normalization to y_data"""
        if norm_type == "Min-Max":
            return DataNormalizer.min_max_normalize(y_data)
        elif norm_type == "Z-Score":
            return DataNormalizer.z_score_normalize(y_data)
        elif norm_type == "Area":
            return DataNormalizer.area_normalize(x_data, y_data)
        elif norm_type == "Baseline":
            return DataNormalizer.baseline_correct(x_data, y_data)
        elif norm_type == "Peak":
            return DataNormalizer.peak_normalize(y_data)
        elif norm_type == "Vector":
            return DataNormalizer.vector_normalize(y_data)
        else:
            return y_data
    
    def get_plotly_mode(self, dataset_item):
        """Convert dataset styling to Plotly mode"""
        mode = []
        if dataset_item.line_style != 'none':
            mode.append('lines')
        if dataset_item.marker_style != 'none':
            mode.append('markers')
        return '+'.join(mode) if mode else 'lines'
    
    def get_plotly_line_style(self, style):
        """Convert line style to Plotly dash style"""
        style_map = {
            'solid': 'solid',
            'dashed': 'dash',
            'dotted': 'dot',
            'dashdot': 'dashdot'
        }
        return style_map.get(style, 'solid')
    
    def get_plotly_marker(self, marker_style):
        """Convert marker style to Plotly symbol"""
        marker_map = {
            'circle': 'circle',
            'square': 'square',
            'diamond': 'diamond',
            'triangle': 'triangle-up',
            'cross': 'cross',
            'x': 'x'
        }
        return marker_map.get(marker_style, 'circle')


class DatasetManager(QWidget):
    """Widget for managing multiple datasets with enhanced controls"""
    
    dataset_changed = Signal()
    
    def __init__(self):
        super().__init__()
        self.datasets = {}
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Header
        header = QLabel("Loaded Datasets")
        header.setFont(QFont("Arial", 10, QFont.Bold))
        layout.addWidget(header)
        
        # Scroll area for datasets
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(400)
        
        self.dataset_container = QWidget()
        self.dataset_layout = QVBoxLayout()
        self.dataset_container.setLayout(self.dataset_layout)
        scroll.setWidget(self.dataset_container)
        
        layout.addWidget(scroll)
        
        # Control buttons
        btn_layout = QHBoxLayout()
        
        self.remove_btn = QPushButton("Remove Selected")
        self.remove_btn.clicked.connect(self.remove_selected_dataset)
        btn_layout.addWidget(self.remove_btn)
        
        self.clear_all_btn = QPushButton("Clear All")
        self.clear_all_btn.clicked.connect(self.clear_all_datasets)
        btn_layout.addWidget(self.clear_all_btn)
        
        layout.addLayout(btn_layout)
        self.setLayout(layout)
    
    def add_dataset(self, dataset_item):
        """Add a new dataset to the manager"""
        self.datasets[dataset_item.name] = dataset_item
        self.create_dataset_widget(dataset_item)
        self.dataset_changed.emit()
    
    def create_dataset_widget(self, dataset_item):
        """Create enhanced widget for individual dataset control"""
        frame = QFrame()
        frame.setFrameStyle(QFrame.Box)
        frame.setLineWidth(1)
        
        layout = QGridLayout()
        
        # Dataset name and visibility
        name_layout = QHBoxLayout()
        
        visibility_cb = QCheckBox()
        visibility_cb.setChecked(dataset_item.visible)
        visibility_cb.toggled.connect(lambda checked, name=dataset_item.name: self.set_visibility(name, checked))
        name_layout.addWidget(visibility_cb)
        
        name_label = QLabel(dataset_item.name)
        name_label.setFont(QFont("Arial", 9, QFont.Bold))
        name_layout.addWidget(name_label)
        name_layout.addStretch()
        
        layout.addLayout(name_layout, 0, 0, 1, 4)
        
        # Column selection
        layout.addWidget(QLabel("X:"), 1, 0)
        x_combo = QComboBox()
        x_combo.addItems([f"Col {i}" for i in range(len(dataset_item.data.columns))])
        x_combo.setCurrentIndex(dataset_item.x_column)
        x_combo.currentIndexChanged.connect(lambda idx, name=dataset_item.name: self.set_x_column(name, idx))
        layout.addWidget(x_combo, 1, 1)
        
        layout.addWidget(QLabel("Y:"), 1, 2)
        y_combo = QComboBox()
        y_combo.addItems([f"Col {i}" for i in range(len(dataset_item.data.columns))])
        y_combo.setCurrentIndex(dataset_item.y_column)
        y_combo.currentIndexChanged.connect(lambda idx, name=dataset_item.name: self.set_y_column(name, idx))
        layout.addWidget(y_combo, 1, 3)
        
        # Y-axis assignment and styling
        layout.addWidget(QLabel("Y-axis:"), 2, 0)
        axis_combo = QComboBox()
        axis_combo.addItems(["Left", "Right"])
        axis_combo.setCurrentText("Left" if dataset_item.y_axis == 'left' else "Right")
        axis_combo.currentTextChanged.connect(lambda text, name=dataset_item.name: self.set_y_axis(name, text.lower()))
        layout.addWidget(axis_combo, 2, 1)
        
        # Line style
        layout.addWidget(QLabel("Line:"), 2, 2)
        line_combo = QComboBox()
        line_combo.addItems(["solid", "dashed", "dotted", "dashdot", "none"])
        line_combo.setCurrentText(dataset_item.line_style)
        line_combo.currentTextChanged.connect(lambda text, name=dataset_item.name: self.set_line_style(name, text))
        layout.addWidget(line_combo, 2, 3)
        
        # Marker and color
        layout.addWidget(QLabel("Marker:"), 3, 0)
        marker_combo = QComboBox()
        marker_combo.addItems(["none", "circle", "square", "diamond", "triangle", "cross", "x"])
        marker_combo.setCurrentText(dataset_item.marker_style)
        marker_combo.currentTextChanged.connect(lambda text, name=dataset_item.name: self.set_marker_style(name, text))
        layout.addWidget(marker_combo, 3, 1)
        
        layout.addWidget(QLabel("Color:"), 3, 2)
        color_btn = QPushButton()
        color_btn.setStyleSheet(f"background-color: {dataset_item.color}; border: 1px solid black;")
        color_btn.clicked.connect(lambda checked, name=dataset_item.name: self.select_color(name, color_btn))
        layout.addWidget(color_btn, 3, 3)
        
        # Normalization and opacity
        layout.addWidget(QLabel("Norm:"), 4, 0)
        norm_combo = QComboBox()
        norm_combo.addItems(["None", "Min-Max", "Z-Score", "Area", "Baseline", "Peak", "Vector"])
        norm_combo.setCurrentText(dataset_item.normalization)
        norm_combo.currentTextChanged.connect(lambda text, name=dataset_item.name: self.set_normalization(name, text))
        layout.addWidget(norm_combo, 4, 1)
        
        layout.addWidget(QLabel("Opacity:"), 4, 2)
        opacity_slider = QSlider(Qt.Horizontal)
        opacity_slider.setRange(10, 100)
        opacity_slider.setValue(int(dataset_item.opacity * 100))
        opacity_slider.valueChanged.connect(lambda val, name=dataset_item.name: self.set_opacity(name, val/100))
        layout.addWidget(opacity_slider, 4, 3)
        
        # Advanced options
        advanced_layout = QHBoxLayout()
        
        fill_cb = QCheckBox("Fill Area")
        fill_cb.setChecked(dataset_item.fill_area)
        fill_cb.toggled.connect(lambda checked, name=dataset_item.name: self.set_fill_area(name, checked))
        advanced_layout.addWidget(fill_cb)
        
        error_cb = QCheckBox("Error Bars")
        error_cb.setChecked(dataset_item.error_bars)
        error_cb.toggled.connect(lambda checked, name=dataset_item.name: self.set_error_bars(name, checked))
        advanced_layout.addWidget(error_cb)
        
        layout.addLayout(advanced_layout, 5, 0, 1, 4)
        
        # Store references for removal
        frame.dataset_name = dataset_item.name
        frame.setProperty("dataset_name", dataset_item.name)
        
        frame.setLayout(layout)
        self.dataset_layout.addWidget(frame)
    
    # Dataset property setters
    def set_visibility(self, name, visible):
        if name in self.datasets:
            self.datasets[name].visible = visible
            self.dataset_changed.emit()
    
    def set_x_column(self, name, column):
        if name in self.datasets:
            self.datasets[name].x_column = column
            self.dataset_changed.emit()
    
    def set_y_column(self, name, column):
        if name in self.datasets:
            self.datasets[name].y_column = column
            self.dataset_changed.emit()
    
    def set_y_axis(self, name, axis):
        if name in self.datasets:
            self.datasets[name].y_axis = axis
            self.dataset_changed.emit()
    
    def set_line_style(self, name, style):
        if name in self.datasets:
            self.datasets[name].line_style = style
            self.dataset_changed.emit()
    
    def set_marker_style(self, name, style):
        if name in self.datasets:
            self.datasets[name].marker_style = style
            self.dataset_changed.emit()
    
    def set_normalization(self, name, norm_type):
        if name in self.datasets:
            self.datasets[name].normalization = norm_type
            self.dataset_changed.emit()
    
    def set_opacity(self, name, opacity):
        if name in self.datasets:
            self.datasets[name].opacity = opacity
            self.dataset_changed.emit()
    
    def set_fill_area(self, name, fill):
        if name in self.datasets:
            self.datasets[name].fill_area = fill
            self.dataset_changed.emit()
    
    def set_error_bars(self, name, error_bars):
        if name in self.datasets:
            self.datasets[name].error_bars = error_bars
            self.dataset_changed.emit()
    
    def select_color(self, name, button):
        """Open color picker"""
        color = QColorDialog.getColor()
        if color.isValid():
            color_name = color.name()
            self.datasets[name].color = color_name
            button.setStyleSheet(f"background-color: {color_name}; border: 1px solid black;")
            self.dataset_changed.emit()
    
    def remove_selected_dataset(self):
        """Remove selected dataset (placeholder - could add selection logic)"""
        if self.datasets:
            last_name = list(self.datasets.keys())[-1]
            self.remove_dataset(last_name)
    
    def remove_dataset(self, name):
        """Remove specific dataset"""
        if name in self.datasets:
            del self.datasets[name]
            
            # Remove widget
            for i in range(self.dataset_layout.count()):
                widget = self.dataset_layout.itemAt(i).widget()
                if widget and hasattr(widget, 'dataset_name') and widget.dataset_name == name:
                    widget.deleteLater()
                    break
            
            self.dataset_changed.emit()
    
    def clear_all_datasets(self):
        """Clear all datasets"""
        self.datasets.clear()
        
        # Clear all widgets
        while self.dataset_layout.count():
            child = self.dataset_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        self.dataset_changed.emit()
    
    def get_datasets(self):
        """Get all datasets"""
        return self.datasets


class TabbedControlPanel(QWidget):
    """Enhanced control panel with tabbed interface for better organization"""
    
    plot_updated = Signal()
    plot_type_changed = Signal(str)
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Create tab widget for control panel
        self.control_tabs = QTabWidget()
        self.control_tabs.setTabPosition(QTabWidget.North)
        
        # Tab 1: Data Management
        self.data_tab = self.create_data_management_tab()
        self.control_tabs.addTab(self.data_tab, "📊 Data")
        
        # Tab 2: Visualization Controls
        self.viz_tab = self.create_visualization_tab()
        self.control_tabs.addTab(self.viz_tab, "🎨 Style")
        
        # Tab 3: Advanced Options
        self.advanced_tab = self.create_advanced_tab()
        self.control_tabs.addTab(self.advanced_tab, "⚙️ Advanced")
        
        layout.addWidget(self.control_tabs)
        
        # Update button at bottom
        update_btn = QPushButton("Update Visualization")
        update_btn.clicked.connect(lambda: self.plot_updated.emit)
        update_btn.setStyleSheet("""
            QPushButton { 
                background-color: #4CAF50; 
                color: white; 
                font-weight: bold; 
                padding: 8px;
                border-radius: 4px;
                border: none;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        layout.addWidget(update_btn)
        
        self.setLayout(layout)
    
    def create_data_management_tab(self):
        """Create the data management tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Dataset manager with more space
        self.dataset_manager = EnhancedDatasetManager()
        self.dataset_manager.dataset_changed.connect(lambda: self.plot_updated.emit)
        layout.addWidget(self.dataset_manager)
        
        # Plot type selection (moved here for better organization)
        plot_type_group = QGroupBox("Visualization Type")
        plot_type_layout = QVBoxLayout()
        
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems([
            "Line Plot", "Correlation Matrix", "Distributions", 
            "Scatter Matrix", "Box Plots", "Violin Plots"
        ])
        self.plot_type_combo.currentTextChanged.connect(lambda: self.on_plot_type_changed)
        plot_type_layout.addWidget(self.plot_type_combo)
        
        plot_type_group.setLayout(plot_type_layout)
        layout.addWidget(plot_type_group)
        
        #layout.addStretch()
        tab.setLayout(layout)
        return tab
    
    def create_visualization_tab(self):
        """Create the visualization styling tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Plot titles and labels
        titles_group = QGroupBox("Plot Labels")
        titles_layout = QGridLayout()
        
        titles_layout.addWidget(QLabel("Title:"), 0, 0)
        self.title_edit = QLineEdit()
        self.title_edit.setPlaceholderText("Plot Title")
        #self.title_edit.textChanged.connect(lambda: self.plot_updated.emit())
        titles_layout.addWidget(self.title_edit, 0, 1)
        
        titles_layout.addWidget(QLabel("X-axis:"), 1, 0)
        self.xlabel_edit = QLineEdit()
        self.xlabel_edit.setPlaceholderText("X-axis Label")
        #self.xlabel_edit.textChanged.connect(lambda: self.plot_updated.emit())
        titles_layout.addWidget(self.xlabel_edit, 1, 1)
        
        titles_layout.addWidget(QLabel("Y-axis:"), 2, 0)
        self.ylabel_edit = QLineEdit()
        self.ylabel_edit.setPlaceholderText("Y-axis Label")
        #self.ylabel_edit.textChanged.connect(lambda: self.plot_updated.emit())
        titles_layout.addWidget(self.ylabel_edit, 2, 1)
        
        titles_layout.addWidget(QLabel("Right Y:"), 3, 0)
        self.ylabel2_edit = QLineEdit()
        self.ylabel2_edit.setPlaceholderText("Right Y-axis Label")
        #self.ylabel2_edit.textChanged.connect(lambda: self.plot_updated.emit())
        titles_layout.addWidget(self.ylabel2_edit, 3, 1)
        
        titles_group.setLayout(titles_layout)
        layout.addWidget(titles_group)
        
        # Theme selection
        theme_group = QGroupBox("Theme & Appearance")
        theme_layout = QGridLayout()
        
        theme_layout.addWidget(QLabel("Theme:"), 0, 0)
        self.theme_combo = QComboBox()
        self.theme_combo.addItems([
            "plotly_white", "plotly", "plotly_dark", "ggplot2", 
            "seaborn", "simple_white", "presentation"
        ])
        self.theme_combo.currentTextChanged.connect(lambda: self.plot_updated.emit())
        theme_layout.addWidget(self.theme_combo, 0, 1)
        
        # Grid options
        self.show_grid_cb = QCheckBox("Show Grid")
        self.show_grid_cb.setChecked(True)
        self.show_grid_cb.toggled.connect(lambda: self.plot_updated.emit())
        theme_layout.addWidget(self.show_grid_cb, 1, 0, 1, 2)
        
        theme_group.setLayout(theme_layout)
        layout.addWidget(theme_group)
        
        layout.addStretch()
        tab.setLayout(layout)
        return tab
    
    def create_advanced_tab(self):
        """Create the advanced options tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Axis limits
        limits_group = QGroupBox("Axis Limits")
        limits_layout = QGridLayout()
        
        # X-axis limits
        limits_layout.addWidget(QLabel("X-axis:"), 0, 0)
        self.x_auto_cb = QCheckBox("Auto")
        self.x_auto_cb.setChecked(True)
        self.x_auto_cb.toggled.connect(lambda: self.on_axis_auto_toggled())
        limits_layout.addWidget(self.x_auto_cb, 0, 1)
        
        limits_layout.addWidget(QLabel("Min:"), 0, 2)
        self.x_min_edit = QDoubleSpinBox()
        self.x_min_edit.setRange(-1e6, 1e6)
        self.x_min_edit.setDecimals(3)
        self.x_min_edit.setEnabled(False)
        self.x_min_edit.valueChanged.connect(lambda: self.plot_updated.emit())
        limits_layout.addWidget(self.x_min_edit, 0, 3)
        
        limits_layout.addWidget(QLabel("Max:"), 0, 4)
        self.x_max_edit = QDoubleSpinBox()
        self.x_max_edit.setRange(-1e6, 1e6)
        self.x_max_edit.setDecimals(3)
        self.x_max_edit.setEnabled(False)
        self.x_max_edit.valueChanged.connect(lambda: self.plot_updated.emit())
        limits_layout.addWidget(self.x_max_edit, 0, 5)
        
        # Left Y-axis limits
        limits_layout.addWidget(QLabel("Left Y:"), 1, 0)
        self.y_left_auto_cb = QCheckBox("Auto")
        self.y_left_auto_cb.setChecked(True)
        self.y_left_auto_cb.toggled.connect(lambda: self.on_axis_auto_toggled())
        limits_layout.addWidget(self.y_left_auto_cb, 1, 1)
        
        limits_layout.addWidget(QLabel("Min:"), 1, 2)
        self.y_left_min_edit = QDoubleSpinBox()
        self.y_left_min_edit.setRange(-1e6, 1e6)
        self.y_left_min_edit.setDecimals(3)
        self.y_left_min_edit.setEnabled(False)
        self.y_left_min_edit.valueChanged.connect(lambda: self.plot_updated.emit())
        limits_layout.addWidget(self.y_left_min_edit, 1, 3)
        
        limits_layout.addWidget(QLabel("Max:"), 1, 4)
        self.y_left_max_edit = QDoubleSpinBox()
        self.y_left_max_edit.setRange(-1e6, 1e6)
        self.y_left_max_edit.setDecimals(3)
        self.y_left_max_edit.setEnabled(False)
        self.y_left_max_edit.valueChanged.connect(lambda: self.plot_updated.emit())
        limits_layout.addWidget(self.y_left_max_edit, 1, 5)
        
        # Right Y-axis limits
        limits_layout.addWidget(QLabel("Right Y:"), 2, 0)
        self.y_right_auto_cb = QCheckBox("Auto")
        self.y_right_auto_cb.setChecked(True)
        self.y_right_auto_cb.toggled.connect(lambda: self.on_axis_auto_toggled())
        limits_layout.addWidget(self.y_right_auto_cb, 2, 1)
        
        limits_layout.addWidget(QLabel("Min:"), 2, 2)
        self.y_right_min_edit = QDoubleSpinBox()
        self.y_right_min_edit.setRange(-1e6, 1e6)
        self.y_right_min_edit.setDecimals(3)
        self.y_right_min_edit.setEnabled(False)
        self.y_right_min_edit.valueChanged.connect(lambda: self.plot_updated.emit())
        limits_layout.addWidget(self.y_right_min_edit, 2, 3)
        
        limits_layout.addWidget(QLabel("Max:"), 2, 4)
        self.y_right_max_edit = QDoubleSpinBox()
        self.y_right_max_edit.setRange(-1e6, 1e6)
        self.y_right_max_edit.setDecimals(3)
        self.y_right_max_edit.setEnabled(False)
        self.y_right_max_edit.valueChanged.connect(self.plot_updated.emit)
        limits_layout.addWidget(self.y_right_max_edit, 2, 5)
        
        limits_group.setLayout(limits_layout)
        layout.addWidget(limits_group)
        
        # Export options
        export_group = QGroupBox("Export Options")
        export_layout = QVBoxLayout()
        
        export_btn_layout = QHBoxLayout()
        
        export_png_btn = QPushButton("📷 Export PNG")
        export_png_btn.setToolTip("Export as high-resolution PNG image")
        export_btn_layout.addWidget(export_png_btn)
        
        export_html_btn = QPushButton("🌐 Export HTML")
        export_html_btn.setToolTip("Export as interactive HTML")
        export_btn_layout.addWidget(export_html_btn)
        
        export_data_btn = QPushButton("📊 Export Data")
        export_data_btn.setToolTip("Export processed data as CSV")
        export_btn_layout.addWidget(export_data_btn)
        
        export_layout.addLayout(export_btn_layout)
        export_group.setLayout(export_layout)
        layout.addWidget(export_group)
        
        layout.addStretch()
        tab.setLayout(layout)
        return tab
    
    def on_plot_type_changed(self, plot_type):
        """Handle plot type change"""
        plot_type_map = {
            "Line Plot": "line_plot",
            "Correlation Matrix": "correlation_matrix",
            "Distributions": "distributions",
            "Scatter Matrix": "scatter_matrix",
            "Box Plots": "box_plots",
            "Violin Plots": "violin_plots"
        }
        self.plot_type_changed.emit(plot_type_map.get(plot_type, "line_plot"))
    
    def on_axis_auto_toggled(self):
        """Handle axis auto toggle"""
        self.x_min_edit.setEnabled(not self.x_auto_cb.isChecked())
        self.x_max_edit.setEnabled(not self.x_auto_cb.isChecked())
        self.y_left_min_edit.setEnabled(not self.y_left_auto_cb.isChecked())
        self.y_left_max_edit.setEnabled(not self.y_left_auto_cb.isChecked())
        self.y_right_min_edit.setEnabled(not self.y_right_auto_cb.isChecked())
        self.y_right_max_edit.setEnabled(not self.y_right_auto_cb.isChecked())
        self.plot_updated.emit()
    
    def get_plot_settings(self):
        """Get current plot settings"""
        return {
            'title': self.title_edit.text(),
            'xlabel': self.xlabel_edit.text(),
            'ylabel_left': self.ylabel_edit.text(),
            'ylabel_right': self.ylabel2_edit.text(),
            'theme': self.theme_combo.currentText(),
            'x_range': None if self.x_auto_cb.isChecked() else [self.x_min_edit.value(), self.x_max_edit.value()],
            'y_left_range': None if self.y_left_auto_cb.isChecked() else [self.y_left_min_edit.value(), self.y_left_max_edit.value()],
            'y_right_range': None if self.y_right_auto_cb.isChecked() else [self.y_right_min_edit.value(), self.y_right_max_edit.value()],
            'show_grid': getattr(self, 'show_grid_cb', None) and self.show_grid_cb.isChecked()
        }
    
    def get_current_plot_type(self):
        """Get current plot type"""
        plot_type_map = {
            "Line Plot": "line_plot",
            "Correlation Matrix": "correlation_matrix",
            "Distributions": "distributions",
            "Scatter Matrix": "scatter_matrix",
            "Box Plots": "box_plots",
            "Violin Plots": "violin_plots"
        }
        return plot_type_map.get(self.plot_type_combo.currentText(), "line_plot")


class EnhancedDatasetManager(QWidget):
    """Enhanced dataset manager with better layout and more space"""
    
    dataset_changed = Signal()
    
    def __init__(self):
        super().__init__()
        self.datasets = {}
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Header with actions
        header_layout = QHBoxLayout()
        header = QLabel("Loaded Datasets")
        header.setFont(QFont("Arial", 11, QFont.Bold))
        header_layout.addWidget(header)
        
        # Add dataset info
        self.dataset_count_label = QLabel("(0 datasets)")
        self.dataset_count_label.setStyleSheet("color: #666; font-size: 10px;")
        header_layout.addWidget(self.dataset_count_label)
        header_layout.addStretch()
        
        layout.addLayout(header_layout)
        
        # Scroll area for datasets with more generous sizing
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(500)
        scroll.setMaximumHeight(800)  # Allow more height
        
        self.dataset_container = QWidget()
        self.dataset_layout = QVBoxLayout()
        self.dataset_layout.setSpacing(8)  # More spacing between datasets
        self.dataset_container.setLayout(self.dataset_layout)
        scroll.setWidget(self.dataset_container)
        
        layout.addWidget(scroll)
        
        # Control buttons with better styling
        btn_layout = QHBoxLayout()
        
        self.remove_btn = QPushButton("Remove Last")
        self.remove_btn.setToolTip("Remove the most recently added dataset")
        self.remove_btn.clicked.connect(self.remove_selected_dataset)
        self.remove_btn.setStyleSheet("""
            QPushButton { 
                background-color: #f44336; 
                color: white; 
                border: none; 
                padding: 6px 12px;
                border-radius: 3px;
            }
            QPushButton:hover { background-color: #da190b; }
            QPushButton:disabled { background-color: #cccccc; }
        """)
        btn_layout.addWidget(self.remove_btn)
        
        self.clear_all_btn = QPushButton("Clear All")
        self.clear_all_btn.setToolTip("Remove all datasets")
        self.clear_all_btn.clicked.connect(self.clear_all_datasets)
        self.clear_all_btn.setStyleSheet("""
            QPushButton { 
                background-color: #ff9800; 
                color: white; 
                border: none; 
                padding: 6px 12px;
                border-radius: 3px;
            }
            QPushButton:hover { background-color: #e68900; }
            QPushButton:disabled { background-color: #cccccc; }
        """)
        btn_layout.addWidget(self.clear_all_btn)
        
        layout.addLayout(btn_layout)
        self.setLayout(layout)
        
        self.update_button_states()
    
    def create_dataset_widget(self, dataset_item):
        """Create enhanced widget for individual dataset control with better layout"""
        frame = QFrame()
        frame.setFrameStyle(QFrame.StyledPanel)
        frame.setLineWidth(1)
        frame.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                margin: 2px;
            }
        """)
        
        layout = QVBoxLayout()
        layout.setSpacing(6)
        
        # Dataset name and visibility
        name_layout = QHBoxLayout()
        
        visibility_cb = QCheckBox()
        visibility_cb.setChecked(dataset_item.visible)
        visibility_cb.toggled.connect(lambda checked, name=dataset_item.name: self.set_visibility(name, checked))
        name_layout.addWidget(visibility_cb)
        
        name_label = QLabel(dataset_item.name)
        name_label.setFont(QFont("Arial", 9, QFont.Bold))
        name_label.setWordWrap(True)
        name_layout.addWidget(name_label)
        name_layout.addStretch()
        
        layout.addLayout(name_layout)
        
        # Column selection in a grid
        col_group = QGroupBox("Data Columns")
        col_layout = QGridLayout()
        col_layout.setSpacing(4)
        
        col_layout.addWidget(QLabel("X:"), 0, 0)
        x_combo = QComboBox()
        x_combo.addItems([f"Col {i}" for i in range(len(dataset_item.data.columns))])
        x_combo.setCurrentIndex(dataset_item.x_column)
        x_combo.currentIndexChanged.connect(lambda idx, name=dataset_item.name: self.set_x_column(name, idx))
        col_layout.addWidget(x_combo, 0, 1)
        
        col_layout.addWidget(QLabel("Y:"), 0, 2)
        y_combo = QComboBox()
        y_combo.addItems([f"Col {i}" for i in range(len(dataset_item.data.columns))])
        y_combo.setCurrentIndex(dataset_item.y_column)
        y_combo.currentIndexChanged.connect(lambda idx, name=dataset_item.name: self.set_y_column(name, idx))
        col_layout.addWidget(y_combo, 0, 3)
        
        col_layout.addWidget(QLabel("Axis:"), 1, 0)
        axis_combo = QComboBox()
        axis_combo.addItems(["Left", "Right"])
        axis_combo.setCurrentText("Left" if dataset_item.y_axis == 'left' else "Right")
        axis_combo.currentTextChanged.connect(lambda text, name=dataset_item.name: self.set_y_axis(name, text.lower()))
        col_layout.addWidget(axis_combo, 1, 1)
        
        col_layout.addWidget(QLabel("Norm:"), 1, 2)
        norm_combo = QComboBox()
        norm_combo.addItems(["None", "Min-Max", "Z-Score", "Area", "Baseline", "Peak", "Vector"])
        norm_combo.setCurrentText(dataset_item.normalization)
        norm_combo.currentTextChanged.connect(lambda text, name=dataset_item.name: self.set_normalization(name, text))
        col_layout.addWidget(norm_combo, 1, 3)
        
        col_group.setLayout(col_layout)
        layout.addWidget(col_group)
        
        # Styling options
        style_group = QGroupBox("Style")
        style_layout = QGridLayout()
        style_layout.setSpacing(4)
        
        style_layout.addWidget(QLabel("Line:"), 0, 0)
        line_combo = QComboBox()
        line_combo.addItems(["solid", "dashed", "dotted", "dashdot", "none"])
        line_combo.setCurrentText(dataset_item.line_style)
        line_combo.currentTextChanged.connect(lambda text, name=dataset_item.name: self.set_line_style(name, text))
        style_layout.addWidget(line_combo, 0, 1)
        
        style_layout.addWidget(QLabel("Marker:"), 0, 2)
        marker_combo = QComboBox()
        marker_combo.addItems(["none", "circle", "square", "diamond", "triangle", "cross", "x"])
        marker_combo.setCurrentText(dataset_item.marker_style)
        marker_combo.currentTextChanged.connect(lambda text, name=dataset_item.name: self.set_marker_style(name, text))
        style_layout.addWidget(marker_combo, 0, 3)
        
        style_layout.addWidget(QLabel("Color:"), 1, 0)
        color_btn = QPushButton()
        color_btn.setFixedSize(40, 25)
        color_btn.setStyleSheet(f"background-color: {dataset_item.color}; border: 1px solid black; border-radius: 3px;")
        color_btn.clicked.connect(lambda checked, name=dataset_item.name: self.select_color(name, color_btn))
        style_layout.addWidget(color_btn, 1, 1)
        
        style_layout.addWidget(QLabel("Opacity:"), 1, 2)
        opacity_slider = QSlider(Qt.Horizontal)
        opacity_slider.setRange(10, 100)
        opacity_slider.setValue(int(dataset_item.opacity * 100))
        opacity_slider.valueChanged.connect(lambda val, name=dataset_item.name: self.set_opacity(name, val/100))
        style_layout.addWidget(opacity_slider, 1, 3)
        
        style_group.setLayout(style_layout)
        layout.addWidget(style_group)
        
        # Store references for removal
        frame.dataset_name = dataset_item.name
        frame.setLayout(layout)
        self.dataset_layout.addWidget(frame)
    
    def add_dataset(self, dataset_item):
        """Add a new dataset to the manager"""
        self.datasets[dataset_item.name] = dataset_item
        self.create_dataset_widget(dataset_item)
        self.update_dataset_count()
        self.update_button_states()
        self.dataset_changed.emit()
    
    def update_dataset_count(self):
        """Update the dataset count label"""
        count = len(self.datasets)
        self.dataset_count_label.setText(f"({count} dataset{'s' if count != 1 else ''})")
    
    def update_button_states(self):
        """Update button enabled states"""
        has_datasets = len(self.datasets) > 0
        self.remove_btn.setEnabled(has_datasets)
        self.clear_all_btn.setEnabled(has_datasets)
    
    def remove_selected_dataset(self):
        """Remove selected dataset (most recent)"""
        if self.datasets:
            last_name = list(self.datasets.keys())[-1]
            self.remove_dataset(last_name)
    
    def remove_dataset(self, name):
        """Remove specific dataset"""
        if name in self.datasets:
            del self.datasets[name]
            
            # Remove widget
            for i in range(self.dataset_layout.count()):
                widget = self.dataset_layout.itemAt(i).widget()
                if widget and hasattr(widget, 'dataset_name') and widget.dataset_name == name:
                    widget.deleteLater()
                    break
            
            self.update_dataset_count()
            self.update_button_states()
            self.dataset_changed.emit()
    
    def clear_all_datasets(self):
        """Clear all datasets"""
        self.datasets.clear()
        
        # Clear all widgets
        while self.dataset_layout.count():
            child = self.dataset_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        self.update_dataset_count()
        self.update_button_states()
        self.dataset_changed.emit()
    
    def get_datasets(self):
        """Get all datasets"""
        return self.datasets
    
    # All the setter methods remain the same as in the original DatasetManager
    def set_visibility(self, name, visible):
        if name in self.datasets:
            self.datasets[name].visible = visible
            self.dataset_changed.emit()
    
    def set_x_column(self, name, column):
        if name in self.datasets:
            self.datasets[name].x_column = column
            self.dataset_changed.emit()
    
    def set_y_column(self, name, column):
        if name in self.datasets:
            self.datasets[name].y_column = column
            self.dataset_changed.emit()
    
    def set_y_axis(self, name, axis):
        if name in self.datasets:
            self.datasets[name].y_axis = axis
            self.dataset_changed.emit()
    
    def set_line_style(self, name, style):
        if name in self.datasets:
            self.datasets[name].line_style = style
            self.dataset_changed.emit()
    
    def set_marker_style(self, name, style):
        if name in self.datasets:
            self.datasets[name].marker_style = style
            self.dataset_changed.emit()
    
    def set_normalization(self, name, norm_type):
        if name in self.datasets:
            self.datasets[name].normalization = norm_type
            self.dataset_changed.emit()
    
    def set_opacity(self, name, opacity):
        if name in self.datasets:
            self.datasets[name].opacity = opacity
            self.dataset_changed.emit()
    
    def select_color(self, name, button):
        """Open color picker"""
        color = QColorDialog.getColor()
        if color.isValid():
            color_name = color.name()
            self.datasets[name].color = color_name
            button.setStyleSheet(f"background-color: {color_name}; border: 1px solid black; border-radius: 3px;")
            self.dataset_changed.emit()


class EnhancedDatasetManager(QWidget):
    """Enhanced dataset manager with better layout and more space"""
    
    dataset_changed = Signal()
    
    def __init__(self):
        super().__init__()
        self.datasets = {}
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Header with actions
        header_layout = QHBoxLayout()
        header = QLabel("Loaded Datasets")
        header.setFont(QFont("Arial", 11, QFont.Bold))
        header_layout.addWidget(header)
        
        # Add dataset info
        self.dataset_count_label = QLabel("(0 datasets)")
        self.dataset_count_label.setStyleSheet("color: #666; font-size: 10px;")
        header_layout.addWidget(self.dataset_count_label)
        header_layout.addStretch()
        
        layout.addLayout(header_layout)
        
        # Scroll area for datasets with more generous sizing
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(200)
        scroll.setMaximumHeight(500)  # Allow more height
        
        self.dataset_container = QWidget()
        self.dataset_layout = QVBoxLayout()
        self.dataset_layout.setSpacing(8)  # More spacing between datasets
        self.dataset_container.setLayout(self.dataset_layout)
        scroll.setWidget(self.dataset_container)
        
        layout.addWidget(scroll)
        
        # Control buttons with better styling
        btn_layout = QHBoxLayout()
        
        self.remove_btn = QPushButton("Remove Last")
        self.remove_btn.setToolTip("Remove the most recently added dataset")
        self.remove_btn.clicked.connect(self.remove_selected_dataset)
        self.remove_btn.setStyleSheet("""
            QPushButton { 
                background-color: #f44336; 
                color: white; 
                border: none; 
                padding: 6px 12px;
                border-radius: 3px;
            }
            QPushButton:hover { background-color: #da190b; }
            QPushButton:disabled { background-color: #cccccc; }
        """)
        btn_layout.addWidget(self.remove_btn)
        
        self.clear_all_btn = QPushButton("Clear All")
        self.clear_all_btn.setToolTip("Remove all datasets")
        self.clear_all_btn.clicked.connect(self.clear_all_datasets)
        self.clear_all_btn.setStyleSheet("""
            QPushButton { 
                background-color: #ff9800; 
                color: white; 
                border: none; 
                padding: 6px 12px;
                border-radius: 3px;
            }
            QPushButton:hover { background-color: #e68900; }
            QPushButton:disabled { background-color: #cccccc; }
        """)
        btn_layout.addWidget(self.clear_all_btn)
        
        layout.addLayout(btn_layout)
        self.setLayout(layout)
        
        self.update_button_states()
    
    def create_dataset_widget(self, dataset_item):
        """Create enhanced widget for individual dataset control with better layout"""
        frame = QFrame()
        frame.setFrameStyle(QFrame.StyledPanel)
        frame.setLineWidth(1)
        frame.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                margin: 2px;
            }
        """)
        
        layout = QVBoxLayout()
        layout.setSpacing(6)
        
        # Dataset name and visibility
        name_layout = QHBoxLayout()
        
        visibility_cb = QCheckBox()
        visibility_cb.setChecked(dataset_item.visible)
        visibility_cb.toggled.connect(lambda checked, name=dataset_item.name: self.set_visibility(name, checked))
        name_layout.addWidget(visibility_cb)
        
        name_label = QLabel(dataset_item.name)
        name_label.setFont(QFont("Arial", 9, QFont.Bold))
        name_label.setWordWrap(True)
        name_layout.addWidget(name_label)
        name_layout.addStretch()
        
        layout.addLayout(name_layout)
        
        # Column selection in a grid
        col_group = QGroupBox("Data Columns")
        col_layout = QGridLayout()
        col_layout.setSpacing(4)
        
        col_layout.addWidget(QLabel("X:"), 0, 0)
        x_combo = QComboBox()
        x_combo.addItems([f"Col {i}" for i in range(len(dataset_item.data.columns))])
        x_combo.setCurrentIndex(dataset_item.x_column)
        x_combo.currentIndexChanged.connect(lambda idx, name=dataset_item.name: self.set_x_column(name, idx))
        col_layout.addWidget(x_combo, 0, 1)
        
        col_layout.addWidget(QLabel("Y:"), 0, 2)
        y_combo = QComboBox()
        y_combo.addItems([f"Col {i}" for i in range(len(dataset_item.data.columns))])
        y_combo.setCurrentIndex(dataset_item.y_column)
        y_combo.currentIndexChanged.connect(lambda idx, name=dataset_item.name: self.set_y_column(name, idx))
        col_layout.addWidget(y_combo, 0, 3)
        
        col_layout.addWidget(QLabel("Axis:"), 1, 0)
        axis_combo = QComboBox()
        axis_combo.addItems(["Left", "Right"])
        axis_combo.setCurrentText("Left" if dataset_item.y_axis == 'left' else "Right")
        axis_combo.currentTextChanged.connect(lambda text, name=dataset_item.name: self.set_y_axis(name, text.lower()))
        col_layout.addWidget(axis_combo, 1, 1)
        
        col_layout.addWidget(QLabel("Norm:"), 1, 2)
        norm_combo = QComboBox()
        norm_combo.addItems(["None", "Min-Max", "Z-Score", "Area", "Baseline", "Peak", "Vector"])
        norm_combo.setCurrentText(dataset_item.normalization)
        norm_combo.currentTextChanged.connect(lambda text, name=dataset_item.name: self.set_normalization(name, text))
        col_layout.addWidget(norm_combo, 1, 3)
        
        col_group.setLayout(col_layout)
        layout.addWidget(col_group)
        
        # Styling options
        style_group = QGroupBox("Style")
        style_layout = QGridLayout()
        style_layout.setSpacing(4)
        
        style_layout.addWidget(QLabel("Line:"), 0, 0)
        line_combo = QComboBox()
        line_combo.addItems(["solid", "dashed", "dotted", "dashdot", "none"])
        line_combo.setCurrentText(dataset_item.line_style)
        line_combo.currentTextChanged.connect(lambda text, name=dataset_item.name: self.set_line_style(name, text))
        style_layout.addWidget(line_combo, 0, 1)
        
        style_layout.addWidget(QLabel("Marker:"), 0, 2)
        marker_combo = QComboBox()
        marker_combo.addItems(["none", "circle", "square", "diamond", "triangle", "cross", "x"])
        marker_combo.setCurrentText(dataset_item.marker_style)
        marker_combo.currentTextChanged.connect(lambda text, name=dataset_item.name: self.set_marker_style(name, text))
        style_layout.addWidget(marker_combo, 0, 3)
        
        style_layout.addWidget(QLabel("Color:"), 1, 0)
        color_btn = QPushButton()
        color_btn.setFixedSize(40, 25)
        color_btn.setStyleSheet(f"background-color: {dataset_item.color}; border: 1px solid black; border-radius: 3px;")
        color_btn.clicked.connect(lambda checked, name=dataset_item.name: self.select_color(name, color_btn))
        style_layout.addWidget(color_btn, 1, 1)
        
        style_layout.addWidget(QLabel("Opacity:"), 1, 2)
        opacity_slider = QSlider(Qt.Horizontal)
        opacity_slider.setRange(10, 100)
        opacity_slider.setValue(int(dataset_item.opacity * 100))
        opacity_slider.valueChanged.connect(lambda val, name=dataset_item.name: self.set_opacity(name, val/100))
        style_layout.addWidget(opacity_slider, 1, 3)
        
        style_group.setLayout(style_layout)
        layout.addWidget(style_group)
        
        # Store references for removal
        frame.dataset_name = dataset_item.name
        frame.setLayout(layout)
        self.dataset_layout.addWidget(frame)
    
    def add_dataset(self, dataset_item):
        """Add a new dataset to the manager"""
        self.datasets[dataset_item.name] = dataset_item
        self.create_dataset_widget(dataset_item)
        self.update_dataset_count()
        self.update_button_states()
        self.dataset_changed.emit()
    
    def update_dataset_count(self):
        """Update the dataset count label"""
        count = len(self.datasets)
        self.dataset_count_label.setText(f"({count} dataset{'s' if count != 1 else ''})")
    
    def update_button_states(self):
        """Update button enabled states"""
        has_datasets = len(self.datasets) > 0
        self.remove_btn.setEnabled(has_datasets)
        self.clear_all_btn.setEnabled(has_datasets)
    
    def remove_selected_dataset(self):
        """Remove selected dataset (most recent)"""
        if self.datasets:
            last_name = list(self.datasets.keys())[-1]
            self.remove_dataset(last_name)
    
    def remove_dataset(self, name):
        """Remove specific dataset"""
        if name in self.datasets:
            del self.datasets[name]
            
            # Remove widget
            for i in range(self.dataset_layout.count()):
                widget = self.dataset_layout.itemAt(i).widget()
                if widget and hasattr(widget, 'dataset_name') and widget.dataset_name == name:
                    widget.deleteLater()
                    break
            
            self.update_dataset_count()
            self.update_button_states()
            self.dataset_changed.emit()
    
    def clear_all_datasets(self):
        """Clear all datasets"""
        self.datasets.clear()
        
        # Clear all widgets
        while self.dataset_layout.count():
            child = self.dataset_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        self.update_dataset_count()
        self.update_button_states()
        self.dataset_changed.emit()
    
    def get_datasets(self):
        """Get all datasets"""
        return self.datasets
    
    # All the setter methods remain the same as in the original DatasetManager
    def set_visibility(self, name, visible):
        if name in self.datasets:
            self.datasets[name].visible = visible
            self.dataset_changed.emit()
    
    def set_x_column(self, name, column):
        if name in self.datasets:
            self.datasets[name].x_column = column
            self.dataset_changed.emit()
    
    def set_y_column(self, name, column):
        if name in self.datasets:
            self.datasets[name].y_column = column
            self.dataset_changed.emit()
    
    def set_y_axis(self, name, axis):
        if name in self.datasets:
            self.datasets[name].y_axis = axis
            self.dataset_changed.emit()
    
    def set_line_style(self, name, style):
        if name in self.datasets:
            self.datasets[name].line_style = style
            self.dataset_changed.emit()
    
    def set_marker_style(self, name, style):
        if name in self.datasets:
            self.datasets[name].marker_style = style
            self.dataset_changed.emit()
    
    def set_normalization(self, name, norm_type):
        if name in self.datasets:
            self.datasets[name].normalization = norm_type
            self.dataset_changed.emit()
    
    def set_opacity(self, name, opacity):
        if name in self.datasets:
            self.datasets[name].opacity = opacity
            self.dataset_changed.emit()
    
    def select_color(self, name, button):
        """Open color picker"""
        color = QColorDialog.getColor()
        if color.isValid():
            color_name = color.name()
            self.datasets[name].color = color_name
            button.setStyleSheet(f"background-color: {color_name}; border: 1px solid black; border-radius: 3px;")
            self.dataset_changed.emit()





# Replace your PlotTab class with this updated version:

class PlotTab(QWidget):
    """Individual tab containing a plot and its enhanced tabbed control panel"""
    
    def __init__(self, tab_name="Analysis"):
        super().__init__()
        self.tab_name = tab_name
        self.dataset_counter = 0
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the UI for this tab"""
        # Create horizontal splitter for main layout
        main_splitter = QSplitter(Qt.Horizontal)
        
        # Create unified Plotly canvas
        self.plot_canvas = UnifiedPlotlyCanvas(self)
        main_splitter.addWidget(self.plot_canvas)
        
        # Create TABBED control panel (this is the key change)
        self.control_panel = TabbedControlPanel()
        self.control_panel.setMinimumWidth(450)  # Slightly wider for tabbed interface
        self.control_panel.setMaximumWidth(500)
        self.control_panel.plot_updated.connect(self.update_plot)
        self.control_panel.plot_type_changed.connect(self.on_plot_type_changed)
        
        main_splitter.addWidget(self.control_panel)
        
        # Set splitter sizes (plot area gets more space)
        main_splitter.setSizes([1100, 500])  # Adjusted for wider control panel
        main_splitter.setStretchFactor(0, 1)  # Plot area stretches
        main_splitter.setStretchFactor(1, 0)  # Control panel doesn't stretch
        
        # Create main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.addWidget(main_splitter)
        
        self.setLayout(main_layout)
    
    def load_data_file(self, file_path):
        """Load data file into this tab"""
        try:
            data, metadata = DataParser.load_data(file_path)
            
            # Create dataset item
            filename = os.path.basename(file_path)
            self.dataset_counter += 1
            dataset_name = f"{filename} ({self.dataset_counter})"
            
            # Assign alternating colors and properties
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
            line_styles = ['solid', 'dashed', 'dotted', 'dashdot']
            axes = ['left', 'right']
            
            dataset_item = DatasetItem(
                name=dataset_name,
                data=data,
                metadata=metadata,
                file_path=file_path
            )
            
            # Set intelligent defaults
            existing_count = len(self.control_panel.dataset_manager.get_datasets())
            dataset_item.color = colors[existing_count % len(colors)]
            dataset_item.line_style = line_styles[existing_count % len(line_styles)]
            dataset_item.y_axis = axes[existing_count % len(axes)]
            
            # Add to dataset manager (now in the Data tab)
            self.control_panel.dataset_manager.add_dataset(dataset_item)
            
            # Auto-plot
            self.update_plot()
            
            return True, f"Loaded: {filename} - {len(data)} rows, {len(data.columns)} columns"
            
        except Exception as e:
            return False, f"Failed to load file: {str(e)}"
    
    def on_plot_type_changed(self, plot_type):
        """Handle plot type change"""
        self.current_plot_type = plot_type
        self.update_plot()
    
    def update_plot(self):
        """Update plot with current settings"""
        datasets = self.control_panel.dataset_manager.get_datasets()
        plot_settings = self.control_panel.get_plot_settings()
        plot_type = self.control_panel.get_current_plot_type()
        
        self.plot_canvas.update_plot(datasets, plot_settings, plot_type)
    
    def clear_data(self):
        """Clear all data in this tab"""
        self.control_panel.dataset_manager.clear_all_datasets()
        self.plot_canvas.create_empty_plot()
    
    def get_datasets(self):
        """Get datasets in this tab"""
        return self.control_panel.dataset_manager.get_datasets()


class DataVisualizerGUI(QMainWindow):
    """Main application window with tab management"""
    
    def __init__(self):
        super().__init__()
        self.tab_counter = 0
        self.setup_ui()
        self.setup_drag_drop()
        
    def setup_ui(self):
        self.setWindowTitle("Advanced Scientific Data Visualizer - Multi-Tab Edition")
        self.setGeometry(100, 100, 1600, 1000)
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabsClosable(True)
        self.tab_widget.setMovable(True)
        self.tab_widget.tabCloseRequested.connect(self.close_tab)
        
        # Add initial tab
        self.add_new_tab("Analysis 1")
        
        # Create toolbar
        self.create_toolbar()
        
        # Create main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.addWidget(self.tab_widget)
        
        # Add status bar info
        self.status_label = QLabel("Ready - Drag and drop data files to begin | Multi-Tab Plotly Interface")
        self.status_label.setStyleSheet("QLabel { padding: 5px; background-color: #f0f0f0; }")
        main_layout.addWidget(self.status_label)
        
        main_widget.setLayout(main_layout)
    
    def create_toolbar(self):
        """Create toolbar with file operations and tab management"""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        
        # Tab operations
        new_tab_action = QAction("📄 New Tab", self)
        new_tab_action.setToolTip("Create a new analysis tab")
        new_tab_action.triggered.connect(self.add_new_tab_dialog)
        toolbar.addAction(new_tab_action)
        
        close_tab_action = QAction("❌ Close Tab", self)
        close_tab_action.setToolTip("Close the current tab")
        close_tab_action.triggered.connect(self.close_current_tab)
        toolbar.addAction(close_tab_action)
        
        toolbar.addSeparator()
        
        # File operations
        open_action = QAction("📁 Open File", self)
        open_action.setToolTip("Open a new data file in current tab")
        open_action.triggered.connect(self.open_file)
        toolbar.addAction(open_action)
        
        open_new_tab_action = QAction("📂 Open in New Tab", self)
        open_new_tab_action.setToolTip("Open a data file in a new tab")
        open_new_tab_action.triggered.connect(self.open_file_new_tab)
        toolbar.addAction(open_new_tab_action)
        
        add_file_action = QAction("➕ Add File", self)
        add_file_action.setToolTip("Add another data file to the current tab")
        add_file_action.triggered.connect(self.add_file)
        toolbar.addAction(add_file_action)
        
        toolbar.addSeparator()
        
        # Export operations
        save_action = QAction("💾 Save Plot", self)
        save_action.setToolTip("Save the current plot as an image")
        save_action.triggered.connect(self.save_plot)
        toolbar.addAction(save_action)
        
        export_html_action = QAction("🌐 Export HTML", self)
        export_html_action.setToolTip("Export interactive plot as HTML")
        export_html_action.triggered.connect(self.export_html)
        toolbar.addAction(export_html_action)
        
        export_data_action = QAction("📊 Export Data", self)
        export_data_action.setToolTip("Export processed data as CSV")
        export_data_action.triggered.connect(self.export_data)
        toolbar.addAction(export_data_action)
        
        toolbar.addSeparator()
        
        # Clear operations
        clear_tab_action = QAction("🗑️ Clear Tab", self)
        clear_tab_action.setToolTip("Clear all data in current tab")
        clear_tab_action.triggered.connect(self.clear_current_tab)
        toolbar.addAction(clear_tab_action)
        
        clear_all_action = QAction("🗑️ Clear All Tabs", self)
        clear_all_action.setToolTip("Clear all data in all tabs")
        clear_all_action.triggered.connect(self.clear_all_tabs)
        toolbar.addAction(clear_all_action)
    
    def setup_drag_drop(self):
        """Enable drag and drop functionality"""
        self.setAcceptDrops(True)
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter events"""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()
    
    def dropEvent(self, event: QDropEvent):
        """Handle drop events - load files into current tab"""
        urls = event.mimeData().urls()
        current_tab = self.get_current_tab()
        if current_tab:
            for url in urls:
                file_path = url.toLocalFile()
                success, message = current_tab.load_data_file(file_path)
                if success:
                    self.update_status(message)
                else:
                    QMessageBox.critical(self, "Error", message)
    
    def add_new_tab(self, tab_name=None):
        """Add a new analysis tab"""
        if tab_name is None:
            self.tab_counter += 1
            tab_name = f"Analysis {self.tab_counter}"
        
        new_tab = PlotTab(tab_name)
        index = self.tab_widget.addTab(new_tab, tab_name)
        self.tab_widget.setCurrentIndex(index)
        return new_tab
    
    def add_new_tab_dialog(self):
        """Add new tab with dialog for name"""
        from PySide6.QtWidgets import QInputDialog
        tab_name, ok = QInputDialog.getText(self, "New Tab", "Enter tab name:")
        if ok and tab_name:
            self.add_new_tab(tab_name)
        else:
            self.add_new_tab()
    
    def close_tab(self, index):
        """Close tab at specified index"""
        if self.tab_widget.count() > 1:
            widget = self.tab_widget.widget(index)
            self.tab_widget.removeTab(index)
            widget.deleteLater()
        else:
            # Don't close the last tab, just clear it
            current_tab = self.get_current_tab()
            if current_tab:
                current_tab.clear_data()
    
    def close_current_tab(self):
        """Close current tab"""
        current_index = self.tab_widget.currentIndex()
        self.close_tab(current_index)
    
    def get_current_tab(self):
        """Get the currently active tab"""
        return self.tab_widget.currentWidget()
    
    def open_file(self):
        """Open file in current tab (replaces current data)"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Data File", "", 
            "Data Files (*.txt *.csv *.dat *.xy *.asc *.xlsx);;All Files (*)")
        
        if file_path:
            current_tab = self.get_current_tab()
            if current_tab:
                current_tab.clear_data()  # Clear existing data
                success, message = current_tab.load_data_file(file_path)
                if success:
                    self.update_status(message)
                else:
                    QMessageBox.critical(self, "Error", message)
    
    def open_file_new_tab(self):
        """Open file in a new tab"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Data File in New Tab", "", 
            "Data Files (*.txt *.csv *.dat *.xy *.asc *.xlsx);;All Files (*)")
        
        if file_path:
            filename = os.path.basename(file_path)
            new_tab = self.add_new_tab(f"Analysis - {filename}")
            success, message = new_tab.load_data_file(file_path)
            if success:
                self.update_status(message)
            else:
                QMessageBox.critical(self, "Error", message)
    
    def add_file(self):
        """Add file to current tab (adds to existing data)"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Add Data File", "", 
            "Data Files (*.txt *.csv *.dat *.xy *.asc *.xlsx);;All Files (*)")
        
        if file_path:
            current_tab = self.get_current_tab()
            if current_tab:
                success, message = current_tab.load_data_file(file_path)
                if success:
                    self.update_status(message)
                else:
                    QMessageBox.critical(self, "Error", message)
    
    def save_plot(self):
        """Save the current plot"""
        current_tab = self.get_current_tab()
        if current_tab and current_tab.get_datasets():
            QMessageBox.information(self, "Save Plot", 
                                   "Use the camera icon in the plot toolbar to save as PNG.\n"
                                   "Or export as interactive HTML using the toolbar.")
        else:
            QMessageBox.warning(self, "Warning", "No data to save in current tab")
    
    def export_html(self):
        """Export current tab's plot as interactive HTML"""
        current_tab = self.get_current_tab()
        if not current_tab or not current_tab.get_datasets():
            QMessageBox.warning(self, "Warning", "No data to export in current tab")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export HTML", "", "HTML Files (*.html)")
        
        if file_path:
            try:
                QMessageBox.information(self, "Export HTML", 
                                       f"HTML export functionality can be implemented.\n"
                                       f"Would save current tab to: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export HTML:\n{str(e)}")
    
    def export_data(self):
        """Export current tab's processed data to CSV"""
        current_tab = self.get_current_tab()
        if not current_tab or not current_tab.get_datasets():
            QMessageBox.warning(self, "Warning", "No data to export in current tab")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Data", "", "CSV Files (*.csv)")
        
        if file_path:
            try:
                datasets = current_tab.get_datasets()
                combined_data = pd.DataFrame()
                
                for name, dataset_item in datasets.items():
                    if dataset_item.visible:
                        x_data = dataset_item.data.iloc[:, dataset_item.x_column]
                        y_data = dataset_item.data.iloc[:, dataset_item.y_column]
                        
                        # Apply normalization
                        if dataset_item.normalization != "None":
                            y_data = self.apply_normalization(x_data, y_data, dataset_item.normalization)
                        
                        combined_data[f"{name}_X"] = x_data
                        combined_data[f"{name}_Y"] = y_data
                
                combined_data.to_csv(file_path, index=False)
                QMessageBox.information(self, "Success", f"Data exported to {file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export data:\n{str(e)}")
    
    def apply_normalization(self, x_data, y_data, norm_type):
        """Apply normalization to data"""
        if norm_type == "Min-Max":
            return DataNormalizer.min_max_normalize(y_data)
        elif norm_type == "Z-Score":
            return DataNormalizer.z_score_normalize(y_data)
        elif norm_type == "Area":
            return DataNormalizer.area_normalize(x_data, y_data)
        elif norm_type == "Baseline":
            return DataNormalizer.baseline_correct(x_data, y_data)
        elif norm_type == "Peak":
            return DataNormalizer.peak_normalize(y_data)
        elif norm_type == "Vector":
            return DataNormalizer.vector_normalize(y_data)
        else:
            return y_data
    
    def clear_current_tab(self):
        """Clear all data in current tab"""
        current_tab = self.get_current_tab()
        if current_tab:
            current_tab.clear_data()
            self.update_status("Current tab cleared")
    
    def clear_all_tabs(self):
        """Clear all data in all tabs"""
        reply = QMessageBox.question(self, "Clear All Tabs", 
                                   "Are you sure you want to clear all data in all tabs?",
                                   QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            for i in range(self.tab_widget.count()):
                tab = self.tab_widget.widget(i)
                if tab:
                    tab.clear_data()
            self.update_status("All tabs cleared")
    
    def update_status(self, message):
        """Update status bar message"""
        total_tabs = self.tab_widget.count()
        current_index = self.tab_widget.currentIndex() + 1
        self.status_label.setText(f"{message} | Tab {current_index}/{total_tabs} | Multi-Tab Plotly Interface")


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Advanced Scientific Data Visualizer - Multi-Tab Edition")
    app.setOrganizationName("SciViz Pro")
    
    # Create and show main window
    window = DataVisualizerGUI()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()