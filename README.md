# SmartPlot - Scientific Data Visualization Tool

SmartPlot is a powerful, interactive data visualization application designed for scientific data analysis. Built with Python, PySide6, and Plotly, it provides an intuitive interface for exploring and visualizing complex datasets with various plot types and customization options.

## Features

- **Multiple Plot Types**: Line plots, scatter plots, box plots, violin plots, correlation matrices, and more
- **Interactive Visualization**: Zoom, pan, and hover tooltips for data exploration
- **Data Management**: Load and manage multiple datasets simultaneously
- **Advanced Styling**: Customize colors, line styles, markers, and more
- **Data Normalization**: Multiple normalization methods including min-max, z-score, and area normalization
- **Tabbed Interface**: Work with multiple visualizations in separate tabs
- **Drag and Drop**: Easily load data files by dragging and dropping them into the application
- **Export Options**: Save plots as PNG or export data as CSV

## Installation

1. **Prerequisites**:
   - Python 3.8 or higher
   - pip (Python package manager)

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   (Note: If you don't have a requirements.txt, install the following packages:)
   ```bash
   pip install numpy pandas PySide6 plotly
   ```

## Usage

1. **Launch the application**:
   ```bash
   python sci_data_viz_gui.py
   ```

2. **Loading Data**:
   - Use the "Open" button to load a data file
   - Or simply drag and drop files into the application window
   - Supported formats: CSV, TSV, TXT (with various delimiters)

3. **Creating Visualizations**:
   - Select the desired plot type from the dropdown
   - Customize the appearance using the control panel
   - Toggle datasets on/off using the dataset manager
   - Adjust axes, labels, and other plot properties

4. **Saving Your Work**:
   - Save plots as high-resolution PNG images
   - Export interactive HTML visualizations
   - Save processed data as CSV for further analysis

## Keyboard Shortcuts

- **Ctrl+O**: Open file
- **Ctrl+N**: New tab
- **Ctrl+W**: Close current tab
- **Ctrl+S**: Save plot
- **Ctrl+E**: Export data
- **Ctrl+Q**: Quit application

## License

This project is licensed under the terms of the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues or feature requests, please open an issue in the project repository.
