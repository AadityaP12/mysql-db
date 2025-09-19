import pandas as pd
import pdfplumber
from pathlib import Path
import logging
from typing import List, Dict, Any
import numpy as np

# Try to import camelot with proper error handling
try:
    import camelot
except ImportError:
    try:
        import camelot.io as camelot
    except ImportError:
        print("Camelot not available. Install with: pip install 'camelot-py[base]'")
        camelot = None

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WaterQualityPDFExtractor:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.all_data = []
        
    def extract_with_camelot(self, pages: str = "all") -> List[pd.DataFrame]:
        """
        Extract tables using Camelot - best for well-structured tables
        """
        if camelot is None:
            logger.warning("Camelot not available, skipping camelot extraction")
            return []
            
        logger.info("Extracting tables with Camelot...")
        
        try:
            # Try lattice method first (better for tables with clear borders)
            tables = camelot.read_pdf(
                self.pdf_path,
                pages=pages,
                flavor='lattice',  # Use 'stream' if lattice fails
                line_scale=40,     # Adjust for better line detection
                split_text=True,   # Split text in cells
                strip_text='\n'    # Clean whitespace
            )
            
            logger.info(f"Found {len(tables)} tables with lattice method")
            return [table.df for table in tables]
            
        except Exception as e:
            logger.warning(f"Lattice method failed: {e}")
            
            # Fallback to stream method
            try:
                tables = camelot.read_pdf(
                    self.pdf_path,
                    pages=pages,
                    flavor='stream',
                    edge_tol=500,      # Tolerance for detecting table edges
                    row_tol=10,        # Tolerance for detecting rows
                    column_tol=10      # Tolerance for detecting columns
                )
                
                logger.info(f"Found {len(tables)} tables with stream method")
                return [table.df for table in tables]
                
            except Exception as e2:
                logger.error(f"Both Camelot methods failed: {e2}")
                return []
    
    def extract_with_pdfplumber(self, pages: List[int] = None) -> List[pd.DataFrame]:
        """
        Extract tables using pdfplumber - better for complex layouts
        """
        logger.info("Extracting tables with pdfplumber...")
        
        tables = []
        
        with pdfplumber.open(self.pdf_path) as pdf:
            pages_to_process = pages if pages else range(len(pdf.pages))
            
            for page_num in pages_to_process:
                if page_num >= len(pdf.pages):
                    continue
                    
                page = pdf.pages[page_num]
                
                # Extract tables from the page
                page_tables = page.extract_tables(
                    table_settings={
                        "vertical_strategy": "lines",
                        "horizontal_strategy": "lines",
                        "snap_tolerance": 3,
                        "join_tolerance": 3,
                        "edge_min_length": 3,
                        "min_words_vertical": 3,
                        "min_words_horizontal": 1,
                    }
                )
                
                for table_data in page_tables:
                    if table_data and len(table_data) > 1:  # Skip empty or single-row tables
                        df = pd.DataFrame(table_data[1:], columns=table_data[0])
                        df['source_page'] = page_num + 1
                        tables.append(df)
        
        logger.info(f"Found {len(tables)} tables with pdfplumber")
        return tables
    
    def clean_and_structure_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and structure the extracted data for water quality tables
        """
        # Create proper column names for nested structure
        if len(df.columns) >= 15:  # Expected number of columns based on your image
            
            # Define the expected column structure
            column_mapping = {
                0: 'Station_Code',
                1: 'Monitoring_Location', 
                2: 'State',
                3: 'Temperature_Min_C',
                4: 'Temperature_Max_C',
                5: 'Dissolved_Oxygen_Min_mgL',
                6: 'Dissolved_Oxygen_Max_mgL',
                7: 'pH_Min',
                8: 'pH_Max',
                9: 'Conductivity_Min_umho_cm',
                10: 'Conductivity_Max_umho_cm',
                11: 'BOD_Min_mgL',
                12: 'BOD_Max_mgL',
                13: 'Nitrate_N_Min_mgL',
                14: 'Nitrate_N_Max_mgL',
                15: 'Fecal_Coliform_Min_MPN_100ml',
                16: 'Fecal_Coliform_Max_MPN_100ml',
                17: 'Total_Coliform_Min_MPN_100ml',
                18: 'Total_Coliform_Max_MPN_100ml',
                19: 'Fecal_Streptococci_Min_MPN_100ml',
                20: 'Fecal_Streptococci_Max_MPN_100ml'
            }
            
            # Rename columns based on position
            new_columns = []
            for i, col in enumerate(df.columns):
                if i in column_mapping:
                    new_columns.append(column_mapping[i])
                else:
                    new_columns.append(f'Column_{i}')
            
            df.columns = new_columns[:len(df.columns)]
        
        # Clean data
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.strip()
                # Replace common OCR artifacts
                df[col] = df[col].str.replace('|', 'I', regex=False)
                df[col] = df[col].str.replace('O', '0', regex=False)  # Only if it's clearly a zero
                
        # Convert numeric columns
        numeric_columns = [col for col in df.columns if any(keyword in col.lower() 
                          for keyword in ['min', 'max', 'temperature', 'ph', 'conductivity', 'bod', 'nitrate'])]
        
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove completely empty rows
        df = df.dropna(how='all')
        
        # Remove rows that are likely headers repeated in the middle of data
        df = df[~df['Station_Code'].str.contains('Station|Code', case=False, na=False)]
        
        return df
    
    def extract_all_tables(self, method: str = "both") -> pd.DataFrame:
        """
        Extract all tables from the PDF and combine them
        
        Args:
            method: "camelot", "pdfplumber", or "both"
        """
        all_dataframes = []
        
        if method in ["camelot", "both"]:
            camelot_dfs = self.extract_with_camelot()
            for i, df in enumerate(camelot_dfs):
                if not df.empty:
                    cleaned_df = self.clean_and_structure_data(df)
                    cleaned_df['extraction_method'] = 'camelot'
                    cleaned_df['table_index'] = i
                    all_dataframes.append(cleaned_df)
        
        if method in ["pdfplumber", "both"]:
            pdfplumber_dfs = self.extract_with_pdfplumber()
            for i, df in enumerate(pdfplumber_dfs):
                if not df.empty:
                    cleaned_df = self.clean_and_structure_data(df)
                    cleaned_df['extraction_method'] = 'pdfplumber'
                    cleaned_df['table_index'] = i
                    all_dataframes.append(cleaned_df)
        
        if not all_dataframes:
            logger.error("No tables extracted successfully")
            return pd.DataFrame()
        
        # Combine all dataframes
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        
        # Remove duplicates (in case both methods extracted the same data)
        if method == "both":
            combined_df = combined_df.drop_duplicates(
                subset=['Station_Code', 'Monitoring_Location'], 
                keep='first'
            )
        
        logger.info(f"Total records extracted: {len(combined_df)}")
        return combined_df
    
    def save_to_csv(self, df: pd.DataFrame, output_path: str = None):
        """Save the extracted data to CSV"""
        if output_path is None:
            output_path = str(Path(self.pdf_path).with_suffix('.csv'))
        
        df.to_csv(output_path, index=False)
        logger.info(f"Data saved to: {output_path}")
        
        # Save a summary report
        summary_path = str(Path(output_path).with_suffix('.summary.txt'))
        with open(summary_path, 'w') as f:
            f.write(f"PDF Extraction Summary\n")
            f.write(f"====================\n")
            f.write(f"Source PDF: {self.pdf_path}\n")
            f.write(f"Total records: {len(df)}\n")
            f.write(f"Columns: {len(df.columns)}\n")
            f.write(f"Column names: {', '.join(df.columns[:10])}...\n")
            f.write(f"\nFirst few rows:\n")
            f.write(df.head().to_string())
        
        logger.info(f"Summary saved to: {summary_path}")

# Usage example
def main():
    # Initialize extractor
    pdf_path = "C:/SIH/WQuality_River_Data_2023.pdf"  # Replace with your PDF path
    extractor = WaterQualityPDFExtractor(pdf_path)
    
    # Extract tables using both methods and compare
    logger.info("Starting PDF extraction...")
    
    # Try both methods
    combined_data = extractor.extract_all_tables(method="both")
    
    if not combined_data.empty:
        # Save to CSV
        extractor.save_to_csv(combined_data, "C:/SIH/WQuality_River_Data_2023.csv")
        
        # Display summary
        print(f"\nExtraction Summary:")
        print(f"Total records: {len(combined_data)}")
        print(f"Columns: {list(combined_data.columns)}")
        print(f"\nSample data:")
        print(combined_data.head())
        
        # Check for data quality
        print(f"\nData Quality Check:")
        print(f"Missing values per column:")
        print(combined_data.isnull().sum())
        
    else:
        print("No data extracted. Try adjusting extraction parameters.")

if __name__ == "__main__":
    main()


# Alternative: Simple pdfplumber-only extraction (RECOMMENDED for now)
def extract_water_quality_tables_simple(pdf_path: str, output_csv: str = "water_quality_data.csv"):
    """
    Simple and reliable extraction using only pdfplumber
    Recommended for immediate use
    """
    print("Extracting tables with pdfplumber (simple method)...")
    
    all_data = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            print(f"Processing page {page_num + 1}/{len(pdf.pages)}")
            
            # Extract tables from the page
            tables = page.extract_tables()
            
            for table_idx, table in enumerate(tables):
                if not table or len(table) < 2:
                    continue
                
                # Convert to DataFrame
                headers = table[0]
                data_rows = table[1:]
                
                # Create DataFrame
                df = pd.DataFrame(data_rows, columns=range(len(headers)))
                
                # Add metadata
                df['source_page'] = page_num + 1
                df['table_on_page'] = table_idx + 1
                
                # Basic cleaning
                for col in df.columns:
                    if df[col].dtype == 'object':
                        df[col] = df[col].astype(str).str.strip()
                        df[col] = df[col].replace('nan', '')
                        df[col] = df[col].replace('None', '')
                
                # Remove empty rows
                df = df.dropna(how='all')
                
                if len(df) > 0:
                    all_data.append(df)
    
    if all_data:
        # Combine all data
        combined = pd.concat(all_data, ignore_index=True)
        
        # Basic column naming (you'll need to adjust based on your specific PDF)
        if len(combined.columns) >= 20:  # Assuming 20+ columns based on your image
            column_names = [
                'Station_Code', 'Monitoring_Location', 'State',
                'Temp_Min', 'Temp_Max', 'DO_Min', 'DO_Max',
                'pH_Min', 'pH_Max', 'Conductivity_Min', 'Conductivity_Max',
                'BOD_Min', 'BOD_Max', 'Nitrate_Min', 'Nitrate_Max',
                'Fecal_Coliform_Min', 'Fecal_Coliform_Max',
                'Total_Coliform_Min', 'Total_Coliform_Max',
                'Fecal_Strep_Min', 'Fecal_Strep_Max'
            ]
            
            # Apply column names (adjust as needed)
            for i, name in enumerate(column_names):
                if i < len(combined.columns) - 2:  # -2 for metadata columns
                    combined.rename(columns={i: name}, inplace=True)
        
        # Save to CSV
        combined.to_csv(output_csv, index=False)
        print(f"Successfully extracted {len(combined)} rows to {output_csv}")
        
        # Show sample
        print("\nFirst few rows:")
        print(combined.head())
        
        # Show column info
        print(f"\nColumns ({len(combined.columns)}):")
        print(combined.columns.tolist())
        
        return combined
    else:
        print("No tables found")
        return pd.DataFrame()


# Even simpler: Raw text extraction
def extract_with_raw_text(pdf_path: str, output_csv: str = "raw_extracted.csv"):
    """
    Extract raw text and attempt to parse it
    Use this if table extraction fails
    """
    all_text_data = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                lines = text.split('\n')
                for line in lines:
                    # Look for lines that might be table rows
                    if any(keyword in line.lower() for keyword in ['river', 'beas', 'himachal']):
                        # Split by multiple spaces (common in PDFs)
                        parts = [part.strip() for part in line.split('  ') if part.strip()]
                        if len(parts) > 5:  # Likely a data row
                            row_dict = {f'col_{i}': part for i, part in enumerate(parts)}
                            row_dict['source_page'] = page_num + 1
                            all_text_data.append(row_dict)
    
    if all_text_data:
        df = pd.DataFrame(all_text_data)
        df.to_csv(output_csv, index=False)
        print(f"Raw text extraction: {len(df)} rows saved to {output_csv}")
        return df
    else:
        print("No text data found")
        return pd.DataFrame()