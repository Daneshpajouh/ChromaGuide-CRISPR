"""
Automated report generation for Overleaf paper.

Features:
- LaTeX report generation
- Automatic figure insertion
- Bibliography management
- Table generation from results
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import json
from datetime import datetime


@dataclass
class ReportSection:
    """Report section metadata."""
    title: str
    content: str
    figures: List[Path] = None
    tables: List[Dict] = None


class LatexReportGenerator:
    """Generate LaTeX/Overleaf compatible reports."""
    
    def __init__(self, title: str = "CRISPR Prediction Analysis"):
        self.title = title
        self.sections = []
        self.figures = []
        self.tables = []
        self.bibliography = []
    
    def create_document_header(self) -> str:
        """Create LaTeX document header."""
        header = r"""
\documentclass[12pt]{article}
\usepackage[utf8]{inputenc}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{booktabs}
\usepackage{float}
\usepackage{hyperref}
\usepackage{cite}

\title{""" + self.title + r"""}
\author{ChromaGuide Team}
\date{\today}

\begin{document}
\maketitle

\tableofcontents
\newpage
"""
        return header
    
    def add_section(self, section: ReportSection):
        """Add section to report."""
        self.sections.append(section)
    
    def create_section_latex(self, section: ReportSection) -> str:
        """Convert section to LaTeX."""
        latex = f"\n\\section{{{section.title}}}\n\n"
        latex += section.content + "\n\n"
        
        # Add figures
        if section.figures:
            for fig_path in section.figures:
                latex += f"""
\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.8\\textwidth]{{{fig_path}}}
\\caption{{{fig_path.stem}}}
\\end{{figure}}
"""
        
        # Add tables
        if section.tables:
            for table_data in section.tables:
                latex += self._create_table_latex(table_data)
        
        return latex
    
    def _create_table_latex(self, table_data: Dict) -> str:
        """Create LaTeX table from data."""
        latex = r"\begin{table}[H]" + "\n"
        latex += r"\centering" + "\n"
        
        rows = table_data.get('rows', [])
        cols = table_data.get('columns', [])
        
        # Table header
        latex += r"\begin{tabular}{" + "c" * len(cols) + "}\n"
        latex += r"\toprule" + "\n"
        latex += " & ".join(cols) + r" \\" + "\n"
        latex += r"\midrule" + "\n"
        
        # Table rows
        for row in rows:
            latex += " & ".join([str(v) for v in row.values()]) + r" \\" + "\n"
        
        latex += r"\bottomrule" + "\n"
        latex += r"\end{tabular}" + "\n"
        latex += r"\caption{" + table_data.get('caption', 'Table') + "}\n"
        latex += r"\end{table}" + "\n\n"
        
        return latex
    
    def generate_document(self) -> str:
        """Generate complete LaTeX document."""
        doc = self.create_document_header()
        
        for section in self.sections:
            doc += self.create_section_latex(section)
        
        doc += "\n\\end{document}\n"
        
        return doc
    
    def save_document(self, filepath: Path):
        """Save LaTeX document."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        doc = self.generate_document()
        with open(filepath, 'w') as f:
            f.write(doc)


class ResultsTableGenerator:
    """Generate tables from results."""
    
    @staticmethod
    def create_metrics_table(results: Dict[str, Dict]) -> Dict:
        """Create metrics comparison table."""
        rows = []
        columns = ['Model', 'Accuracy', 'F1-Score', 'Precision', 'Recall']
        
        for model, metrics in results.items():
            row = {
                'Model': model,
                'Accuracy': f"{metrics.get('accuracy', 0):.4f}",
                'F1-Score': f"{metrics.get('f1', 0):.4f}",
                'Precision': f"{metrics.get('precision', 0):.4f}",
                'Recall': f"{metrics.get('recall', 0):.4f}"
            }
            rows.append(row)
        
        return {
            'columns': columns,
            'rows': rows,
            'caption': 'Model Performance Metrics'
        }
    
    @staticmethod
    def create_benchmark_table(benchmarks: List[Dict]) -> Dict:
        """Create benchmarking table."""
        rows = []
        columns = ['Model', 'Inference Time (ms)', 'Memory (MB)', 'Throughput (samples/s)']
        
        for bench in benchmarks:
            row = {
                'Model': bench.get('model', 'N/A'),
                'Inference Time (ms)': f"{bench.get('inference_time', 0):.2f}",
                'Memory (MB)': f"{bench.get('memory', 0):.2f}",
                'Throughput (samples/s)': f"{bench.get('throughput', 0):.0f}"
            }
            rows.append(row)
        
        return {
            'columns': columns,
            'rows': rows,
            'caption': 'Model Benchmarking Results'
        }
    
    @staticmethod
    def create_ablation_table(ablation_results: Dict[str, float]) -> Dict:
        """Create ablation study table."""
        rows = []
        columns = ['Component', 'Accuracy Drop', 'Impact']
        
        for component, drop in ablation_results.items():
            impact = "High" if drop > 0.05 else "Medium" if drop > 0.02 else "Low"
            row = {
                'Component': component,
                'Accuracy Drop': f"{drop:.4f}",
                'Impact': impact
            }
            rows.append(row)
        
        return {
            'columns': columns,
            'rows': rows,
            'caption': 'Ablation Study Results'
        }


class AutomatedReportBuilder:
    """Build automated reports from pipeline results."""
    
    def __init__(self, results_dir: Path = Path("results")):
        self.results_dir = Path(results_dir)
        self.generator = LatexReportGenerator()
    
    def build_methods_section(self, methods: Dict) -> ReportSection:
        """Build methods section."""
        content = "\\subsection{Data}\n\n"
        content += methods.get('data_description', 'Dataset description goes here.\n\n')
        
        content += "\\subsection{Model Architecture}\n\n"
        content += methods.get('model_description', 'Model description goes here.\n\n')
        
        content += "\\subsection{Training Procedure}\n\n"
        content += methods.get('training_description', 'Training details go here.\n\n')
        
        return ReportSection(
            title='Methods',
            content=content
        )
    
    def build_results_section(self, results: Dict) -> ReportSection:
        """Build results section."""
        content = "\\subsection{Performance Metrics}\n\n"
        content += "Our model achieved the following metrics:\n\n"
        
        for metric, value in results.items():
            content += f"- {metric}: {value:.4f}\n"
        
        table = ResultsTableGenerator.create_metrics_table({'Our Model': results})
        
        return ReportSection(
            title='Results',
            content=content,
            tables=[table]
        )
    
    def build_conclusion_section(self, conclusion: str) -> ReportSection:
        """Build conclusion section."""
        return ReportSection(
            title='Conclusion',
            content=conclusion
        )
    
    def generate_full_report(
        self,
        methods: Dict,
        results: Dict,
        conclusion: str,
        output_path: Path
    ):
        """Generate complete report."""
        self.generator.add_section(self.build_methods_section(methods))
        self.generator.add_section(self.build_results_section(results))
        self.generator.add_section(self.build_conclusion_section(conclusion))
        
        self.generator.save_document(output_path)
        
        return output_path


class OverleafIntegrator:
    """Integrate with Overleaf project."""
    
    def __init__(self, overleaf_token: Optional[str] = None):
        self.overleaf_token = overleaf_token
        self.api_url = "https://www.overleaf.com/api/v0"
    
    def get_project_files(self, project_id: str) -> List[Dict]:
        """Get project files from Overleaf."""
        try:
            import requests
            
            headers = {'Authorization': f'Bearer {self.overleaf_token}'}
            response = requests.get(
                f"{self.api_url}/projects/{project_id}",
                headers=headers
            )
            
            if response.status_code == 200:
                return response.json().get('rootFolder/docs', [])
        except Exception as e:
            print(f"Failed to fetch Overleaf files: {e}")
        
        return []
    
    def upload_file(
        self,
        project_id: str,
        file_path: Path,
        overleaf_path: str
    ) -> bool:
        """Upload file to Overleaf."""
        try:
            import requests
            
            with open(file_path, 'rb') as f:
                files = {'file': f}
                headers = {'Authorization': f'Bearer {self.overleaf_token}'}
                
                response = requests.post(
                    f"{self.api_url}/projects/{project_id}/upload",
                    files=files,
                    headers=headers,
                    data={'filePath': overleaf_path}
                )
            
            return response.status_code == 200
        except Exception as e:
            print(f"Failed to upload to Overleaf: {e}")
            return False
    
    def recompile_project(self, project_id: str) -> bool:
        """Trigger project recompilation."""
        try:
            import requests
            
            headers = {'Authorization': f'Bearer {self.overleaf_token}'}
            response = requests.post(
                f"{self.api_url}/projects/{project_id}/compiler",
                headers=headers,
                json={'rootResourceId': 'main.tex'}
            )
            
            return response.status_code == 200
        except Exception as e:
            print(f"Failed to recompile project: {e}")
            return False


class ReportScheduler:
    """Schedule automatic report generation."""
    
    def __init__(self, schedule_interval: int = 3600):
        self.schedule_interval = schedule_interval
        self.last_report_time = None
    
    def should_generate_report(self) -> bool:
        """Check if report should be generated."""
        import time
        
        current_time = time.time()
        if self.last_report_time is None:
            return True
        
        return (current_time - self.last_report_time) >= self.schedule_interval
    
    def generate_scheduled_report(
        self,
        builder: AutomatedReportBuilder,
        methods: Dict,
        results: Dict,
        conclusion: str,
        output_path: Path
    ):
        """Generate report on schedule."""
        if self.should_generate_report():
            builder.generate_full_report(methods, results, conclusion, output_path)
            import time
            self.last_report_time = time.time()
            return output_path
        
        return None
